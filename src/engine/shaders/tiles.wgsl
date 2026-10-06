// Tiles: the sparse grid's directory (TileDirectory.ts). A tile is 4x4x4 bricks at integer
// coordinates of the world lattice: brick `b` lies in tile `b >> 2`. Each tile owns a tile
// slot; a brick's id is its tile slot * 64 plus its place in the tile, and indexes every
// per-brick array. The directory is read through integer textures, which leave a kernel's
// storage buffers to its fields: WebGPU allows only eight per stage by default.
// Every tile slot's record, a row of 8 texels: the tile's coordinates and state (0 free,
// 1 live, 2 closing: its bricks give up their slots this step), then its 27 neighbors'
// slots, -1 where there is no tile.
@group(0) @binding(63) var tileRecords: texture_2d<i32>;
// Tile coordinates hashed with open addressing, 256 entries a row: x, y, z and the tile
// slot, -1 when empty.
@group(0) @binding(64) var tileTable: texture_2d<i32>;
// Each brick id's entry, 64 a row (a tile slot's bricks): its page (pool.wgsl, 24 bits)
// with a flag for a slot given this step, or -1 without a slot. A brick computes exactly
// while it has a slot. Written by every step's allocation.
@group(0) @binding(65) var brickEntries: texture_2d<i32>;
// Each brick id's record, 7 texels per brick and a tile slot's bricks per row: its
// 27 neighbor entries and whether all its velocity cells are pressure unknowns. Built
// after allocation for every slotted brick; a released brick keeps its last record.
const BRICK_RECORD_TEXELS: u32 = 7u;
const BRICK_PRESSURE_FLAG: u32 = 27u;
@group(0) @binding(66) var brickRecords: texture_2d<i32>;
// A slot given this step: its apron does not copy its neighbors yet (fluid-zeroing.wgsl).
const FRESH_ENTRY: i32 = 0x2000000;
fn entryPage(entry: i32) -> i32 {
  return select(entry & 0xffffff, -1, entry < 0);
}

fn entryComputes(entry: i32) -> bool {
  return entry >= 0;
}

fn entryFresh(entry: i32) -> bool {
  return entry >= 0 && (entry & FRESH_ENTRY) != 0;
}

// The place of offset `d`, each axis -1 to 1, in a table of 27 neighbors.
fn neighborIndex(d: vec3i) -> u32 {
  return u32((d.z + 1) * 9 + (d.y + 1) * 3 + d.x + 1);
}

// Coordinates and state of the tile in slot `tile`.
fn tileRecord(tile: u32) -> vec4i {
  return textureLoad(tileRecords, vec2u(0u, tile), 0);
}

// The slot of neighbor `n` of the tile in slot `tile`.
fn tileNeighbor(tile: u32, n: u32) -> i32 {
  return textureLoad(tileRecords, vec2u(1u + n / 4u, tile), 0)[n % 4u];
}

fn tileHash(t: vec3i) -> u32 {
  let c = bitcast<vec3u>(t);
  return (c.x * 73856093u) ^ (c.y * 19349663u) ^ (c.z * 83492791u);
}

// The slot of the tile at `t`, or -1. The table is at most half full.
fn findTile(t: vec3i) -> i32 {
  let size = textureDimensions(tileTable);
  let mask = size.x * size.y - 1u;
  var h = tileHash(t) & mask;
  for (var probe = 0u; probe <= mask; probe++) {
    let entry = textureLoad(tileTable, vec2u(h & 255u, h >> 8u), 0);
    if entry.w < 0 {
      break;
    }
    if all(entry.xyz == t) {
      return entry.w;
    }
    h = (h + 1u) & mask;
  }
  return -1;
}

fn brickLocal(b: vec3i) -> u32 {
  let l = vec3u(b & vec3i(3));
  return (l.z * 4u + l.y) * 4u + l.x;
}

// The world brick of brick id `id`.
fn brickOf(id: u32) -> vec3i {
  let l = id & 63u;
  return tileRecord(id >> 6u).xyz * 4 + vec3i(vec3u(l & 3u, (l >> 2u) & 3u, l >> 4u));
}

fn entryById(id: u32) -> i32 {
  return textureLoad(brickEntries, vec2u(id & 63u, id >> 6u), 0).x;
}

// Word `k` of brick id `id`'s record.
fn recordWord(id: u32, k: u32) -> i32 {
  return textureLoad(brickRecords,
    vec2u((id & 63u) * BRICK_RECORD_TEXELS + k / 4u, id >> 6u),
    0)[k % 4u];
}

// Where an invocation works: a brick and its tile. Lookups within a brick of the home brick
// read its record, and lookups within a tile of the home tile read the tile's neighbors;
// the rest search the directory. `id` is -1 without a home brick, `tile` -1 without a home.
// `inWorkgroup` when the workgroup holds the home brick's record in `homeRecord`.
struct Home {
  brick: vec3i,
  id: i32,
  tileCoord: vec3i,
  tile: i32,
  inWorkgroup: bool
};

var<private> home: Home = Home(vec3i(0), -1, vec3i(0), -1, false);
// The home brick's record, loaded once by a workgroup that lies inside one brick.
var<workgroup> homeRecord: array<i32, 28>;
// Make brick id `id` home; `recorded` when its record describes its neighbors.
fn enterBrick(id: u32, recorded: bool) {
  let tile = id >> 6u;
  home = Home(brickOf(id), select(-1, i32(id), recorded), tileRecord(tile).xyz, i32(tile), false);
}

// Make tile slot `tile` home, without a home brick.
fn enterTile(tile: u32) {
  home = Home(vec3i(0), -1, tileRecord(tile).xyz, i32(tile), false);
}

// Load brick id `id`'s record into `homeRecord`: lane `lane` of the workgroup loads its
// texel, if `load`. Every lane must call this, then may read it.
fn loadHomeRecord(id: u32, lane: u32, load: bool) {
  if load && lane < 7u {
    let words = textureLoad(brickRecords,
      vec2u((id & 63u) * BRICK_RECORD_TEXELS + lane, id >> 6u),
      0);
    for (var k = 0u; k < 4u; k++) {
      homeRecord[lane * 4u + k] = words[k];
    }
  }
  workgroupBarrier();
}

// The slot of the tile at `t`, or -1.
fn tileAt(t: vec3i) -> i32 {
  if home.tile >= 0 {
    let d = t - home.tileCoord;
    if all(d == vec3i(0)) {
      return home.tile;
    }
    if all(abs(d) <= vec3i(1)) {
      return tileNeighbor(u32(home.tile), neighborIndex(d));
    }
  }
  return findTile(t);
}

// The id of world brick `b`, or -1 outside every tile.
fn brickId(b: vec3i) -> i32 {
  let tile = tileAt(b >> vec3u(2u));
  if tile < 0 {
    return -1;
  }
  return tile * 64 + i32(brickLocal(b));
}

// The entry of world brick `b`.
fn brickEntry(b: vec3i) -> i32 {
  if home.id >= 0 {
    let d = b - home.brick;
    if all(abs(d) <= vec3i(1)) {
      if home.inWorkgroup {
        return homeRecord[neighborIndex(d)];
      }
      return recordWord(u32(home.id), neighborIndex(d));
    }
  }
  let id = brickId(b);
  if id < 0 {
    return -1;
  }
  return entryById(u32(id));
}

// The page of world brick `b`, or -1 without a slot.
fn brickPage(b: vec3i) -> i32 {
  return entryPage(brickEntry(b));
}
