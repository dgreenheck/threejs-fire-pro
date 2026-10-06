// Deterministic slot allocation, neighbor records, collider masks, and dispatch lists.
// Assembled with the other fluid modules by ShaderSources.ts.
// A brick holds a slot while it computes: when occupied cells request it for their
// stencil/motion halo, or an emitter requests it. Mesh emitters retain a full-brick
// margin. A slot given this step starts as still, empty
// air, and its apron does not copy its neighbors until they next write: for that step its
// entry is marked fresh (tiles.wgsl), and samples reaching its apron read cell by cell.
// Bricks of closing tiles give up their slots, and every brick of a held tile computes.
// Flags: bit 0 makes every brick compute, bit 1 treats every brick as computing last
// step, so all others are zeroed.
@group(0) @binding(41) var<storage, read_write> activeBricks: array<atomic<u32>>;
@group(0) @binding(43) var<storage, read_write> dispatchArgs: array<u32>;
// Each brick id's page, or -1, which the allocator keeps; it publishes them with whether
// each brick computes as the directory's entries (tiles.wgsl).
@group(0) @binding(45) var<storage, read_write> pagesOut: array<i32>;
@group(0) @binding(69) var entriesOut: texture_storage_2d<r32sint, write>;
// A stack of free slots: `free` counts them, and `failures` counts the slots the pool
// could not grant since the host last read it.
@group(0) @binding(46) var<storage, read_write> freeSlots: array<u32>;
struct PoolState {
  free: atomic<i32>,
  failures: atomic<u32>,
  reserved: vec2u
};

@group(0) @binding(47) var<storage, read_write> poolState: PoolState;
// Bricks whose slots are zeroed this step: a count, then each brick id and its page.
// FRESH_SLOT marks a slot just allocated, whose apron is zeroed too.
@group(0) @binding(48) var<storage, read_write> zeroBricks: array<atomic<u32>>;
const FRESH_SLOT: u32 = 0x80000000u;
// Slots are freed and taken in brick id order (tile slot, then brick), so a run hands out
// the same slots every time: filtered samples round a little differently in different
// slots. Each block of bricks counts the slots its bricks free and request. A block's
// first ranks are the sums of the counts before it.
const RANK_BLOCK: u32 = 256u;
@group(0) @binding(57) var<storage, read_write> blockRanks: array<vec2u>;
var<workgroup> rankScan: array<vec2u, 256>;
// Brick state: bit 0 computes, 2 computed last step, 3 frees its slot, 4 requests one;
// bits 8 to 15 and 16 to 23 rank the free and the request in its block.
const BRICK_FREES: u32 = 8u;
const BRICK_REQUESTS: u32 = 16u;
fn brickCount() -> u32 {
  return u32(u.bricks.x);
}

fn rankBlocks() -> u32 {
  return (brickCount() + RANK_BLOCK - 1u) / RANK_BLOCK;
}

struct WorkgroupScan {
  before: vec2u,
  total: vec2u
};

// The sum of `value` over the lanes before this one, and over all of them. Every lane of
// a 256-lane workgroup must call it.
fn scanWorkgroup(value: vec2u, lane: u32) -> WorkgroupScan {
  rankScan[lane] = value;
  workgroupBarrier();
  for (var offset = 1u; offset < RANK_BLOCK; offset <<= 1u) {
    var sum = rankScan[lane];
    if lane >= offset {
      sum += rankScan[lane - offset];
    }
    workgroupBarrier();
    rankScan[lane] = sum;
    workgroupBarrier();
  }
  let total = rankScan[RANK_BLOCK - 1u];
  let result = WorkgroupScan(rankScan[lane] - value, total);
  workgroupBarrier();
  return result;
}

// A brick's rank among the frees (x) or requests (y) of the step.
fn brickRank(index: u32, state: u32) -> vec2u {
  return vec2u((state >> 8u) & 255u, (state >> 16u) & 255u) + blocksBefore(index / RANK_BLOCK);
}

// The frees (x) and requests (y) of the blocks before `block`; of them all at rankBlocks().
fn blocksBefore(block: u32) -> vec2u {
  var sum = vec2u(0);
  for (var i = 0u; i < block; i++) {
    sum += blockRanks[i];
  }
  return sum;
}

// Source coverage with the same stencil/motion margin used for occupied cells.
fn sourceNear(b: vec3i) -> bool {
  let size = f32(1u << u32(u.pool.w)) * u.fieldGrid.xyz;
  let lo = vec3f(b) * size;
  let hi = (vec3f(b) + 1.0) * size;
  let list = tileList(EMITTERS);
  for (var i = 0u; i < list.y; i++) {
    let e = emitters[tileLists[list.x + i]];
    if e.position.w < .5 {
      continue;
    }
    let motion = abs(e.velocity.xyz) + abs(e.velocity.w);
    let pad = min(size, 2.0 * u.velocityGrid.xyz + motion * u.grid.w);
    let extent = max(e.size.xyz, vec3f(.03)) + pad;
    if all(e.position.xyz + extent > lo) && all(e.position.xyz - extent < hi) {
      return true;
    }
  }
  return false;
}

fn zeroBrick(id: u32, payload: u32) {
  let i = atomicAdd(&zeroBricks[0], 1u);
  atomicStore(&zeroBricks[1u + 2u * i], id);
  atomicStore(&zeroBricks[2u + 2u * i], payload);
}

// The page of free-list slot `slot` (pool.wgsl).
fn slotPage(slot: u32) -> i32 {
  let row = u32(u.pool.x);
  let rows = u32(u.pool.y);
  return i32((slot % row) | (((slot / row) % rows) << 8u) | ((slot / (row * rows)) << 16u));
}

// Free slots the voxel budget holds back: the bottom of the stack.
fn heldSlots() -> i32 {
  return i32(u.pool.z);
}

// Decides each brick's state, and ranks the slots bricks free and request.
@compute @workgroup_size(256)
fn buildBricks(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3u
) {
  let index = gid.x;
  var state = 0u;
  if index < brickCount() && tileRecord(index >> 6u).w != 0 {
    enterBrick(index, false);
    let b = home.brick;
    let open = tileRecord(index >> 6u).w != 2;
    var listed = false;
    for (var z = -1; z <= 1 && open && !listed; z++) {
      for (var y = -1; y <= 1 && !listed; y++) {
        for (var x = -1; x <= 1 && !listed; x++) {
          let neighbor = brickId(b + vec3i(x, y, z));
          if neighbor < 0 {
            continue;
          }
          let toward = neighborIndex(-vec3i(x, y, z));
          listed = ((atomicLoad(&activity[neighbor]) >> 3u) & (1u << toward)) != 0u || meshTable(neighbor) >= 0;
        }
      }
    }
    if open && !listed {
      listed = sourceNear(b);
    }
    let page = pagesOut[index];
    let wasListed = (brickState[index] & 1u) != 0u;
    state = u32(listed) | (u32(wasListed) << 2u);
    if !listed && page >= 0 {
      state |= BRICK_FREES;
    }
    if listed && page < 0 {
      state |= BRICK_REQUESTS;
    }
  }
  let ranks = scanWorkgroup(vec2u(u32((state & BRICK_FREES) != 0u), u32((state & BRICK_REQUESTS) != 0u)),
    lane);
  if lane == 0u {
    blockRanks[group.x] = ranks.total;
  }
  if index < brickCount() {
    brickState[index] = state | (ranks.before.x << 8u) | (ranks.before.y << 16u);
  }
}

// Bricks that stop computing push their slots onto the stack in brick order, and are
// zeroed: their interior, and the aprons of the bricks before them.
@compute @workgroup_size(64)
fn freeBricks(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if index >= brickCount() {
    return;
  }
  let state = brickState[index];
  if (state & BRICK_FREES) == 0u {
    return;
  }
  let page = pagesOut[index];
  freeSlots[u32(atomicLoad(&poolState.free)) + brickRank(index, state).x] = pageSlot(page);
  if (state & 4u) != 0u {
    zeroBrick(index, u32(page));
  }
  pagesOut[index] = -1;
}

// Gives the bricks that compute slots from the top of the stack in brick order, then lists
// them and the fresh slots to zero. Requests past the free slots, down to the held-back
// ones, fail, and those bricks do not compute.
@compute @workgroup_size(64)
fn allocateBricks(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if index >= brickCount() {
    return;
  }
  let state = brickState[index];
  let listed = (state & 1u) != 0u;
  var page = pagesOut[index];
  var fresh = false;
  if (state & BRICK_REQUESTS) != 0u {
    let freed = i32(blocksBefore(rankBlocks()).x);
    let top = atomicLoad(&poolState.free) + freed - i32(brickRank(index, state).y);
    if top > heldSlots() {
      page = slotPage(freeSlots[u32(top - 1)]);
      pagesOut[index] = page;
      zeroBrick(index, u32(page) | FRESH_SLOT);
      fresh = true;
    }
  }
  if page >= 0 {
    atomicStore(&activeBricks[atomicAdd(&activeBricks[0], 1u) + 1u], index);
  }
  brickState[index] = u32(page >= 0);
  let entry = select(-1, page | select(0, FRESH_ENTRY, fresh), page >= 0);
  textureStore(entriesOut, vec2u(index & 63u, index >> 6u), vec4i(entry, 0, 0, 0));
  // This step's evolution records content and expansion afresh. A brick whose last
  // evolution produced expansion may still hold it.
  let flags = atomicLoad(&activity[index]);
  atomicStore(&activity[index], select(0u, MAY_EXPAND, (flags & EXPANDS) != 0u));
}

// Each slotted brick's record: 27 neighbor entries and the pressure-unknown flag.
@group(0) @binding(67) var recordsOut: texture_storage_2d<rgba32sint, write>;
@compute @workgroup_size(64)
fn linkBricks(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if index >= brickCount() || entryById(index) < 0 {
    return;
  }
  enterBrick(index, false);
  var words: array<i32, 28>;
  for (var k = 0u; k < 27u; k++) {
    let d = vec3i(i32(k % 3u), i32((k / 3u) % 3u), i32(k / 9u)) - 1;
    let neighbor = brickId(home.brick + d);
    words[k] = select(-1, entryById(u32(max(neighbor, 0))), neighbor >= 0);
  }
  // Whether every velocity cell of the brick is a pressure unknown (pressure.wgsl
  // activeCell): it and its negative neighbors compute, those below the ground aside.
  var unknowns = 1;
  for (var k = 0u; k < 8u; k++) {
    let d = -vec3i(vec3u(k, k >> 1u, k >> 2u) & vec3u(1u));
    if u.counts.w > .5 && home.brick.y + d.y < 0 {
      continue;
    }
    if words[neighborIndex(d)] < 0 {
      unknowns = 0;
    }
  }
  words[BRICK_PRESSURE_FLAG] = unknowns;
  let row = index >> 6u;
  let column = (index & 63u) * BRICK_RECORD_TEXELS;
  for (var t = 0u; t < 7u; t++) {
    let k = t * 4u;
    textureStore(recordsOut,
      vec2u(column + t, row),
      vec4i(words[k], words[k + 1u], words[k + 2u], words[k + 3u]));
  }
}

// Draws the colliders into the solids (solids.wgsl) of the bricks given slots this step,
// from the zeroed list, or of every brick with a slot when `u.counts.x` is set, after the
// colliders changed. A workgroup per brick, a word of 32 cells per lane and pass.
@group(0) @binding(72) var<storage, read_write> solidsOut: array<u32>;
fn insideCollider(c: Collider, cell: vec3i) -> bool {
  let kind = u32(c.center.w);
  if kind == 2u {
    let q = cell - c.lo.xyz;
    if any(q < vec3i(0)) || any(q >= c.size.xyz) {
      return false;
    }
    let index = u32((q.z * c.size.y + q.y) * c.size.x + q.x);
    return ((colliderMasks[u32(c.lo.w) + (index >> 5u)] >> (index & 31u)) & 1u) != 0u;
  }
  let w = (vec3f(cell) + .5) * u.fieldGrid.xyz;
  let d = abs((transpose(c.axes) * (w - c.center.xyz)) / c.extent.xyz);
  if kind == 1u {
    return all(d < vec3f(1));
  }
  return dot(d, d) < 1.0;
}

@compute @workgroup_size(64)
fn buildSolids(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  var id: u32;
  var page: i32;
  if u.counts.x > .5 {
    id = group.x + group.y * BRICK_LIST_SPLIT;
    if id >= brickCount() {
      return;
    }
    page = entryPage(entryById(id));
  } else {
    let index = listedBrick(group);
    if index >= zeroList[0] {
      return;
    }
    id = zeroList[1u + 2u * index];
    let payload = zeroList[2u + 2u * index];
    if (payload & FRESH_SLOT) == 0u {
      return;
    }
    page = i32(payload & 0xffffffu);
  }
  if page < 0 {
    return;
  }
  enterBrick(id, false);
  let shift = u32(u.pool.w);
  let side = 1u << shift;
  let words = 1u << (3u * shift - 5u);
  let first = poolSlot(page, u.pool.xy) * words;
  let origin = home.brick * i32(side);
  let list = tileList(COLLIDERS);
  for (var word = lane; word < words; word += 64u) {
    var bits = 0u;
    for (var k = 0u; k < 32u; k++) {
      let bit = word * 32u + k;
      let cell = origin + vec3i(vec3u(bit & (side - 1u),
          (bit >> shift) & (side - 1u),
          bit >> (2u * shift)));
      for (var i = 0u; i < list.y; i++) {
        if insideCollider(colliders[tileLists[list.x + i]], cell) {
          bits |= 1u << k;
          break;
        }
      }
    }
    solidsOut[first + word] = bits;
  }
}

fn writeDispatch(offset: u32, tiles: u32, count: u32) {
  dispatchArgs[offset] = tiles;
  dispatchArgs[offset + 1u] = min(count, BRICK_LIST_SPLIT);
  dispatchArgs[offset + 2u] = (count + BRICK_LIST_SPLIT - 1u) / BRICK_LIST_SPLIT;
}

// Commits the step's frees and grants to the stack, then writes indirect dispatch sizes:
// active bricks for 8x4x4 and 8x8x4 velocity kernels and 8x4x4 field kernels, then
// zeroed bricks for velocity and field kernels.
@compute @workgroup_size(1)
fn finishBricks() {
  let totals = blocksBefore(rankBlocks());
  let free = atomicLoad(&poolState.free) + i32(totals.x);
  let granted = min(i32(totals.y), max(free - heldSlots(), 0));
  atomicStore(&poolState.free, free - granted);
  atomicAdd(&poolState.failures, totals.y - u32(granted));
  for (var i = 0u; i < rankBlocks(); i++) {
    blockRanks[i] = vec2u(0);
  }
  let listed = atomicLoad(&activeBricks[0]);
  let zeroed = atomicLoad(&zeroBricks[0]);
  let side = 1u << u32(u.velocityGrid.w);
  let velocityTiles = max(1u, side / 8u) * max(1u, side / 4u) * max(1u, side / 4u);
  let tallTiles = max(1u, side / 8u) * max(1u, side / 8u) * max(1u, side / 4u);
  let fieldTiles = (1u << (3u * u32(u.pool.w))) / 128u;
  writeDispatch(0u, velocityTiles, listed);
  writeDispatch(3u, tallTiles, listed);
  writeDispatch(6u, fieldTiles, listed);
  // Independent 64-lane chunks include the coarse smoke cache's apron.
  let smokeEdge = (1u << u32(u.noise.w)) + 1u;
  writeDispatch(36u, (smokeEdge * smokeEdge * smokeEdge + 63u) / 64u, listed);
  writeDispatch(9u, velocityTiles, zeroed);
  writeDispatch(12u, fieldTiles, zeroed);
  // For the pressure levels in each brick's slot (pressure.wgsl): a workgroup per listed
  // brick, per zeroed brick, and per eight listed bricks.
  writeDispatch(15u, 1u, listed);
  writeDispatch(18u, 1u, zeroed);
  writeDispatch(21u, 1u, (listed + 7u) / 8u);
  // 64-lane chunks for each possible coarse slot level. The 2³ level uses octets.
  for (var level = 1u; level < 5u; level++) {
    let shift = u32(u.velocityGrid.w) - min(level, u32(u.velocityGrid.w));
    let cells = 1u << (3u * shift);
    writeDispatch(24u + (level - 1u) * 3u, max(1u, cells / 64u), listed);
  }
}
