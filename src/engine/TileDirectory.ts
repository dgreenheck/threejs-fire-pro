import type { Vec3 } from './types.ts';

/** Bricks per side of a tile. */
export const TILE_BRICKS = 4;
/** Bricks per tile: a tile's page table has one entry each (tiles.wgsl). */
export const TILE_BRICK_COUNT = TILE_BRICKS ** 3;
/** Steps a tile stays after nothing needs it any more. */
const LINGER_STEPS = 60;
/** Tile states as the GPU reads them (tiles.wgsl). */
const FREE = 0;
const LIVE = 1;
const CLOSING = 2;

/** Words of a tile slot's record (tiles.wgsl): coordinates and state, then 27 neighbors. */
export const TILE_RECORD_WORDS = 32;
/** Entries of the hash per row of its texture (tiles.wgsl findTile). */
export const HASH_ROW = 256;

interface Tile {
  coord: Vec3;
  slot: number;
  /** The last step something needed the tile. */
  needed: number;
  /** Its bricks give up their slots in the step being prepared; removed in the next. */
  closing: boolean;
}

/** The tiles from `lo` to `hi`, inclusive. */
export interface TileBox {
  lo: Vec3;
  hi: Vec3;
}
/** The directory's GPU tables, rebuilt when the tiles change. */
export interface TileTables {
  /** Every tile slot's record: its tile's coordinates and state (FREE, LIVE or
   * CLOSING), then its 27 neighbors' slots, -1 where there is no tile. */
  records: Int32Array<ArrayBuffer>;
  /** The open-addressed hash of tile coordinates: x, y, z and the slot, or -1 when empty,
   * in whole rows of `HASH_ROW` entries. */
  hash: Int32Array<ArrayBuffer>;
  /** Boxes around the groups of touching tiles, merged until no two overlap: the renderer
   * marches each. */
  clusters: TileBox[];
  /** Every tile slot's tile coordinates and cluster, or -1 for a free slot, four words a
   * slot. */
  tiles: Int32Array<ArrayBuffer>;
}

/** The hash of tile coordinates `x, y, z` (tiles.wgsl tileHash). */
function tileHash(x: number, y: number, z: number): number {
  return (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
}

/**
 * The tiles of the sparse grid: cubes of 4x4x4 bricks at integer coordinates of the world
 * lattice, in which the GPU gives bricks pool slots. Tiles cover the sources and the
 * content with a tile of margin, and stay a while after nothing needs them, so the GPU
 * never waits on the CPU and the CPU never waits on readback. Each tile owns a slot that
 * indexes its page table and every per-brick array; slots are reused lowest first.
 */
export class TileDirectory {
  private readonly tiles = new Map<string, Tile>();
  private readonly slots: (Tile | undefined)[] = [];
  /** Changes since the tables were last built. */
  private dirty = true;
  /** Tile slots the GPU arrays hold. */
  capacity = 0;
  /** Slots given to tiles created since the last call to `takeCreated`. */
  private created: number[] = [];
  /** Coordinates of every tile slot when the tables were last built. */
  private snapshot: (Vec3 | undefined)[] = [];
  /** Below the ground, world y < 0, is solid: no tile is created there. */
  ground = true;
  /** Counts the tables built: brick ids keep their meaning until it changes. */
  version = 0;

  get count(): number {
    return this.tiles.size;
  }
  /** Whether the tiles changed since the tables were last built. */
  get changed(): boolean {
    return this.dirty;
  }
  /** Mark every tile overlapping the box of tile coordinates `lo` to `hi` as needed at
   * `step`, creating the missing ones. With the ground, tiles below it are never created. */
  need(lo: Vec3, hi: Vec3, step: number): void {
    for (let z = lo[2]; z <= hi[2]; z++)
      for (let y = this.ground ? Math.max(0, lo[1]) : lo[1]; y <= hi[1]; y++)
        for (let x = lo[0]; x <= hi[0]; x++) {
          const key = `${x},${y},${z}`;
          const tile = this.tiles.get(key);
          if (tile) {
            tile.needed = step;
            if (tile.closing) {
              tile.closing = false;
              this.dirty = true;
            }
            continue;
          }
          let slot = this.slots.indexOf(undefined);
          if (slot < 0) slot = this.slots.length;
          const created: Tile = { coord: [x, y, z], slot, needed: step, closing: false };
          this.slots[slot] = created;
          this.tiles.set(key, created);
          this.created.push(slot);
          this.dirty = true;
        }
  }
  /** Remove the tiles that closed in the previous step, then close the tiles nothing has
   * needed for a while: the step being prepared frees their bricks' slots. */
  retire(step: number): void {
    for (const [key, tile] of this.tiles) {
      if (tile.closing) {
        this.tiles.delete(key);
        this.slots[tile.slot] = undefined;
        this.dirty = true;
      } else if (step - tile.needed > LINGER_STEPS) {
        tile.closing = true;
        this.dirty = true;
      }
    }
    while (this.slots.length && !this.slots[this.slots.length - 1]) this.slots.pop();
  }
  /** Remove every tile. */
  clear(): void {
    this.tiles.clear();
    this.slots.length = 0;
    this.created = [];
    this.dirty = true;
  }
  /** The slot of the tile at `coord`, or -1. */
  slotAt(coord: readonly number[]): number {
    return this.tiles.get(`${coord[0]},${coord[1]},${coord[2]}`)?.slot ?? -1;
  }
  /** Slots that need the GPU arrays to hold them. */
  get slotsUsed(): number {
    return this.slots.length;
  }
  /** Slots of the tiles created since the last call, whose page tables start empty. */
  takeCreated(): number[] {
    const created = this.created.filter((slot) => this.slots[slot]);
    this.created = [];
    return created;
  }
  /** The tile coordinates of every slot when the tables were last built, to read GPU
   * results recorded against them. */
  get coordinates(): readonly (Vec3 | undefined)[] {
    return this.snapshot;
  }
  /** Build the GPU tables for `capacity` slots, at least `slotsUsed`. */
  tables(capacity: number): TileTables {
    if (capacity < this.slots.length) throw new Error('Tile capacity is below the tiles in use.');
    this.capacity = capacity;
    const records = new Int32Array(capacity * TILE_RECORD_WORDS).fill(-1);
    // At most half full, so every probe sequence reaches an empty entry.
    let size = HASH_ROW;
    while (size < this.tiles.size * 2) size *= 2;
    const hash = new Int32Array(size * 4).fill(-1);
    for (let slot = 0; slot < capacity; slot++)
      records.set([0, 0, 0, FREE], slot * TILE_RECORD_WORDS);
    for (const tile of this.tiles.values()) {
      const [x, y, z] = tile.coord;
      const record = tile.slot * TILE_RECORD_WORDS;
      records.set([x, y, z, tile.closing ? CLOSING : LIVE], record);
      let entry = tileHash(x, y, z) & (size - 1);
      while (hash[entry * 4 + 3] >= 0) entry = (entry + 1) & (size - 1);
      hash.set([x, y, z, tile.slot], entry * 4);
      for (let dz = -1; dz <= 1; dz++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const neighbor = this.tiles.get(`${x + dx},${y + dy},${z + dz}`);
            if (neighbor)
              records[record + 4 + (dz + 1) * 9 + (dy + 1) * 3 + dx + 1] = neighbor.slot;
          }
    }
    const { clusters, clusterOf } = this.clusters();
    const tiles = new Int32Array(capacity * 4);
    for (let slot = 0; slot < capacity; slot++) {
      const tile = this.slots[slot];
      tiles.set(tile ? [...tile.coord, clusterOf.get(tile)!] : [0, 0, 0, -1], slot * 4);
    }
    this.snapshot = this.slots.map((tile) => tile && ([...tile.coord] as Vec3));
    this.dirty = false;
    this.version++;
    return { records, hash, clusters, tiles };
  }
  /** Group the tiles that touch, faces, edges or corners, and box each group; then merge
   * boxes that overlap until none do, so every point lies in one box at most. */
  private clusters(): { clusters: TileBox[]; clusterOf: Map<Tile, number> } {
    const group = new Map<Tile, number>();
    const boxes: TileBox[] = [];
    for (const first of this.tiles.values()) {
      if (group.has(first)) continue;
      const index = boxes.length;
      const box: TileBox = { lo: [...first.coord], hi: [...first.coord] };
      group.set(first, index);
      const stack = [first];
      while (stack.length) {
        const [x, y, z] = stack.pop()!.coord;
        for (const [axis, n] of [x, y, z].entries()) {
          box.lo[axis] = Math.min(box.lo[axis], n);
          box.hi[axis] = Math.max(box.hi[axis], n);
        }
        for (let dz = -1; dz <= 1; dz++)
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const neighbor = this.tiles.get(`${x + dx},${y + dy},${z + dz}`);
              if (neighbor && !group.has(neighbor)) {
                group.set(neighbor, index);
                stack.push(neighbor);
              }
            }
      }
      boxes.push(box);
    }
    // Each box merged into another names it; a merged box can overlap boxes it did not
    // before, so repeat until nothing changes.
    const into = boxes.map((_, i) => i);
    const root = (i: number): number => (into[i] === i ? i : (into[i] = root(into[i])));
    for (let merged = true; merged;) {
      merged = false;
      for (let i = 0; i < boxes.length; i++)
        for (let j = i + 1; j < boxes.length; j++) {
          const a = root(i);
          const b = root(j);
          if (a === b || !overlaps(boxes[a], boxes[b])) continue;
          boxes[a] = {
            lo: boxes[a].lo.map((n, axis) => Math.min(n, boxes[b].lo[axis])) as Vec3,
            hi: boxes[a].hi.map((n, axis) => Math.max(n, boxes[b].hi[axis])) as Vec3,
          };
          into[b] = a;
          merged = true;
        }
    }
    const clusters: TileBox[] = [];
    const renumbered = new Map<number, number>();
    for (let i = 0; i < boxes.length; i++)
      if (root(i) === i) {
        renumbered.set(i, clusters.length);
        clusters.push(boxes[i]);
      }
    const clusterOf = new Map<Tile, number>();
    for (const [tile, index] of group) clusterOf.set(tile, renumbered.get(root(index))!);
    return { clusters, clusterOf };
  }
}

/** Whether two boxes of tiles share a tile. */
function overlaps(a: TileBox, b: TileBox): boolean {
  return a.lo.every((n, axis) => n <= b.hi[axis] && b.lo[axis] <= a.hi[axis]);
}
