import { MeshSourceRasterizer } from './MeshSourceRasterizer.ts';
import type { SimulationForce, SimulationSource, Vec3 } from './types.ts';

/** Rows of vec4s per force, then per target region (force-fields.wgsl). */
const FORCE_ROWS = 5;
const REGION_ROWS = 2;
const NO_COVERAGE = 0xffffffff;

/** A world-space box; forces without targets reach everywhere. */
export interface ForceReach {
  lo: Vec3;
  hi: Vec3;
}
interface MeshTarget {
  index: number;
  source: SimulationSource;
}

/** Packs forces for the GPU: their rows, the regions of analytic targets, and for mesh
 * targets the velocity cells they cover, by brick. */
export class ForceFields {
  private rasterizer: MeshSourceRasterizer;
  private previous: MeshTarget[] = [];
  private version = -1;
  private bricks = 0;
  private coverage = new Uint32Array(1).fill(NO_COVERAGE);
  private spacing: Vec3;
  private readonly brickCells: number;
  /** `spacing` is the velocity cell size: mesh targets cover velocity cells of the
   * world lattice. */
  constructor(spacing: Vec3, brickCells = 8) {
    this.brickCells = brickCells;
    this.spacing = [...spacing];
    this.rasterizer = new MeshSourceRasterizer(spacing);
  }
  /**
   * The GPU rows of `forces`, the coverage of their mesh targets and how far each force
   * reaches. Coverage is indexed by brick id (`bricks` of them): `brickId` gives the id of
   * a brick, or -1 outside every tile, and `version` changes when ids do.
   */
  prepare(
    forces: readonly SimulationForce[],
    bricks: number,
    version: number,
    brickId: (brick: Vec3) => number,
  ) {
    const regions = forces.reduce(
      (count, force) =>
        count + (force.sources?.filter((source) => source.shape !== 'mesh').length ?? 0),
      0,
    );
    const data = new Float32Array(
      Math.max(1, forces.length * FORCE_ROWS + regions * REGION_ROWS) * 4,
    );
    let region = forces.length * FORCE_ROWS;
    const reach: (ForceReach | undefined)[] = [];
    forces.forEach((force, index) => {
      const analytic = force.sources?.filter((source) => source.shape !== 'mesh') ?? [];
      data.set(
        [
          ...force.vector,
          force.strength,
          ...force.center,
          ['wind', 'turbulence', 'vortex', 'radial'].indexOf(force.type),
          force.scale,
          force.lift,
          force.inward,
          0,
          region,
          analytic.length,
          force.sources?.some((source) => source.shape === 'mesh') ? 1 : 0,
          force.sources === undefined ? 0 : 1,
        ],
        index * FORCE_ROWS * 4,
      );
      let box: ForceReach | undefined;
      const include = (lo: Vec3, hi: Vec3) => {
        box = box
          ? {
              lo: box.lo.map((n, axis) => Math.min(n, lo[axis])) as Vec3,
              hi: box.hi.map((n, axis) => Math.max(n, hi[axis])) as Vec3,
            }
          : { lo, hi };
      };
      for (const source of analytic) {
        data.set(
          [...source.position, source.shape === 'box' ? 3 : 1, ...source.size, 0],
          region * 4,
        );
        region += REGION_ROWS;
        const extent = source.size.map((n) => Math.max(n, 0.03));
        include(
          source.position.map((n, axis) => n - extent[axis]) as Vec3,
          source.position.map((n, axis) => n + extent[axis]) as Vec3,
        );
      }
      for (const source of force.sources ?? []) {
        if (source.shape !== 'mesh' || !source.samples?.length) continue;
        const lo: Vec3 = [Infinity, Infinity, Infinity];
        const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (const { position } of source.samples)
          for (let axis = 0; axis < 3; axis++) {
            lo[axis] = Math.min(lo[axis], position[axis] - this.spacing[axis]);
            hi[axis] = Math.max(hi[axis], position[axis] + this.spacing[axis]);
          }
        include(lo, hi);
      }
      // A targeted force without targets reaches nothing; one without targets, everywhere.
      reach.push(
        force.sources === undefined ? undefined : (box ?? { lo: [0, 0, 0], hi: [-1, -1, -1] }),
      );
    });
    return { data, reach, coverage: this.prepareCoverage(forces, bricks, version, brickId) };
  }
  /** Each brick's coverage entries, rebuilt when the mesh targets or the brick ids change. */
  private prepareCoverage(
    forces: readonly SimulationForce[],
    bricks: number,
    version: number,
    brickId: (brick: Vec3) => number,
  ): Uint32Array<ArrayBuffer> {
    const meshes = forces.flatMap((force, index) =>
      (force.sources ?? [])
        .filter((source) => source.shape === 'mesh')
        .map((source) => ({ index, source })),
    );
    if (
      version === this.version &&
      bricks === this.bricks &&
      meshes.length === this.previous.length &&
      meshes.every((entry, i) => {
        const old = this.previous[i];
        return (
          entry.index === old.index &&
          entry.source.samples === old.source.samples &&
          entry.source.surfaceArea === old.source.surfaceArea
        );
      })
    )
      return this.coverage;
    this.previous = meshes;
    this.version = version;
    this.bricks = bricks;
    // Each brick's entries: a force's index, then its velocity cells' bits.
    const entryWords = 1 + Math.ceil(this.brickCells ** 3 / 32);
    const mask = this.brickCells - 1;
    const shift = Math.log2(this.brickCells);
    const entries = new Map<number, Map<number, Uint32Array>>();
    for (const { index, source } of meshes) {
      const { records, count } = this.rasterizer.prepare([{ ...source, flame: 1 }]);
      const cells = new Int32Array(records);
      for (let record = 1; record <= count; record++) {
        const cell = [0, 1, 2].map((axis) => cells[record * 12 + axis]);
        const id = brickId(cell.map((n) => n >> shift) as Vec3);
        if (id < 0) continue;
        let forcesOfBrick = entries.get(id);
        if (!forcesOfBrick) entries.set(id, (forcesOfBrick = new Map()));
        let bits = forcesOfBrick.get(index);
        if (!bits) forcesOfBrick.set(index, (bits = new Uint32Array(entryWords - 1)));
        const bit = ((cell[2] & mask) * this.brickCells + (cell[1] & mask)) * this.brickCells + (cell[0] & mask);
        bits[bit >> 5] |= (1 << (bit & 31)) >>> 0;
      }
    }
    let size = bricks;
    for (const forcesOfBrick of entries.values()) size += 1 + forcesOfBrick.size * entryWords;
    const coverage = new Uint32Array(Math.max(1, size)).fill(NO_COVERAGE, 0, bricks);
    let next = bricks;
    for (const [id, forcesOfBrick] of entries) {
      coverage[id] = next;
      coverage[next++] = forcesOfBrick.size;
      for (const [index, bits] of forcesOfBrick) {
        coverage[next++] = index;
        coverage.set(bits, next);
        next += bits.length;
      }
    }
    this.coverage = coverage;
    return coverage;
  }
}
