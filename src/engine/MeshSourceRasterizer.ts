import type { SimulationSource, Vec3 } from './types.ts';

interface AccumulatedCell {
  x: number;
  y: number;
  z: number;
  flame: number;
  heat: number;
  smoke: number;
  fuel: number;
  motionX: number;
  motionY: number;
  motionZ: number;
  response: number;
}

export interface MeshSourceRecords {
  records: ArrayBuffer;
  count: number;
}

const MAX_SAMPLE_VISITS = 2_000_000;

type Samples = NonNullable<SimulationSource['samples']>;
type Coverage = CellMap<{ x: number; y: number; z: number; value: number }>;

/** Values by lattice cell, keyed by numbers: by X, then by Y and Z together, which stay
 * exact for coordinates within 2^25 cells of the origin. */
class CellMap<T> {
  private rows = new Map<number, Map<number, T>>();
  private static key(y: number, z: number): number {
    return (y + 0x2000000) * 0x4000000 + z + 0x2000000;
  }
  get(x: number, y: number, z: number): T | undefined {
    return this.rows.get(x)?.get(CellMap.key(y, z));
  }
  set(x: number, y: number, z: number, value: T): void {
    let row = this.rows.get(x);
    if (!row) this.rows.set(x, (row = new Map()));
    row.set(CellMap.key(y, z), value);
  }
  get size(): number {
    let size = 0;
    for (const row of this.rows.values()) size += row.size;
    return size;
  }
  *values(): IterableIterator<T> {
    for (const row of this.rows.values()) yield* row.values();
  }
}

/**
 * Cache transformed surface coverage independently of live emission/velocity rates.
 * Records hold signed cells of the world lattice.
 */
export class MeshSourceRasterizer {
  private coverage = new WeakMap<Samples, { area: number; cells: Coverage }>();
  private previous: SimulationSource[] = [];
  private result: MeshSourceRecords = { records: new ArrayBuffer(48), count: 0 };
  private spacing: Vec3;
  constructor(spacing: Vec3) {
    this.spacing = [...spacing];
  }
  prepare(sources: readonly SimulationSource[]): MeshSourceRecords {
    const spacing = this.spacing;
    const meshSources = sources.filter((source) => source.shape === 'mesh');
    if (
      meshSources.length === this.previous.length &&
      meshSources.every((source, i) => {
        const old = this.previous[i];
        return (
          source.samples === old.samples &&
          source.surfaceArea === old.surfaceArea &&
          source.flame === old.flame &&
          source.heatRate === old.heatRate &&
          source.smokeRate === old.smokeRate &&
          source.fuelRate === old.fuelRate &&
          source.velocityResponse === old.velocityResponse &&
          source.velocity.every((value, axis) => value === old.velocity[axis])
        );
      })
    )
      return this.result;
    const cells = new CellMap<AccumulatedCell>();
    const meanFaceArea =
      (spacing[0] * spacing[1] + spacing[1] * spacing[2] + spacing[2] * spacing[0]) / 3;
    let visits = 0;

    for (const source of meshSources) {
      if (!source.samples || !source.surfaceArea || source.surfaceArea <= 0)
        throw new Error('Mesh sources require surface samples and a positive surface area.');
      visits += source.samples.length * 8;
      if (visits > MAX_SAMPLE_VISITS)
        throw new Error('Mesh source sampling exceeds two million cell visits.');
      const cached = this.coverage.get(source.samples);
      let coverage = cached?.cells;
      if (!coverage || cached!.area !== source.surfaceArea) {
        coverage = new CellMap();
        for (const sample of source.samples) {
          const grid = sample.position.map((value, axis) => value / spacing[axis] - 0.5);
          const base = grid.map(Math.floor);
          const fraction = grid.map((value, axis) => value - base[axis]);
          for (let z = 0; z < 2; z++)
            for (let y = 0; y < 2; y++)
              for (let x = 0; x < 2; x++) {
                const cx = base[0] + x;
                const cy = base[1] + y;
                const cz = base[2] + z;
                const trilinear =
                  (x ? fraction[0] : 1 - fraction[0]) *
                  (y ? fraction[1] : 1 - fraction[1]) *
                  (z ? fraction[2] : 1 - fraction[2]);
                const value = (sample.weight * source.surfaceArea * trilinear) / meanFaceArea;
                const previous = coverage.get(cx, cy, cz);
                if (previous) previous.value += value;
                else coverage.set(cx, cy, cz, { x: cx, y: cy, z: cz, value });
              }
        }
        this.coverage.set(source.samples, { area: source.surfaceArea, cells: coverage });
      }
      for (const covered of coverage.values()) {
        const mask = Math.min(1, covered.value);
        if (mask <= 0) continue;
        let cell = cells.get(covered.x, covered.y, covered.z);
        if (!cell) {
          cell = {
            x: covered.x,
            y: covered.y,
            z: covered.z,
            flame: 0,
            heat: 0,
            smoke: 0,
            fuel: 0,
            motionX: 0,
            motionY: 0,
            motionZ: 0,
            response: 0,
          };
          cells.set(covered.x, covered.y, covered.z, cell);
        }
        cell.flame = Math.max(cell.flame, source.flame * mask);
        cell.heat += source.heatRate * mask;
        cell.smoke += source.smokeRate * mask;
        cell.fuel += (source.fuelRate ?? 0) * mask;
        const response = source.velocityResponse * mask;
        cell.motionX += source.velocity[0] * response;
        cell.motionY += source.velocity[1] * response;
        cell.motionZ += source.velocity[2] * response;
        cell.response += response;
      }
    }

    const records = new ArrayBuffer(Math.max(1, cells.size + 1) * 48);
    const integers = new Int32Array(records);
    const floats = new Float32Array(records);
    let record = 1;
    for (const cell of cells.values()) {
      const offset = record * 12;
      integers.set([cell.x, cell.y, cell.z, 0], offset);
      floats.set([cell.flame, cell.heat, cell.smoke, cell.fuel], offset + 4);
      floats.set([cell.motionX, cell.motionY, cell.motionZ, cell.response], offset + 8);
      record++;
    }
    this.previous = meshSources.map((source) => ({ ...source, velocity: [...source.velocity] }));
    this.result = { records, count: cells.size };
    return this.result;
  }
}
