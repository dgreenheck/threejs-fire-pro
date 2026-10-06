import source from './shaders/pressure.wgsl?raw';
import bricks from './shaders/bricks.wgsl?raw';
import pool from './shaders/pool.wgsl?raw';
import tiles from './shaders/tiles.wgsl?raw';
import solids from './shaders/solids.wgsl?raw';
import type { Vec3 } from './types.ts';
import { beginComputePass } from './GpuProfiler.ts';
import { PRESSURE_KERNELS, pressureBindingLayout } from './pressureKernels.ts';

/** The fluid solver's bricks, which the solve runs over (fluid-allocation.wgsl allocateBricks). */
export interface PressureBricks {
  /** The computing bricks: a count, then one brick id each. */
  list: GPUBuffer;
  /** The bricks zeroed this step: a count, then a brick id and its page each. */
  zeroed: GPUBuffer;
  /** Indirect dispatch sizes: 8x4x4 tiles of the listed bricks, one workgroup per listed
   * brick, one per eight listed bricks, and one per zeroed brick. */
  dispatch: GPUBuffer;
  offsets: { tiles: number; octets: number; zeroed: number; coarse: number[] };
  /** Each slot's solid cells (solids.wgsl). */
  solids: GPUBuffer;
  /** The tile directory's textures (tiles.wgsl): tile records and their hash, and each
   * brick's entry and record. */
  directory: { tiles: GPUTexture; table: GPUTexture; entries: GPUTexture; bricks: GPUTexture };
  /** Tile slots: brick ids are 64 per tile slot. */
  tiles: number;
  /** Slots of the brick pools. */
  slots: number;
}
interface Dispatch {
  pipeline: GPUComputePipeline;
  group: GPUBindGroup;
  workgroups: Vec3 | { buffer: GPUBuffer; offset: number };
}
interface Plan {
  commands: Dispatch[];
  current: number;
}
interface Level {
  p: [GPUBuffer, GPUBuffer];
  rhs: GPUBuffer;
  topology: GPUBuffer;
  current: number;
}
type Levels = [number, number, number, number];
/** Tiles the bottom level's single workgroup holds (pressure.wgsl relaxCoarse). */
const SHARED_TILES = 256;
const LIST_SPLIT = 65535;

/**
 * Geometric multigrid on the bricks: weighted-Jacobi smoothing, residual restriction,
 * trilinear correction. Brick levels coarsen the velocity grid down to 2³ cells;
 * three tile levels then coarsen the 4x4x4 bricks, only where storage exists.
 */
export class PressureSolver {
  private pipelines = new Map<string, GPUComputePipeline>();
  private groups = new Map<string, GPUBindGroup>();
  private uniforms = new Map<string, GPUBuffer>();
  private plans = new Map<string, Plan>();
  /** Coarse levels; level 0's pressure and right-hand side are the caller's. */
  private levels: Level[] = [];
  private fineTopology?: GPUBuffer;
  /** Storage that follows the pool's slots, and storage that follows the tile slots. */
  private brickStorage: GPUBuffer[] = [];
  private tileStorage: GPUBuffer[] = [];
  private ids = new WeakMap<object, number>();
  private nextId = 0;
  private geometryDirty = true;
  private bricks?: PressureBricks;
  private tiles = 0;
  private readonly brickLevels: number;
  private readonly bottom: number;
  private readonly cells: number[];
  private constructor(
    private device: GPUDevice,
    private uniform: GPUBuffer,
    velocityCells: number,
  ) {
    this.brickLevels = Math.log2(velocityCells);
    this.bottom = this.brickLevels + 2;
    this.cells = [
      ...Array.from({ length: this.brickLevels }, (_, i) => (velocityCells / 2 ** i) ** 3),
      64, 8, 1,
    ];
  }
  static async create(device: GPUDevice, uniform: GPUBuffer, velocityCells = 8): Promise<PressureSolver> {
    const solver = new PressureSolver(device, uniform, velocityCells);
    const module = device.createShaderModule({
      label: 'Multigrid pressure solver',
      // The f16 directive comes first.
      code: [source, bricks, pool, tiles, solids].join('\n'),
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length)
      throw new Error(errors.map((m) => `Pressure WGSL ${m.lineNum}: ${m.message}`).join('\n'));
    await Promise.all(
      Object.entries(PRESSURE_KERNELS).map(async ([entryPoint, bindings]) => {
        const layout = device.createBindGroupLayout({
          entries: bindings.map(pressureBindingLayout),
        });
        solver.pipelines.set(
          entryPoint,
          await device.createComputePipelineAsync({
            label: `Multigrid ${entryPoint}`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module, entryPoint },
          }),
        );
      }),
    );
    return solver;
  }
  /** Call after the colliders changed, before their solid cells are redrawn. */
  invalidateGeometry(): void {
    this.geometryDirty = true;
  }
  /** The bricks after the tiles changed or the pool grew. Storage follows the slots and
   * the tile slots, and every brick's topology is rebuilt on the next solve. */
  setBricks(bricks: PressureBricks): void {
    const slots = this.bricks?.slots;
    this.bricks = bricks;
    if (bricks.slots !== slots) this.allocateBricks(bricks.slots);
    if (bricks.tiles !== this.tiles) this.allocateTiles(bricks.tiles);
    this.resetPlans();
  }
  /** Forget bind groups, dimensions and command plans; topology is rebuilt on the next solve. */
  private resetPlans(): void {
    for (const buffer of this.uniforms.values()) buffer.destroy();
    this.uniforms.clear();
    this.groups.clear();
    this.plans.clear();
    this.geometryDirty = true;
  }
  /** A storage buffer of `bytes`, kept in `owner`. */
  private storage(owner: GPUBuffer[], label: string, bytes: number): GPUBuffer {
    const buffer = this.device.createBuffer({
      label,
      size: Math.max(4, Math.ceil(bytes / 4) * 4),
      usage: GPUBufferUsage.STORAGE,
    });
    owner.push(buffer);
    return buffer;
  }
  /** Two pressures and a right-hand side in half precision, and one flag byte per cell. */
  private level(owner: GPUBuffer[], index: number, owners: number): Level {
    const cells = owners * this.cells[index];
    const scalars = (role: string) =>
      this.storage(owner, `Pressure level ${index} ${role}`, cells * 2);
    return {
      p: [scalars('A'), scalars('B')],
      rhs: scalars('right-hand side'),
      topology: this.storage(owner, `Pressure level ${index} topology`, cells),
      current: 0,
    };
  }
  private allocateBricks(slots: number): void {
    for (const buffer of this.brickStorage) buffer.destroy();
    this.brickStorage = [];
    this.fineTopology = this.storage(
      this.brickStorage,
      'Pressure level 0 topology',
      slots * this.cells[0],
    );
    for (let index = 1; index < this.brickLevels; index++)
      this.levels[index] = this.level(this.brickStorage, index, slots);
  }
  private allocateTiles(tiles: number): void {
    for (const buffer of this.tileStorage) buffer.destroy();
    this.tileStorage = [];
    for (let index = this.brickLevels; index <= this.bottom; index++)
      this.levels[index] = this.level(this.tileStorage, index, tiles);
    this.tiles = tiles;
  }
  /** GPU memory this solver owns: coarse values, all topology and dispatch uniforms. */
  get memoryBytes(): number {
    return [...this.brickStorage, ...this.tileStorage, ...this.uniforms.values()].reduce(
      (sum, buffer) => sum + buffer.size,
      0,
    );
  }
  private id(resource: object): number {
    let id = this.ids.get(resource);
    if (id === undefined) this.ids.set(resource, (id = this.nextId++));
    return id;
  }
  /** Per dispatch: input, output and coarse level, bottom iterations, then the brick ids
   * and how the brick list is read, and the tile slots (pressure.wgsl Dimensions). */
  private dimensions(levels: Levels, stride: number): GPUBuffer {
    const key = [...levels, stride].join('/');
    let buffer = this.uniforms.get(key);
    if (!buffer) {
      buffer = this.device.createBuffer({
        label: `Pressure dimensions ${key}`,
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        buffer,
        0,
        new Uint32Array([...levels, this.tiles * 64, 0, 0, stride, this.tiles, 0, 0, 0]),
      );
      this.uniforms.set(key, buffer);
    }
    return buffer;
  }
  /** `stride` reads the brick list: 1 for the listed bricks, 2 for the zeroed ones, 0 for
   * every brick id. */
  private prepare(
    name: string,
    levels: Levels,
    resources: Record<number, GPUBuffer>,
    workgroups: Dispatch['workgroups'],
    stride = 1,
  ): Dispatch {
    const bricks = this.bricks!;
    const bound: Record<number, GPUBuffer | GPUTexture> = {
      0: this.uniform,
      6: this.dimensions(levels, stride),
      12: stride === 2 ? bricks.zeroed : bricks.list,
      71: bricks.solids,
      63: bricks.directory.tiles,
      64: bricks.directory.table,
      65: bricks.directory.entries,
      66: bricks.directory.bricks,
      ...resources,
    };
    const bindings = PRESSURE_KERNELS[name];
    const key = name + ':' + bindings.map((b) => this.id(bound[b])).join('|');
    const pipeline = this.pipelines.get(name)!;
    let group = this.groups.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        label: `Multigrid ${name}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({
          binding,
          resource: pressureBindingLayout(binding).texture
            ? (bound[binding] as GPUTexture).createView()
            : { buffer: bound[binding] as GPUBuffer },
        })),
      });
      this.groups.set(key, group);
    }
    return { pipeline, group, workgroups };
  }
  private topology(index: number): GPUBuffer {
    return index === 0 ? this.fineTopology! : this.levels[index].topology;
  }
  /** Topology of every slotted brick and every tile, after the geometry or the bricks
   * changed. */
  private buildTopology(execute: (dispatch: Dispatch) => void): void {
    const count = this.tiles * 64;
    const everyBrick: Vec3 = [Math.min(count, LIST_SPLIT), Math.ceil(count / LIST_SPLIT), 1];
    for (let index = 0; index < this.brickLevels; index++)
      execute(
        this.prepare(
          'buildBrickTopology',
          [index, index, index, 0],
          { 10: this.topology(index) },
          everyBrick,
          0,
        ),
      );
    for (let index = this.brickLevels; index <= this.bottom; index++)
      execute(
        this.prepare('buildTileTopology', [index, index, index, 0], { 10: this.topology(index) }, [
          Math.ceil((this.tiles * this.cells[index]) / 4 / 256),
          1,
          1,
        ]),
      );
  }
  /** The commands of `cycles` V-cycles. The first finest iterate is initialized from
   * zero directly; coarse correction levels are initialized by restriction. */
  private plan(cycles: number): Plan {
    const bricks = this.bricks!;
    const commands: Dispatch[] = [];
    const indirect = (offset: number) => ({ buffer: bricks.dispatch, offset });
    // Coarse brick chunks use 64 lanes, packing eight 2³-cell bricks per group.
    const workgroups = (name: string, level: number): Dispatch['workgroups'] =>
      name === 'initializePressure' || name === 'relaxBricks' || name === 'prolongateBricks'
        ? indirect(bricks.offsets.tiles)
        : name.endsWith('Bricks')
          ? indirect(this.cells[level] === 8 ? bricks.offsets.octets : bricks.offsets.coarse[level - 1])
          : name === 'relaxCoarse'
            ? [1, 1, 1]
            : [this.tiles, 1, 1];
    const dispatch = (name: string, levels: Levels, resources: Record<number, GPUBuffer>) =>
      commands.push(this.prepare(name, levels, resources, workgroups(name, levels[1])));
    // A fresh slot needs its brick's topology: build it for the bricks zeroed this step
    // that still own their slots.
    for (let index = 0; index < this.brickLevels; index++)
      commands.push(
        this.prepare(
          'buildBrickTopology',
          [index, index, index, 0],
          { 10: this.topology(index) },
          indirect(bricks.offsets.zeroed),
          2,
        ),
      );
    const relax = (index: number) =>
      index === 0 ? 'relaxBricks' : index < this.brickLevels ? 'relaxCoarseBricks' : 'relaxTiles';
    let initialize = true;
    const smooth = (index: number, n: number) => {
      const level = this.levels[index];
      for (let i = 0; i < n; i++) {
        const first = index === 0 && initialize;
        dispatch(first ? 'initializePressure' : relax(index), [index, index, index, 0], {
          1: level.p[level.current],
          2: level.rhs,
          4: level.p[1 - level.current],
          8: level.topology,
        });
        level.current = 1 - level.current;
        if (first) initialize = false;
      }
    };
    const cycle = (index: number): void => {
      const level = this.levels[index];
      if (index === this.bottom) {
        // Restriction ran the first iteration. With a tile slot per lane, one workgroup
        // runs the rest without global barriers.
        if (this.tiles <= SHARED_TILES) {
          dispatch('relaxCoarse', [index, index, index, 19], {
            1: level.p[level.current],
            2: level.rhs,
            4: level.p[1 - level.current],
            8: level.topology,
          });
          level.current = 1 - level.current;
        } else smooth(index, 19);
        return;
      }
      smooth(index, index > 0 ? 2 : 1);
      const coarse = this.levels[index + 1];
      // Restriction also writes the coarse level's first Jacobi iterate from zero.
      dispatch(index + 1 < this.brickLevels ? 'restrictBricks' : 'restrictTiles', [index, index + 1, index + 1, 0], {
        1: level.p[level.current],
        2: level.rhs,
        4: coarse.rhs,
        8: level.topology,
        9: coarse.topology,
        11: coarse.p[1],
      });
      coarse.current = 1;
      cycle(index + 1);
      dispatch(
        index === 0 ? 'prolongateBricks' : index < this.brickLevels ? 'prolongateCoarseBricks' : 'prolongateTiles',
        [index, index, index + 1, 0],
        {
          1: level.p[level.current],
          3: coarse.p[coarse.current],
          4: level.p[1 - level.current],
          8: level.topology,
          9: coarse.topology,
        },
      );
      level.current = 1 - level.current;
      smooth(index, 3);
    };
    for (let i = 0; i < cycles; i++) cycle(0);
    return { commands, current: this.levels[0].current };
  }
  /**
   * Solve for the finest level's pressure from `rhs`, both in slot order: `p` is a
   * ping-pong pair. The first iterate is computed from a zero guess without reading
   * previous pressure. Returns the solution.
   */
  solve(
    encoder: GPUCommandEncoder,
    p: [GPUBuffer, GPUBuffer],
    rhs: GPUBuffer,
    cycles: number,
  ): GPUBuffer {
    if (!this.bricks) throw new Error('Set the bricks before solving.');
    this.levels[0] = { p, rhs, topology: this.fineTopology!, current: 0 };
    const pass = beginComputePass(this.device, encoder, { label: 'Multigrid V-cycles' });
    const execute = ({ pipeline, group, workgroups }: Dispatch) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      if (Array.isArray(workgroups)) pass.dispatchWorkgroups(...workgroups);
      else pass.dispatchWorkgroupsIndirect(workgroups.buffer, workgroups.offset);
    };
    if (this.geometryDirty) {
      this.buildTopology(execute);
      this.geometryDirty = false;
    }
    // Every coarser level is reset by restriction, so the buffers and the cycle count
    // select the schedule. The finest guess starts at zero, consistent with ambient
    // outlets. The previous solution is retained only for diagnostics between steps.
    const key = `${cycles}/${this.id(p[0])}/${this.id(p[1])}/${this.id(rhs)}`;
    let plan = this.plans.get(key);
    if (!plan) {
      plan = this.plan(cycles);
      this.plans.set(key, plan);
    }
    for (const command of plan.commands) execute(command);
    pass.end();
    return p[plan.current];
  }
  dispose(): void {
    for (const buffer of [...this.brickStorage, ...this.tileStorage]) buffer.destroy();
    this.brickStorage = [];
    this.tileStorage = [];
    this.levels = [];
    this.fineTopology = undefined;
    this.tiles = 0;
    this.bricks = undefined;
    this.resetPlans();
    this.pipelines.clear();
  }
}
