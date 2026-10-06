import { ForceFields } from './ForceFields.ts';
import { BufferUploads } from './BufferUploads.ts';
import { beginComputePass } from './GpuProfiler.ts';
import { DEFAULT_FUEL } from './fuel.ts';
import { FLAME_CHANNELS, MAX_FLAME_KNOTS, packFlameStep } from './flame.ts';
import { PressureSolver, type PressureBricks } from './PressureSolver.ts';
import { createCurlNoise } from './CurlNoise.ts';
import { fluidSource, fluidFirstOrderSource } from './ShaderSources.ts';
import { MeshSourceRasterizer } from './MeshSourceRasterizer.ts';
import {
  TileDirectory,
  type TileBox,
  HASH_ROW,
  TILE_BRICKS,
  TILE_BRICK_COUNT,
  TILE_RECORD_WORDS,
} from './TileDirectory.ts';
import {
  FLUID_KERNELS,
  SCALAR_TRANSPORT_KERNELS,
  LISTED,
  VELOCITY_KERNELS,
  SMOKE_KERNELS,
  ZEROING,
  fluidBindingLayout,
  isBufferBinding,
} from './fluidKernels.ts';
import {
  BRICK_SIZES,
  initialPoolSlots,
  maxPoolVoxels,
  poolRows,
  poolSide,
  poolSize,
  poolSlotsPerAxis,
  FIXED_DT,
  VELOCITY_DIVISORS,
  SMOKE_DIVISORS,
  type SmokeDivisor,
  type BrickSize,
  type MeshColliderMask,
  type SimulationFrame,
  type SimulationGrid,
  type SimulationSource,
  type Vec3,
  type VelocityDivisor,
} from './types.ts';

type ComputePass = ReturnType<typeof beginComputePass>;
type Resource = GPUTexture | GPUBuffer;
type Grid = 'velocity' | 'field' | 'smoke';

/** Cold-started multigrid V-cycles per simulation substep. Starting from zero avoids
 * feedback from the previous incomplete solve as active boundaries move. */
const PRESSURE_CYCLES = 1;

/** Indirect dispatch offsets written by `finishBricks`. */
const DISPATCH = {
  velocity: 0,
  velocityTall: 12,
  field: 24,
  zeroedVelocity: 36,
  zeroedField: 48,
  brick: 60,
  zeroedBrick: 72,
  brickOctet: 84,
  smokeRender: 144,
} as const;
/** Brick list entries per dispatch row (bricks.wgsl). */
const LIST_SPLIT = 65535;
/** Tile slots the tile resources start with. */
const MIN_TILES = 16;
/** Bytes read back after every step before the tiles' content flags: the active brick
 * count, then the free slot count and failed allocations. */
const READBACK_HEADER = 16;

/** The scene's storage buffers (fluid-common.wgsl), by the binding that reads them. */
const SCENE_BUFFERS = [
  'emitters',
  'colliders',
  'occupancy',
  'meshTables',
  'meshSources',
  'forces',
  'coverage',
  'tileLists',
] as const;
type SceneBuffer = (typeof SCENE_BUFFERS)[number];
const SCENE_BINDINGS: Record<SceneBuffer, number> = {
  emitters: 9,
  colliders: 10,
  occupancy: 25,
  meshTables: 26,
  meshSources: 27,
  forces: 30,
  coverage: 31,
  tileLists: 70,
};
/** Bytes of one element of each scene buffer (fluid-common.wgsl): a binding holds at least one,
 * even before the first step packs the scene. */
const SCENE_ELEMENT: Record<SceneBuffer, number> = {
  emitters: 80,
  colliders: 112,
  occupancy: 4,
  meshTables: 4,
  meshSources: 48,
  forces: 16,
  coverage: 4,
  tileLists: 4,
};
const WORLD_AXES: readonly [Vec3, Vec3, Vec3] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
/** Words of a collider (fluid-common.wgsl Collider), and of each tile's header in the lists. */
const COLLIDER_WORDS = 28;
const TILE_LIST_WORDS = 8;

/** The bricks from `lo` to `hi`, inclusive. */
export interface BrickBox {
  lo: Vec3;
  hi: Vec3;
}
/** The smallest box holding both, either of which may be absent. */
function union(a: BrickBox | undefined, b: BrickBox): BrickBox {
  if (!a) return b;
  return {
    lo: a.lo.map((n, axis) => Math.min(n, b.lo[axis])) as Vec3,
    hi: a.hi.map((n, axis) => Math.max(n, b.hi[axis])) as Vec3,
  };
}
/** Bricks of margin around what a box must hold: content moves less than a brick a step,
 * and its measurement arrives a step or two late. */
const BOX_MARGIN = 2;
/** Words of a brick's record (tiles.wgsl): 27 neighbor entries and a pressure flag. */
const BRICK_RECORD_WORDS = 28;
/** Resources sized by the tile slots: per tile, then per brick of every tile. The tile
 * directory's tables are integer textures with a row per tile slot (tiles.wgsl). */
interface TileResources {
  /** Tile slots. */
  capacity: number;
  /** Each tile slot's coordinates, state and neighbors (rgba32sint, 8 texels a row). */
  records: GPUTexture;
  /** Each brick's entry: page and whether it computes (r32sint, 64 a row). */
  entries: GPUTexture;
  /** Each brick's record: 27 neighbor entries and a pressure flag (rgba32sint,
   * 7 texels per brick). */
  brickRecords: GPUTexture;
  /** Each brick's page, or -1. */
  pages: GPUBuffer;
  brickState: GPUBuffer;
  /** One content flag per brick, then whether each tile held content this step. */
  activity: GPUBuffer;
  /** The computing bricks: a count, then one brick id each. */
  activeBricks: GPUBuffer;
  /** The bricks whose slots are zeroed: a count, then a brick id and page each. */
  zeroBricks: GPUBuffer;
  /** Slot frees and requests per block of bricks. */
  blockRanks: GPUBuffer;
}
/** What the renderer reads of the bricks (VolumeLighting). */
export interface RenderBricks {
  /** Changes whenever the boxes or the tiles do. */
  version: number;
  /** The boxes of bricks the renderer marches, one per cluster of tiles with something to
   * draw; none overlaps another. */
  boxes: readonly BrickBox[];
  /** Every tile slot's tile coordinates and the box holding its cluster, or -1, four words
   * a slot. */
  tiles: Int32Array<ArrayBuffer>;
  /** Each brick id's page (pool.wgsl), or -1. */
  pages: GPUBuffer;
  /** The bricks that computed in the latest step, which hold every slot: a count, then a
   * brick id each. */
  active: GPUBuffer;
  /** The bricks the latest step zeroed: a count, then a brick id and its page each, the
   * top bit marking a slot given that step (fluid-allocation.wgsl zeroBrick). */
  zeroed: GPUBuffer;
  /** Indirect dispatch arguments of a workgroup per active/zeroed brick, and of
   * fine-grid and smoke-cache workgroups per active brick (bricks.wgsl), at these offsets. */
  dispatch: GPUBuffer;
  activeOffset: number;
  zeroedOffset: number;
  fieldOffset: number;
  smokeOffset: number;
}

/**
 * Grid-based fire and smoke on a sparse brick pool in world space. The world is a lattice
 * of independently sized fine-cell bricks, grouped into tiles of 4x4x4 bricks (TileDirectory).
 * Tiles cover the sources and the content; inside them, only bricks near content and
 * sources hold storage: each owns a slot of the pool textures, found through its page, so
 * memory follows the content (pool.wgsl, tiles.wgsl).
 */
export class FluidSimulation {
  readonly device: GPUDevice;
  /** Flame, heat and fuel cell size in meters. Smoke may use larger cells. */
  readonly voxelSize: number;
  velocityDivisor: VelocityDivisor;
  readonly smokeDivisor: SmokeDivisor;
  readonly scalarMacCormack: boolean;
  readonly smokeBrickCells: number;
  private smokePools: GPUTexture[] = [];
  /** Canonical smoke density: separate R16F pools in the coarser modes. */
  get smoke(): GPUTexture {
    return this.smokePools[0] ?? this.field;
  }
  /** The scalar predictor is dead after evolution, so rendering can reuse it for
   * smoothed density. The next timestep overwrites it before transport reads it. */
  get renderScratch(): GPUTexture {
    // Borrowing exposes writable storage. The renderer submits its work before
    // the next step; that step restores any apron texels it may have changed.
    this.predictorApronDirty = true;
    return this.fieldForward;
  }
  /** Rendering borrows the dead coarse predictor; absent-neighbor aprons must be
   * restored before the next corrected transport reads it. */
  get renderSmokeScratch(): GPUTexture {
    if (!this.smokePools.length) return this.renderScratch;
    this.smokePredictorApronDirty = true;
    return this.smokePools[1];
  }
  /** Field cells per brick side. */
  readonly brickCells: BrickSize;
  /** log2 of `brickCells`: bricks are a power of two cells on a side. */
  readonly brickShift: number;
  /** Velocity cells per brick side, derived from the independent brick size. */
  readonly velocityBrickCells: number;
  readonly velocityBrickShift: number;
  /** Slots per row and per layer of every pool (types.ts poolSide). */
  readonly poolSide: number;
  /** Most field voxels the pools may hold. They grow toward it as content spreads; past
   * it, bricks the content reaches stay empty and `poolLimited` is set. */
  voxelBudget = Infinity;
  /** Slots the pools hold, and slots in use at the latest measurement. */
  poolSlots = 0;
  usedSlots = 0;
  /** Sources of the previous step: a new one wants its slots at once. */
  private sourceIds = new Set<string>();
  /** At the latest measurement, bricks the content reached went without slots. */
  poolLimited = false;
  /** What still counts as content. Content keeps its bricks active and its tiles alive. */
  cutoff = 0;
  /** Active bricks at the latest measurement. */
  activeBrickCount = 0;
  time = 0;
  steps = 0;
  /** Heat, normalized flame lifetime and fuel in YZW. X holds smoke density in Full
   * mode, or this step's smoke production in coarser modes. Scratch swaps each step. */
  field!: GPUTexture;
  velocity!: GPUTexture;
  /** Authored target velocity divergence from flame evolution, in inverse seconds. */
  expansionRate!: GPUTexture;
  private velocityBack!: GPUTexture;
  private fieldForward!: GPUTexture;
  private predictorApronDirty = false;
  private smokePredictorApronDirty = false;
  private fieldCorrected!: GPUTexture;
  /** Velocity bounds occupy the velocity cells inside each fine scratch slot. Bounds
   * die before scalar transport overwrites these textures. Derive the pair so field
   * swaps and pool growth always expose the current scratch, never live scalar state. */
  private get velocityScratch(): [GPUTexture, GPUTexture] {
    return [this.fieldForward, this.fieldCorrected];
  }
  /** Exact backward-trace addresses, reused after velocity correction for scalar
   * transport. R32Uint words, one per field cell or two per velocity cell; no apron
   * because each correction reads only the address stored at its own cell. */
  private donors!: GPUTexture;
  private curl!: GPUTexture;
  /** Rows of slots the pool textures hold (types.ts poolRows). */
  private poolRows = 0;
  /** The most slots the 3D texture limit allows. */
  private readonly maxSlots: number;
  /** A stack of free slots, then its count and failed allocations. */
  private freeSlots!: GPUBuffer;
  /** Pressure's ping-pong pair and the divergence, in half precision and slot order: a
   * velocity brick's cells per slot (pressure.wgsl). */
  private pressure!: [GPUBuffer, GPUBuffer];
  private divergence!: GPUBuffer;
  private poolState!: GPUBuffer;
  /** Slots to grow the pools to before the next step. */
  private poolTarget = 0;
  /** Measurements of steps before this one predate the last growth. */
  private poolGrownAt = 0;
  private directory = new TileDirectory();
  private tiles!: TileResources;
  /** The directory's hash of tile coordinates (rgba32sint, `HASH_ROW` entries a row). */
  private tileTable!: GPUTexture;
  /** The clusters and each tile slot's coordinates and cluster, as the tables were last
   * built (TileDirectory). */
  private clusters: TileBox[] = [];
  private clusterTiles = new Int32Array(0);
  /** Tiles with content at the latest measurement, each with the bricks that held it
   * (bits x, 4 + y and 8 + z, fluid-content.wgsl), and the step it measured. */
  private contentTiles: { tile: Vec3; bricks: number }[] = [];
  private contentStep = -1;
  /** The renderer's boxes, the clusters they belong to, and the tile table naming each
   * tile's box (RenderBricks). */
  private renderBoxes: BrickBox[] = [];
  private renderClusters: number[] = [];
  private renderTiles = new Int32Array(0);
  private renderVersion = 0;
  /** The tiles changed since the boxes were last made. */
  private renderStale = true;
  /** Indirect dispatch sizes for the active and zeroed bricks. */
  private dispatchArgs: GPUBuffer;
  /** The pressure buffer holding the latest solution. */
  private latestPressure!: GPUBuffer;
  private activityReads: GPUBuffer[] = [];
  /** Includes idle and mapping buffers so reset/dispose cannot lose an in-flight
   * readback whose promise resolves after the simulation was released. */
  private activityBuffers = new Set<GPUBuffer>();
  /** Counts resets: measurements of steps before the last one are dropped. */
  private epoch = 0;
  private pipelines = new Map<string, GPUComputePipeline>();
  private pressureSolver?: PressureSolver;
  private bindings = new Map<string, GPUBindGroup>();
  private ids = new WeakMap<object, number>();
  private nextId = 0;
  private forceFields: ForceFields;
  /** Parameters for kernels that write the field grid and the velocity grid. */
  private fieldUniform: GPUBuffer;
  private smokeUniform: GPUBuffer;
  private velocityUniform: GPUBuffer;
  /** The slots a grown pool added (fluid-pool-growth.wgsl PoolGrowth). */
  private growthUniform: GPUBuffer;
  private flameUniform: GPUBuffer;
  private flameRamps: GPUBuffer;
  /** The scene's storage buffers, grown to fit what the host packs into them each step:
   * emitters, colliders and mesh collider occupancy, mesh emission and its tables by
   * brick, forces and their mesh coverage, and what reaches each tile. */
  private scene = new Map<SceneBuffer, GPUBuffer>();
  /** Each slot's solid cells, one bit a field cell (solids.wgsl). */
  private solids!: GPUBuffer;
  /** The colliders changed: draw them into every brick's solid cells. */
  private solidsStale = true;
  /** Mesh collider occupancy as uploaded, and each mask's first word there. */
  private occupancy: MeshColliderMask[] = [];
  private occupancyOffsets: number[] = [];
  /** Mesh emission records as uploaded, and the brick ids their tables follow. */
  private meshRecords?: ArrayBuffer;
  private meshVersion = -1;
  private meshCount = 0;
  /** World bounds of each mesh source's samples. */
  private meshBounds = new WeakMap<object, { lo: Vec3; hi: Vec3 }>();
  private uploads: BufferUploads;
  private meshRasterizer: MeshSourceRasterizer;
  private sampler: GPUSampler;
  private noise?: { texture: GPUTexture; sampler: GPUSampler };
  private uniformData = new Float32Array(48);
  /** Conservative lifetime history. Once fire is possible, retain its transport until
   * reset: removing an emitter or disabling fuel does not remove existing flame. */
  private mayHaveFlame = false;
  /** Below the ground, world y < 0, is solid, and no tile is created there. */
  ground = true;
  /** Colliders of the latest step. */
  private colliderCount = 0;
  private constructor(device: GPUDevice, grid: SimulationGrid) {
    this.device = device;
    this.uploads = new BufferUploads(device.queue);
    if (!Number.isFinite(grid.voxelSize) || grid.voxelSize < 0.0001 || grid.voxelSize > 16)
      throw new Error('voxelSize must be from 0.0001 to 16 meters.');
    this.voxelSize = grid.voxelSize;
    this.velocityDivisor = grid.velocityDivisor ?? 1;
    if (!VELOCITY_DIVISORS.includes(this.velocityDivisor))
      throw new Error('velocityDivisor must be 1, 2 or 4.');
    this.brickCells = grid.brickSize ?? 16;
    if (!BRICK_SIZES.includes(this.brickCells))
      throw new Error('brickSize must be 8, 16 or 32 fine cells.');
    this.brickShift = Math.log2(this.brickCells);
    this.velocityBrickCells = this.brickCells / this.velocityDivisor;
    this.velocityBrickShift = Math.log2(this.velocityBrickCells);
    this.smokeDivisor = grid.smokeDivisor ?? 1;
    this.scalarMacCormack = grid.scalarMacCormack ?? true;
    if (typeof this.scalarMacCormack !== 'boolean')
      throw new Error('scalarMacCormack must be a boolean.');
    if (!SMOKE_DIVISORS.includes(this.smokeDivisor))
      throw new Error('smokeDivisor must be 1, 2 or 4.');
    this.smokeBrickCells = this.brickCells / this.smokeDivisor;
    this.poolSide = poolSide(this.velocityDivisor, device.limits, this.brickCells);
    this.maxSlots = this.poolSide ** 2 * poolSlotsPerAxis(this.brickCells, device.limits);
    // Mesh sources use the field lattice.
    const spacing: Vec3 = [grid.voxelSize, grid.voxelSize, grid.voxelSize];
    this.meshRasterizer = new MeshSourceRasterizer(spacing);
    // Forces are evaluated where they are applied: on the velocity grid.
    this.forceFields = new ForceFields(
      spacing.map((n) => n * this.velocityDivisor) as Vec3,
      this.velocityBrickCells,
    );
    const buffer = (label: string, size: number, usage: number) =>
      device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
    this.fieldUniform = buffer('Field grid parameters', 192, GPUBufferUsage.UNIFORM);
    this.smokeUniform = buffer('Smoke grid parameters', 192, GPUBufferUsage.UNIFORM);
    this.velocityUniform = buffer('Velocity grid parameters', 192, GPUBufferUsage.UNIFORM);
    this.growthUniform = buffer('Pool growth', 16, GPUBufferUsage.UNIFORM);
    this.flameUniform = buffer('Flame evolution parameters', 48, GPUBufferUsage.UNIFORM);
    this.flameRamps = buffer(
      'Flame response ramps',
      FLAME_CHANNELS.length * MAX_FLAME_KNOTS * 8,
      GPUBufferUsage.STORAGE,
    );
    for (const name of SCENE_BUFFERS) this.upload(name, new Uint32Array(4));
    this.dispatchArgs = buffer(
      'Brick dispatch sizes',
      156,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
    this.sampler = device.createSampler({
      label: 'Trilinear advection',
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });
    this.allocate();
  }
  static async create(device: GPUDevice, grid: SimulationGrid): Promise<FluidSimulation> {
    for (const feature of ['shader-f16', 'texture-formats-tier1'] as const)
      if (!device.features.has(feature))
        throw new Error(
          `Fire Pro requires the WebGPU device feature "${feature}" for 16-bit simulation storage.`,
        );
    const sim = new FluidSimulation(device, grid);
    try {
      const module = device.createShaderModule({
        label: 'Three.js Fire Pro fluid solver',
        code: sim.scalarMacCormack ? fluidSource : fluidFirstOrderSource,
      });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length)
        throw new Error(errors.map((m) => `Fluid WGSL ${m.lineNum}: ${m.message}`).join('\n'));
      await Promise.all(
        Object.entries(FLUID_KERNELS).map(async ([entryPoint, bindings]) => {
          const layout = device.createPipelineLayout({
            bindGroupLayouts: [
              device.createBindGroupLayout({
                entries: bindings.map((binding) => {
                  const entry = fluidBindingLayout(binding);
                  if (binding === 13 && !sim.scalarMacCormack)
                    entry.storageTexture!.format = 'r32uint';
                  return entry;
                }),
              }),
            ],
          });
          sim.pipelines.set(
            entryPoint,
            await device.createComputePipelineAsync({
              label: entryPoint,
              layout,
              compute: {
                module,
                entryPoint,
                ...(SCALAR_TRANSPORT_KERNELS.has(entryPoint)
                  ? { constants: { SCALAR_MACCORMACK: Number(sim.scalarMacCormack) } }
                  : {}),
              },
            }),
          );
        }),
      );
      sim.pressureSolver = await PressureSolver.create(
        device, sim.velocityUniform, sim.velocityBrickCells,
      );
      sim.pressureSolver.setBricks(sim.pressureBricks());
      sim.noise = await createCurlNoise(device);
      return sim;
    } catch (error) {
      sim.dispose();
      throw error;
    }
  }
  /** Velocity cell size in meters. */
  get velocitySpacing(): number {
    return this.voxelSize * this.velocityDivisor;
  }
  /** A tile's side in meters. */
  private get tileSize(): number {
    return TILE_BRICKS * this.brickCells * this.voxelSize;
  }
  /** Conservative count of the source box plus its stencil/motion halo. Match
   * sourceNear in fluid-allocation.wgsl so the old full-brick halo does not over-reserve memory. */
  private sourceBricks(source: SimulationSource, dt: number): number {
    const size = this.brickCells * this.voxelSize;
    return source.size.reduce((count, extent, axis) => {
      const motion = Math.abs(source.velocity[axis]) + Math.abs(source.outwardSpeed);
      const pad = Math.min(size, 2 * this.velocitySpacing + motion * dt);
      // A span of width w overlaps at most ceil(w / brickSize) + 1 bricks.
      return count * (Math.ceil((2 * (Math.max(extent, 0.03) + pad)) / size) + 1);
    }, 1);
  }
  /** Empty pools of one layer of slots, and no tiles. */
  private allocate(): void {
    this.predictorApronDirty = false;
    this.smokePredictorApronDirty = false;
    const slots = initialPoolSlots(this.poolSide, this.slotLimit());
    this.poolRows = poolRows(slots, this.poolSide);
    this.poolSlots = slots;
    this.usedSlots = 0;
    this.poolTarget = 0;
    this.sourceIds.clear();
    this.poolLimited = false;
    this.velocity = this.poolTexture('Velocity A', 'velocity');
    this.velocityBack = this.poolTexture('Velocity B', 'velocity');
    this.curl = this.poolTexture('Curl', 'velocity');
    this.field = this.poolTexture('Smoke output heat lifetime fuel', 'field');
    this.smokePools =
      this.smokeDivisor === 1
        ? []
        : ['Smoke A', 'Smoke predictor', 'Smoke B'].map((label) =>
            this.poolTexture(label, 'smoke', 'r16float'),
          );
    this.fieldForward = this.poolTexture('Advected fields', 'field');
    this.fieldCorrected = this.poolTexture('Corrected fields', 'field');
    this.expansionRate = this.poolTexture('Flame target divergence', 'field', 'r16float');
    this.createDonors();
    // The stack pops slot 0 first.
    this.freeSlots = this.device.createBuffer({
      label: 'Free slots',
      size: slots * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      this.freeSlots,
      0,
      Uint32Array.from({ length: slots }, (_, i) => slots - 1 - i),
    );
    this.poolState = this.device.createBuffer({
      label: 'Free slot count and failed allocations',
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.poolState, 0, new Int32Array([slots, 0, 0, 0]));
    this.pressure = [this.slotScalars('Pressure A', slots), this.slotScalars('Pressure B', slots)];
    this.divergence = this.slotScalars('Divergence', slots);
    this.latestPressure = this.pressure[0];
    this.solids = this.slotSolids(slots);
    this.solidsStale = true;
    this.directory.clear();
    this.contentTiles = [];
    this.contentStep = -1;
    this.tiles = this.createTiles(MIN_TILES);
    this.applyTiles();
    this.updateBoxes([]);
    this.bindings.clear();
  }
  /** A pool texture with the current rows of slots: a brick plus an apron per slot. */
  private poolTexture(
    label: string,
    grid: Grid,
    format: GPUTextureFormat = 'rgba16float',
    rows = this.poolRows,
  ): GPUTexture {
    const cells = grid === 'field'
      ? this.brickCells
      : grid === 'smoke' ? this.smokeBrickCells : this.velocityBrickCells;
    const edge = cells + 1;
    return this.device.createTexture({
      label: `${label} pool`,
      size: poolSize(rows, this.poolSide, edge),
      dimension: '3d',
      format,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
  }
  /** Shared scratch: four packed bounds words per texel with Fine Detail enabled,
   * otherwise one coordinate word. Velocity uses X in two texels. */
  private createDonors(rows = this.poolRows): void {
    const size = poolSize(rows, this.poolSide, this.brickCells);
    if (this.velocityDivisor === 1) size[0] *= 2;
    this.donors = this.device.createTexture({
      label: this.scalarMacCormack ? 'Shared compact donor bounds' : 'Velocity trace donors',
      size,
      dimension: '3d',
      format: this.scalarMacCormack ? 'rgba32uint' : 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
  }
  /** Half-precision scalars of every velocity cell of `slots` slots. */
  private slotScalars(label: string, slots: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size: slots * this.velocityBrickCells ** 3 * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }
  /** The solid cells of `slots` slots: a bit per field cell. */
  private slotSolids(slots: number): GPUBuffer {
    return this.device.createBuffer({
      label: 'Solid cells',
      size: (slots * this.brickCells ** 3) / 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }
  /** Write `data` to a scene buffer, growing it to fit: half again what it needs. */
  private upload(name: SceneBuffer, data: ArrayBufferView<ArrayBuffer>): boolean {
    const bytes = Math.max(16, SCENE_ELEMENT[name], Math.ceil(data.byteLength / 4) * 4);
    let buffer = this.scene.get(name);
    if (!buffer || buffer.size < bytes) {
      if (bytes > this.device.limits.maxStorageBufferBindingSize)
        throw new Error(`The scene's ${name} exceed the GPU storage-buffer limit.`);
      buffer?.destroy();
      buffer = this.device.createBuffer({
        label: `Scene ${name}`,
        size: Math.min(
          this.device.limits.maxStorageBufferBindingSize,
          Math.ceil((bytes * 1.5) / 16) * 16,
        ),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.scene.set(name, buffer);
      this.bindings.clear();
    }
    return this.uploads.write(buffer, data);
  }
  /** A 2D integer texture of the tile directory (tiles.wgsl). */
  private directoryTexture(label: string, format: GPUTextureFormat, size: [number, number]) {
    return this.device.createTexture({
      label,
      size,
      format,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
  }
  /** Fill rows `from` to `to` of the brick entries with -1: bricks without slots. */
  private clearEntries(entries: GPUTexture, from: number, to: number): void {
    if (to <= from) return;
    this.device.queue.writeTexture(
      { texture: entries, origin: [0, from] },
      new Int32Array((to - from) * TILE_BRICK_COUNT).fill(-1),
      { bytesPerRow: TILE_BRICK_COUNT * 4 },
      [TILE_BRICK_COUNT, to - from],
    );
  }
  /** Tile resources for `capacity` tile slots: no brick has a slot. */
  private createTiles(capacity: number): TileResources {
    const limit = this.device.limits.maxTextureDimension2D;
    if (capacity > limit) throw new Error(`The simulation needs more than ${limit} tiles.`);
    const bricks = capacity * TILE_BRICK_COUNT;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, size: number) =>
      this.device.createBuffer({ label, size, usage });
    const pages = buffer('Brick pages', bricks * 4);
    this.device.queue.writeBuffer(pages, 0, new Int32Array(bricks).fill(-1));
    const entries = this.directoryTexture('Brick entries', 'r32sint', [TILE_BRICK_COUNT, capacity]);
    this.clearEntries(entries, 0, capacity);
    return {
      capacity,
      records: this.directoryTexture('Tile records', 'rgba32sint', [
        TILE_RECORD_WORDS / 4,
        capacity,
      ]),
      entries,
      brickRecords: this.directoryTexture('Brick records', 'rgba32sint', [
        (TILE_BRICK_COUNT * BRICK_RECORD_WORDS) / 4,
        capacity,
      ]),
      pages,
      activity: buffer('Brick and tile content', (bricks + capacity) * 4),
      brickState: buffer('Brick state', bricks * 4),
      activeBricks: buffer('Active bricks', (bricks + 1) * 4),
      // A brick is zeroed at most twice a step: a fresh slot it stops computing in.
      zeroBricks: buffer('Zeroed bricks', (4 * bricks + 1) * 4),
      blockRanks: buffer('Slot ranks per block of bricks', Math.ceil(bricks / 256) * 8),
    };
  }
  private destroyTiles(tiles: TileResources): void {
    for (const [key, value] of Object.entries(tiles))
      if (key !== 'capacity') (value as GPUBuffer | GPUTexture).destroy();
  }
  /** Upload the directory after its tiles changed: grow the tile resources to hold every
   * tile slot, keeping each brick's page, state, flags, entry and record; and start new
   * tiles' bricks without slots. */
  private applyTiles(): void {
    const directory = this.directory;
    const needed = Math.max(MIN_TILES, directory.slotsUsed);
    if (needed > this.tiles.capacity) {
      let capacity = this.tiles.capacity;
      while (capacity < needed) capacity *= 2;
      const old = this.tiles;
      const next = this.createTiles(capacity);
      const encoder = this.device.createCommandEncoder({ label: 'Grow tiles' });
      const bricks = old.capacity * TILE_BRICK_COUNT * 4;
      for (const key of ['pages', 'brickState', 'activity'] as const)
        encoder.copyBufferToBuffer(old[key], 0, next[key], 0, bricks);
      for (const key of ['entries', 'brickRecords'] as const)
        encoder.copyTextureToTexture({ texture: old[key] }, { texture: next[key] }, [
          old[key].width,
          old.capacity,
        ]);
      this.device.queue.submit([encoder.finish()]);
      this.destroyTiles(old);
      this.tiles = next;
    }
    const tables = directory.tables(this.tiles.capacity);
    this.clusters = tables.clusters;
    this.clusterTiles = tables.tiles;
    this.renderStale = true;
    const queue = this.device.queue;
    queue.writeTexture(
      { texture: this.tiles.records },
      tables.records,
      { bytesPerRow: TILE_RECORD_WORDS * 4 },
      [TILE_RECORD_WORDS / 4, this.tiles.capacity],
    );
    const rows = tables.hash.length / 4 / HASH_ROW;
    if (this.tileTable?.height !== rows) {
      this.tileTable?.destroy();
      this.tileTable = this.directoryTexture('Tile hash', 'rgba32sint', [HASH_ROW, rows]);
    }
    queue.writeTexture({ texture: this.tileTable }, tables.hash, { bytesPerRow: HASH_ROW * 16 }, [
      HASH_ROW,
      rows,
    ]);
    const empty = new Int32Array(TILE_BRICK_COUNT).fill(-1);
    const zero = new Uint32Array(TILE_BRICK_COUNT);
    for (const slot of directory.takeCreated()) {
      const offset = slot * TILE_BRICK_COUNT * 4;
      queue.writeBuffer(this.tiles.pages, offset, empty);
      queue.writeBuffer(this.tiles.brickState, offset, zero);
      queue.writeBuffer(this.tiles.activity, offset, zero);
      this.clearEntries(this.tiles.entries, slot, slot + 1);
    }
    this.bindings.clear();
    this.pressureSolver?.setBricks(this.pressureBricks());
  }
  private pressureBricks(): PressureBricks {
    return {
      list: this.tiles.activeBricks,
      zeroed: this.tiles.zeroBricks,
      dispatch: this.dispatchArgs,
      offsets: {
        tiles: DISPATCH.velocity,
        octets: DISPATCH.brickOctet,
        zeroed: DISPATCH.zeroedBrick,
        coarse: [96, 108, 120, 132],
      },
      solids: this.solids,
      directory: {
        tiles: this.tiles.records,
        table: this.tileTable,
        entries: this.tiles.entries,
        bricks: this.tiles.brickRecords,
      },
      tiles: this.tiles.capacity,
      slots: this.poolSlots,
    };
  }
  /** Most slots the pools may hold: the voxel budget, within the device's limits. */
  private slotLimit(): number {
    const voxels = Math.min(
      this.voxelBudget,
      maxPoolVoxels(this.velocityDivisor, this.device.limits, this.brickCells),
    );
    return Math.max(1, Math.min(this.maxSlots, Math.floor(voxels / this.brickCells ** 3)));
  }
  reset(): void {
    this.epoch++;
    this.releaseActivityReads();
    this.releasePools();
    this.destroyTiles(this.tiles);
    this.allocate();
    this.time = 0;
    this.steps = 0;
    this.mayHaveFlame = false;
    this.poolGrownAt = 0;
    this.activeBrickCount = 0;
  }
  /** What the renderer reads of the bricks. */
  get renderBricks(): RenderBricks {
    return {
      version: this.renderVersion,
      boxes: this.renderBoxes,
      tiles: this.renderTiles,
      pages: this.tiles.pages,
      active: this.tiles.activeBricks,
      zeroed: this.tiles.zeroBricks,
      dispatch: this.dispatchArgs,
      activeOffset: DISPATCH.brick,
      zeroedOffset: DISPATCH.zeroedBrick,
      fieldOffset: DISPATCH.field,
      smokeOffset: DISPATCH.smokeRender,
    };
  }
  /** Requested bytes of owned GPU resources, including capacity, uniform buffers and
   * pending readbacks. Scene mesh/force data has separate getters; rendering owns its
   * page table. This excludes backend alignment and transient growth overlap. */
  get memoryBytes(): number {
    const bytes = (texture: GPUTexture, texel: number) =>
      texture.width * texture.height * texture.depthOrArrayLayers * texel;
    const vectors = new Set([
      this.velocity,
      this.velocityBack,
      this.curl,
      this.field,
      this.fieldForward,
      this.fieldCorrected,
    ]);
    let total = bytes(this.expansionRate, 2) + bytes(this.donors, this.scalarMacCormack ? 16 : 4);
    for (const texture of vectors) total += bytes(texture, 8);
    for (const texture of this.smokePools) total += bytes(texture, 2);
    for (const buffer of [
      ...this.pressure,
      this.divergence,
      this.solids,
      this.freeSlots,
      this.poolState,
      this.fieldUniform,
      this.smokeUniform,
      this.velocityUniform,
      this.growthUniform,
      this.flameUniform,
      this.flameRamps,
      this.dispatchArgs,
      this.tiles.pages,
      this.tiles.brickState,
      this.tiles.activity,
      this.tiles.activeBricks,
      this.tiles.zeroBricks,
      this.tiles.blockRanks,
      ...this.activityBuffers,
    ])
      total += buffer.size;
    total +=
      bytes(this.tiles.records, 16) +
      bytes(this.tiles.entries, 4) +
      bytes(this.tiles.brickRecords, 16) +
      bytes(this.tileTable, 16);
    for (const name of ['emitters', 'colliders', 'occupancy', 'tileLists'] as const)
      total += this.scene.get(name)!.size;
    total += this.pressureSolver?.memoryBytes ?? 0;
    if (this.noise) total += bytes(this.noise.texture, 8);
    return total;
  }
  get forceMemoryBytes(): number {
    return this.scene.get('forces')!.size + this.scene.get('coverage')!.size;
  }
  get meshMemoryBytes(): number {
    return this.scene.get('meshSources')!.size + this.scene.get('meshTables')!.size;
  }
  /** Need the tiles within `margin` meters of the world box `lo` to `hi`. */
  private needBox(lo: readonly number[], hi: readonly number[], margin: number): void {
    const size = this.tileSize;
    this.directory.need(
      lo.map((value) => Math.floor((value - margin) / size)) as Vec3,
      hi.map((value) => Math.floor((value + margin) / size)) as Vec3,
      this.steps,
    );
  }
  /** The bricks that overlap the world box `lo` to `hi`. */
  private bricksOver(lo: readonly number[], hi: readonly number[]): BrickBox {
    const size = this.brickCells * this.voxelSize;
    return {
      lo: lo.map((value) => Math.floor(value / size)) as Vec3,
      hi: hi.map((value) => Math.floor(value / size)) as Vec3,
    };
  }
  /** Keep the tiles the step needs: around every source, the bricks it reaches (two
   * bricks, buildBricks in fluid-allocation.wgsl) and a tile of margin; and around every tile that
   * held content, a tile of margin. Tiles nothing needs linger, then close. Then box what
   * the renderer draws. */
  private updateTiles(sources: readonly SimulationSource[]): void {
    this.directory.retire(this.steps);
    // The bricks a source reaches hold its content; a tile past them is margin.
    const reach = 2 * this.brickCells * this.voxelSize;
    const margin = reach + this.tileSize;
    const sourceBricks: BrickBox[] = [];
    for (const source of sources) {
      if (source.samples?.length) {
        let bounds = this.meshBounds.get(source.samples);
        if (!bounds) {
          const lo: Vec3 = [Infinity, Infinity, Infinity];
          const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
          for (const { position } of source.samples)
            for (let axis = 0; axis < 3; axis++) {
              lo[axis] = Math.min(lo[axis], position[axis]);
              hi[axis] = Math.max(hi[axis], position[axis]);
            }
          bounds = { lo, hi };
          this.meshBounds.set(source.samples, bounds);
        }
        this.needBox(bounds.lo, bounds.hi, margin + this.voxelSize);
        sourceBricks.push(this.bricksOver(bounds.lo, bounds.hi));
        continue;
      }
      // Bursts throw material outward; leave room for a few frames of it.
      const extent = Math.max(...source.size, 0.03) + source.outwardSpeed * 0.25;
      const lo = source.position.map((value) => value - extent);
      const hi = source.position.map((value) => value + extent);
      this.needBox(lo, hi, margin);
      sourceBricks.push(this.bricksOver(lo, hi));
    }
    for (const { tile } of this.contentTiles)
      this.directory.need(
        tile.map((n) => n - 1) as Vec3,
        tile.map((n) => n + 1) as Vec3,
        this.steps,
      );
    if (this.directory.changed) this.applyTiles();
    this.updateBoxes(sourceBricks);
  }
  /** Box what the renderer draws in each cluster of tiles: the bricks with content at the
   * latest measurement and the bricks of `extra` boxes, with margin,
   * within the cluster's tiles. Clusters with none of these get no box. */
  private updateBoxes(extra: readonly BrickBox[]): void {
    const found = new Map<number, BrickBox>();
    // What a box must hold lies in the cluster of its center: the tiles around content
    // and sources touch.
    const include = (box: BrickBox) => {
      const center = box.lo.map((n, axis) => Math.floor((n + box.hi[axis]) / 2));
      if (this.ground) center[1] = Math.max(0, center[1]);
      const slot = this.directory.slotAt(center.map((n) => Math.floor(n / TILE_BRICKS)));
      if (slot < 0) return;
      const cluster = this.clusterTiles[slot * 4 + 3];
      found.set(cluster, union(found.get(cluster), box));
    };
    for (const { tile, bricks } of this.contentTiles) {
      const used = [0, 1, 2].map((axis) => (bricks >> (axis * 4)) & 15);
      include({
        lo: tile.map((n, axis) => n * TILE_BRICKS + Math.log2(used[axis] & -used[axis])) as Vec3,
        hi: tile.map((n, axis) => n * TILE_BRICKS + Math.floor(Math.log2(used[axis]))) as Vec3,
      });
    }
    for (const box of extra) include(box);
    const clusters = [...found.keys()].sort((a, b) => a - b);
    const boxes = clusters.map((cluster) => {
      const box = found.get(cluster)!;
      const tiles = this.clusters[cluster];
      return {
        lo: box.lo.map((n, axis) => Math.max(n - BOX_MARGIN, tiles.lo[axis] * TILE_BRICKS)) as Vec3,
        hi: box.hi.map((n, axis) =>
          Math.min(n + BOX_MARGIN, (tiles.hi[axis] + 1) * TILE_BRICKS - 1),
        ) as Vec3,
      };
    });
    const same =
      clusters.length === this.renderClusters.length &&
      clusters.every((cluster, i) => cluster === this.renderClusters[i]) &&
      boxes.every((box, i) =>
        [0, 1, 2].every(
          (axis) =>
            box.lo[axis] === this.renderBoxes[i].lo[axis] &&
            box.hi[axis] === this.renderBoxes[i].hi[axis],
        ),
      );
    if (same && !this.renderStale) return;
    this.renderStale = false;
    this.renderBoxes = boxes;
    this.renderClusters = clusters;
    this.renderTiles = this.clusterTiles.slice();
    for (let word = 3; word < this.renderTiles.length; word += 4)
      if (this.renderTiles[word] >= 0)
        this.renderTiles[word] = clusters.indexOf(this.renderTiles[word]);
    this.renderVersion++;
  }
  /** Decide each brick's state, free the slots bricks no longer want and give slots to
   * bricks that want one, in brick order; then build the neighbor tables, and with
   * `colliders`, draw them into the new slots' solid cells, or into every slot's after
   * they changed. */
  private encodeAllocation(pass: ComputePass, colliders: boolean): void {
    for (const name of [
      'buildBricks',
      'freeBricks',
      'allocateBricks',
      'linkBricks',
      'finishBricks',
    ])
      this.encode(pass, name);
    if (colliders || this.solidsStale) this.encode(pass, 'buildSolids');
  }
  /** Zero the listed bricks in every pool, and their pressure. */
  private encodeZeroing(pass: ComputePass): void {
    const [pressure, previous] = this.pressure;
    this.encode(
      pass,
      'zeroVelocityBricks',
      {
        6: this.velocity,
        58: this.velocityBack,
        59: this.curl,
        7: pressure,
        60: previous,
        61: this.divergence,
      },
      { grid: 'velocity' },
    );
    this.encode(
      pass,
      'zeroFieldBricks',
      { 6: this.field, 58: this.fieldForward, 59: this.fieldCorrected, 16: this.expansionRate },
      { grid: 'field' },
    );
    if (this.smokeDivisor > 1)
      this.encode(pass, 'zeroSmokeBricks', {
        16: this.smokePools[0],
        74: this.smokePools[1],
        75: this.smokePools[2],
      });
  }
  /** Grow the pools to hold `target` slots, keeping only live state. Predictor,
   * correction and bounds scratch is rewritten before it is next sampled. */
  private growPool(target: number): void {
    const limit = this.slotLimit();
    const rows = poolRows(Math.min(target, limit), this.poolSide);
    const slots = Math.min(limit, rows * this.poolSide);
    if (slots <= this.poolSlots) return;
    const encoder = this.device.createCommandEncoder({ label: 'Grow brick pools' });
    const retired: Resource[] = [];
    const grow = (texture: GPUTexture, grid: Grid, preserve = false) => {
      const next = this.poolTexture(
        texture.label.replace(/ pool$/, ''),
        grid,
        texture.format,
        rows,
      );
      if (preserve)
        encoder.copyTextureToTexture({ texture }, { texture: next }, [
          texture.width,
          texture.height,
          texture.depthOrArrayLayers,
        ]);
      retired.push(texture);
      return next;
    };
    this.velocity = grow(this.velocity, 'velocity', true);
    this.velocityBack = grow(this.velocityBack, 'velocity');
    this.curl = grow(this.curl, 'velocity');
    this.field = grow(this.field, 'field', true);
    this.fieldForward = grow(this.fieldForward, 'field');
    this.fieldCorrected = grow(this.fieldCorrected, 'field');
    this.expansionRate = grow(this.expansionRate, 'field', true);
    this.smokePools = this.smokePools.map((texture, i) => grow(texture, 'smoke', i === 0));
    retired.push(this.donors);
    this.createDonors(rows);
    // Carry the latest pressure over to the new buffers.
    const pressure = [this.slotScalars('Pressure A', slots), this.slotScalars('Pressure B', slots)];
    const latest = this.pressure.indexOf(this.latestPressure);
    encoder.copyBufferToBuffer(
      this.latestPressure,
      0,
      pressure[latest],
      0,
      this.latestPressure.size,
    );
    retired.push(...this.pressure, this.divergence);
    this.pressure = pressure as [GPUBuffer, GPUBuffer];
    this.latestPressure = pressure[latest];
    this.divergence = this.slotScalars('Divergence', slots);
    // Slots keep their solid cells.
    const solids = this.slotSolids(slots);
    encoder.copyBufferToBuffer(this.solids, 0, solids, 0, this.solids.size);
    retired.push(this.solids);
    this.solids = solids;
    // The new slots go under the free stack: move it up to make room.
    const added = slots - this.poolSlots;
    const freeSlots = this.device.createBuffer({
      label: 'Free slots',
      size: slots * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(this.freeSlots, 0, freeSlots, added * 4, this.poolSlots * 4);
    retired.push(this.freeSlots);
    this.freeSlots = freeSlots;
    this.device.queue.writeBuffer(
      this.growthUniform,
      0,
      new Uint32Array([this.poolSlots, added, 0, 0]),
    );
    this.bindings.clear();
    const pass = beginComputePass(this.device, encoder, { label: 'Grow free list' });
    this.encode(pass, 'growFreeList', {}, { workgroups: [Math.ceil(added / 64), 1, 1] });
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    for (const resource of retired) resource.destroy();
    // The replacement predictor starts zeroed; none of its old render data is copied.
    this.predictorApronDirty = false;
    this.smokePredictorApronDirty = false;
    this.poolRows = rows;
    this.poolSlots = slots;
    this.poolGrownAt = this.steps;
    this.pressureSolver?.setBricks(this.pressureBricks());
  }
  private releasePools(): void {
    for (const buffer of [...this.pressure, this.divergence, this.solids]) buffer.destroy();
    for (const texture of new Set([
      this.velocity,
      this.velocityBack,
      this.curl,
      this.field,
      this.fieldForward,
      this.fieldCorrected,
      this.expansionRate,
      this.donors,
      ...this.smokePools,
    ]))
      texture.destroy();
    this.freeSlots.destroy();
    this.poolState.destroy();
    this.bindings.clear();
  }
  /** Write the grid parameters: the last frame's, with the cell sizes, the pools' layout
   * and the brick ids. Slots past the voxel budget are held back from allocation. */
  private writeUniforms(): void {
    const data = this.uniformData;
    data[14] = this.smokeDivisor;
    data[15] = Math.log2(this.smokeBrickCells);
    const velocity = this.velocitySpacing;
    data.set([velocity, velocity, velocity, this.velocityBrickShift], 20);
    data.set([this.voxelSize, this.voxelSize, this.voxelSize], 24);
    const held = Math.max(0, this.poolSlots - this.slotLimit());
    data.set([this.poolSide, this.poolSide, held, this.brickShift], 28);
    data.set([this.tiles.capacity * TILE_BRICK_COUNT, 0, 0, 0], 32);
    data.set([this.voxelSize, this.voxelSize, this.voxelSize], 0);
    data[39] = Number(this.predictorApronDirty);
    this.device.queue.writeBuffer(this.fieldUniform, 0, data);
    data.set([velocity, velocity, velocity], 0);
    this.device.queue.writeBuffer(this.velocityUniform, 0, data);
    data.set(
      [1, 1, 1].map(() => this.voxelSize * this.smokeDivisor),
      0,
    );
    data[39] = Number(this.smokePredictorApronDirty);
    this.device.queue.writeBuffer(this.smokeUniform, 0, data);
    data[39] = Number(this.predictorApronDirty);
  }
  /** The id of world brick `brick`, or -1 outside every tile (tiles.wgsl). */
  private brickId(brick: readonly number[]): number {
    const slot = this.directory.slotAt(brick.map((n) => Math.floor(n / TILE_BRICKS)));
    if (slot < 0) return -1;
    const [x, y, z] = brick.map((n) => n & (TILE_BRICKS - 1));
    return slot * TILE_BRICK_COUNT + (z * TILE_BRICKS + y) * TILE_BRICKS + x;
  }
  /**
   * Pack the step's emitters, colliders, forces and mesh emission into the scene buffers,
   * and list for every tile what reaches it: emitters within the two bricks a source
   * allocates, forces within a brick, and colliders. Changed colliders redraw every
   * slot's solid cells. Returns the counts the parameters need.
   */
  private packScene(frame: SimulationFrame): { colliders: number; meshRecords: number } {
    const tileSize = this.tileSize;
    const brick = this.brickCells * this.voxelSize;
    const capacity = this.tiles.capacity;
    const coordinates = this.directory.coordinates;
    const lists = [0, 1, 2].map(() => Array.from({ length: capacity }, (): number[] => []));
    // Every tile within the world box `lo` to `hi`, or every tile without one.
    const bin = (
      kind: number,
      index: number,
      box?: { lo: readonly number[]; hi: readonly number[] },
    ) => {
      const lo = box?.lo.map((n) => Math.floor(n / tileSize));
      const hi = box?.hi.map((n) => Math.floor(n / tileSize));
      const span =
        lo && hi
          ? hi.reduce((count, n, axis) => count * Math.max(0, n - lo[axis] + 1), 1)
          : Infinity;
      if (span > this.directory.count) {
        coordinates.forEach((coord, slot) => {
          if (coord && (!lo || coord.every((n, axis) => n >= lo[axis] && n <= hi![axis])))
            lists[kind][slot].push(index);
        });
        return;
      }
      for (let z = lo![2]; z <= hi![2]; z++)
        for (let y = lo![1]; y <= hi![1]; y++)
          for (let x = lo![0]; x <= hi![0]; x++) {
            const slot = this.directory.slotAt([x, y, z]);
            if (slot >= 0) lists[kind][slot].push(index);
          }
    };
    const grow = (lo: readonly number[], hi: readonly number[], by: number) => ({
      lo: lo.map((n) => n - by),
      hi: hi.map((n) => n + by),
    });
    // Emitters (fluid-common.wgsl Emitter).
    const emitters = frame.sources.filter((source) => source.shape !== 'mesh');
    const emitterData = new Float32Array(Math.max(1, emitters.length) * 20);
    emitters.forEach((source, i) => {
      emitterData.set(
        [
          ...source.position,
          1,
          ...source.size,
          Number(source.shape === 'box'),
          source.flame,
          source.heatRate,
          source.smokeRate,
          source.velocityResponse,
          ...source.velocity,
          source.outwardSpeed,
          source.variation?.period ?? 0.5,
          source.variation?.strength ?? 0,
          source.fuelRate ?? 0,
          0,
        ],
        i * 20,
      );
      const extent = source.size.map((n) => Math.max(n, 0.03));
      bin(
        0,
        i,
        grow(
          source.position.map((n, axis) => n - extent[axis]),
          source.position.map((n, axis) => n + extent[axis]),
          2 * brick,
        ),
      );
    });
    this.upload('emitters', emitterData);
    // Colliders (fluid-common.wgsl Collider), with each mesh occupancy packed to one bit
    // per cell. Every mask starts at a word boundary and retains nonzero-is-solid.
    const occupancy = frame.colliders.flatMap((collider) =>
      collider.shape === 'mesh' ? [collider.occupancy] : [],
    );
    let solidsChanged = false;
    if (
      occupancy.length !== this.occupancy.length ||
      occupancy.some((mask, i) => mask !== this.occupancy[i])
    ) {
      this.occupancyOffsets = [];
      let words = 0;
      for (const mask of occupancy) {
        if (mask.mask.length !== mask.size.reduce((a, b) => a * b, 1))
          throw new Error('Mesh collider occupancy does not match its size.');
        this.occupancyOffsets.push(words);
        words += Math.ceil(mask.mask.length / 32);
      }
      const data = new Uint32Array(Math.max(1, words));
      occupancy.forEach(({ mask }, i) => {
        const first = this.occupancyOffsets[i];
        for (let cell = 0; cell < mask.length; cell++)
          if (mask[cell] !== 0) data[first + (cell >>> 5)] |= 1 << (cell & 31);
      });
      this.upload('occupancy', data);
      this.occupancy = occupancy;
      solidsChanged = true;
    }
    const colliderData = new ArrayBuffer(Math.max(1, frame.colliders.length) * COLLIDER_WORDS * 4);
    const floats = new Float32Array(colliderData);
    const ints = new Int32Array(colliderData);
    let mesh = 0;
    frame.colliders.forEach((collider, i) => {
      const offset = i * COLLIDER_WORDS;
      if (collider.shape === 'mesh') {
        const { lo, size } = collider.occupancy;
        floats.set([0, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], offset);
        ints.set([...lo, this.occupancyOffsets[mesh++], ...size, 0], offset + 20);
        bin(2, i, {
          lo: lo.map((n) => n * this.voxelSize),
          hi: lo.map((n, axis) => (n + size[axis]) * this.voxelSize),
        });
        return;
      }
      const [a, b, c] = collider.axes ?? WORLD_AXES;
      floats.set(
        [
          ...collider.position,
          collider.shape === 'box' ? 1 : 0,
          ...collider.size,
          0,
          ...a,
          0,
          ...b,
          0,
          ...c,
          0,
        ],
        offset,
      );
      // A box's extent along each world axis.
      const extent =
        collider.shape === 'box'
          ? [0, 1, 2].map(
              (axis) =>
                Math.abs(a[axis]) * collider.size[0] +
                Math.abs(b[axis]) * collider.size[1] +
                Math.abs(c[axis]) * collider.size[2],
            )
          : collider.size;
      bin(2, i, {
        lo: collider.position.map((n, axis) => n - extent[axis]),
        hi: collider.position.map((n, axis) => n + extent[axis]),
      });
    });
    if (
      this.upload('colliders', new Uint8Array(colliderData)) ||
      frame.colliders.length !== this.colliderCount
    )
      solidsChanged = true;
    this.colliderCount = frame.colliders.length;
    if (solidsChanged) {
      this.solidsStale = true;
      this.pressureSolver!.invalidateGeometry();
    }
    // Forces, and the velocity cells their mesh targets cover by brick. A force reaches the
    // cells within a brick of a tile, where applying forces averages them.
    const bricks = capacity * TILE_BRICK_COUNT;
    const forces = this.forceFields.prepare(
      frame.forces ?? [],
      bricks,
      this.directory.version,
      (b) => this.brickId(b),
    );
    this.upload('forces', forces.data);
    this.upload('coverage', forces.coverage);
    forces.reach.forEach((reach, i) => bin(1, i, reach && grow(reach.lo, reach.hi, brick)));
    // Mesh emission records, and each brick's table of them.
    const records = this.meshRasterizer.prepare(frame.sources);
    if (records.records !== this.meshRecords)
      this.upload('meshSources', new Uint8Array(records.records));
    if (
      records.records !== this.meshRecords ||
      this.directory.version !== this.meshVersion ||
      bricks !== this.meshCount
    ) {
      this.meshRecords = records.records;
      this.meshVersion = this.directory.version;
      this.meshCount = bricks;
      const cells = new Int32Array(records.records);
      const shift = this.brickShift;
      const mask = this.brickCells - 1;
      const tables = new Map<number, number>();
      const placed: [number, number][] = [];
      for (let record = 1; record <= records.count; record++) {
        const cell = [0, 1, 2].map((axis) => cells[record * 12 + axis]);
        const id = this.brickId(cell.map((n) => n >> shift));
        if (id < 0) continue;
        let table = tables.get(id);
        if (table === undefined) tables.set(id, (table = tables.size));
        placed.push([
          (((((table << shift) | (cell[2] & mask)) << shift) | (cell[1] & mask)) << shift) |
            (cell[0] & mask),
          record,
        ]);
      }
      const data = new Int32Array(bricks + tables.size * this.brickCells ** 3);
      data.fill(-1, 0, bricks);
      for (const [id, table] of tables) data[id] = table;
      for (const [index, record] of placed) data[bricks + index] = record;
      this.upload('meshTables', data);
    }
    // What reaches each tile: a header per tile slot, then the indices.
    let length = capacity * TILE_LIST_WORDS;
    for (const kind of lists) for (const list of kind) length += list.length;
    const tileLists = new Uint32Array(length);
    let next = capacity * TILE_LIST_WORDS;
    for (let slot = 0; slot < capacity; slot++)
      lists.forEach((kind, k) => {
        tileLists.set([next, kind[slot].length], slot * TILE_LIST_WORDS + 2 * k);
        tileLists.set(kind[slot], next);
        next += kind[slot].length;
      });
    this.upload('tileLists', tileLists);
    return { colliders: frame.colliders.length, meshRecords: records.count };
  }
  step(frame: SimulationFrame, dt = FIXED_DT): void {
    if (!Number.isFinite(dt) || dt <= 0 || dt > 1 / 30)
      throw new Error('Simulation step must be in (0, 1/30]. Use fixed substeps.');
    this.substep(frame, dt);
    this.time += dt;
    this.steps++;
  }
  private substep(frame: SimulationFrame, dt: number): void {
    if (this.directory.ground !== this.ground) {
      // The open faces at y = 0 change.
      this.directory.ground = this.ground;
      this.pressureSolver!.invalidateGeometry();
    }
    this.updateTiles(frame.sources);
    // A new source, like a burst that injects for a single step, wants its slots in this
    // step: grow the pool for them now rather than after failed requests are measured.
    let wanted = 0;
    for (const source of frame.sources)
      if (source.shape !== 'mesh' && !this.sourceIds.has(source.id))
        wanted += this.sourceBricks(source, dt);
    this.sourceIds = new Set(frame.sources.map((source) => source.id));
    if (wanted > 0)
      this.poolTarget = Math.max(this.poolTarget, Math.ceil((this.usedSlots + wanted) * 1.5));
    if (this.poolTarget > this.poolSlots) this.growPool(this.poolTarget);
    const s = frame.simulation;
    const scene = this.packScene(frame);
    const packed = packFlameStep(frame.combustion, dt);
    const fuel = frame.fuel ?? DEFAULT_FUEL;
    this.mayHaveFlame ||= fuel.enabled || frame.sources.some((source) => source.flame > 0);
    this.uniformData.set([
      0,
      0,
      0,
      dt,
      s.buoyancy,
      s.cooling,
      s.smokeWeight,
      s.velocityDamping,
      s.dissipation,
      s.vorticity,
      this.cutoff,
      Number(!this.mayHaveFlame),
      this.time,
      s.seed,
      0,
      0,
      Number(this.solidsStale),
      scene.colliders,
      scene.meshRecords,
      Number(this.ground),
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ]);
    this.uniformData.set(
      [fuel.ignitionHeat, fuel.burnRate, Number(fuel.enabled), Number(this.predictorApronDirty)],
      36,
    );
    this.writeUniforms();
    this.uploads.write(this.flameUniform, packed.uniform);
    this.uploads.write(this.flameRamps, packed.ramps);
    // Each step lists its bricks afresh and records new content.
    const zero = new Uint32Array([0]);
    this.device.queue.writeBuffer(this.tiles.activeBricks, 0, zero);
    this.device.queue.writeBuffer(this.tiles.zeroBricks, 0, zero);
    // No tile holds content yet.
    const bricks = this.tiles.capacity * TILE_BRICK_COUNT;
    this.device.queue.writeBuffer(
      this.tiles.activity,
      bricks * 4,
      new Uint32Array(this.tiles.capacity),
    );
    const encoder = this.device.createCommandEncoder({ label: `Fluid step ${this.steps}` });
    let pass = beginComputePass(this.device, encoder, { label: 'Brick allocation' });
    const dispatch = (
      name: string,
      overrides: Record<number, Resource | undefined> = {},
      options: { grid?: Grid } = {},
    ) => this.encode(pass, name, overrides, options);
    this.encodeAllocation(pass, scene.colliders > 0);
    pass.end();
    this.solidsStale = false;
    pass = beginComputePass(this.device, encoder, { label: 'Fluid transport and forces' });
    // Bricks given slots, and bricks that stopped computing, become still, empty air.
    this.encodeZeroing(pass);
    const bounds = this.velocityScratch;
    dispatch('advectVelocityWithBounds', {
      6: this.velocityBack,
      13: this.donors,
      19: bounds[1],
      29: bounds[0],
    });
    // Reuse the curl grid as correction scratch, then reuse the old velocity
    // grid for curl. Every pass has distinct sampled inputs and writable outputs.
    // Advection also built donor bounds in the velocity scratch volumes.
    dispatch('correctVelocity', {
      3: this.velocityBack,
      5: this.donors,
      6: this.curl,
      20: bounds[0],
      21: bounds[1],
    });
    const transported = this.curl,
      curls = this.velocity,
      forced = this.velocityBack;
    // Lifetime travels in the existing field channel. Intermediate texture
    // stores retain the established half-precision rounding between stages.
    dispatch('advectScalars', { 6: this.fieldForward, 13: this.donors });
    // Pipeline specialization selects corrected or first-order scalar transport.
    // Both evolve into a distinct texture, then swap roles with the original fields.
    dispatch('evolveFields', {
      3: this.fieldForward,
      5: this.donors,
      6: this.fieldCorrected,
      16: this.expansionRate,
    });
    [this.field, this.fieldCorrected] = [this.fieldCorrected, this.field];
    if (this.smokeDivisor > 1) {
      // Scalar correction has finished with donor scratch. Coarse smoke can now
      // overwrite it, then consume its cached bounds before velocity/forces change.
      dispatch('advectSmoke', { 13: this.donors, 16: this.smokePools[1] });
      dispatch('evolveSmoke', { 3: this.smokePools[1], 5: this.donors, 16: this.smokePools[2] });
      [this.smokePools[0], this.smokePools[2]] = [this.smokePools[2], this.smokePools[0]];
    }
    if (s.vorticity !== 0) dispatch('computeCurl', { 1: transported, 6: curls });
    // Cell forces, averaged to each component's MAC face (Fedkiw2001 Appendix A).
    dispatch('applyForces', { 1: transported, 3: curls, 6: forced });
    dispatch('computeDivergence', { 1: forced, 7: this.divergence });
    pass.end();
    const [a, b] = this.pressure;
    this.latestPressure = this.pressureSolver!.solve(
      encoder,
      [this.latestPressure, this.latestPressure === a ? b : a],
      this.divergence,
      PRESSURE_CYCLES,
    );
    pass = beginComputePass(this.device, encoder, { label: 'Fluid projection' });
    const projected = this.curl;
    dispatch('project', { 1: forced, 4: this.latestPressure, 6: projected });
    pass.end();
    this.curl = this.velocity;
    this.velocity = projected;
    const read = this.activityRead();
    encoder.copyBufferToBuffer(this.tiles.activeBricks, 0, read, 0, 4);
    encoder.copyBufferToBuffer(this.poolState, 0, read, 4, 8);
    encoder.copyBufferToBuffer(
      this.tiles.activity,
      bricks * 4,
      read,
      READBACK_HEADER,
      this.tiles.capacity * 4,
    );
    this.encodeSolverField(encoder);
    // Failed allocations count from one measurement to the next.
    encoder.clearBuffer(this.poolState, 4, 4);
    this.device.queue.submit([encoder.finish()]);
    this.predictorApronDirty = false;
    this.smokePredictorApronDirty = false;
    this.readActivity(read);
  }
  /** The solver field a debug view shows, copied into `solverFieldTexture` after every
   * step: the pressure, or the divergence it solved for. */
  private solverField?: 'pressure' | 'divergence';
  /** Pressure/divergence occupy the velocity cells and their apron inside each fine
   * correction-scratch slot. Rendering samples this layout with the fine slot stride.
   * No kernel reads this scratch between steps. */
  get solverFieldTexture(): GPUTexture {
    // The first scratch volume may hold the rendering-only smoothed fields.
    return this.velocityScratch[1];
  }
  /** Show `field` in `solverFieldTexture`, now and after every step, or stop copying. */
  showSolverField(field: 'pressure' | 'divergence' | undefined): void {
    this.solverField = field;
    if (!field || !this.steps) return;
    const encoder = this.device.createCommandEncoder({ label: 'Solver field view' });
    this.encodeSolverField(encoder);
    this.device.queue.submit([encoder.finish()]);
  }
  private encodeSolverField(encoder: GPUCommandEncoder): void {
    if (!this.solverField) return;
    const pass = beginComputePass(this.device, encoder, { label: 'Solver field view' });
    this.encode(pass, 'exportScalars', {
      4: this.solverField === 'pressure' ? this.latestPressure : this.divergence,
      6: this.velocityScratch[1],
    });
    pass.end();
  }
  /** Bind and dispatch one fluid kernel. `workgroups` sizes a dense dispatch. */
  private encode(
    pass: ComputePass,
    name: string,
    overrides: Record<number, Resource | undefined> = {},
    options: { grid?: Grid; workgroups?: Vec3 } = {},
  ): void {
    const pipeline = this.pipelines.get(name)!;
    const onSmokeGrid = options.grid === 'smoke' || SMOKE_KERNELS.has(name);
    const onVelocityGrid = options.grid ? options.grid === 'velocity' : VELOCITY_KERNELS.has(name);
    const tiles = this.tiles;
    const resources: Record<number, Resource | GPUSampler | undefined> = {
      ...Object.fromEntries(
        SCENE_BUFFERS.map((buffer) => [SCENE_BINDINGS[buffer], this.scene.get(buffer)]),
      ),
      0: onSmokeGrid
        ? this.smokeUniform
        : onVelocityGrid
          ? this.velocityUniform
          : this.fieldUniform,
      1: this.velocity,
      2: this.field,
      3: this.curl,
      4: this.latestPressure,
      8: this.sampler,
      11: this.noise?.texture,
      12: this.noise?.sampler,
      15: this.expansionRate,
      73: this.smoke,
      22: this.flameUniform,
      23: this.flameRamps,
      36: tiles.activity,
      37: tiles.activeBricks,
      39: tiles.brickState,
      41: tiles.activeBricks,
      43: this.dispatchArgs,
      44: tiles.pages,
      45: tiles.pages,
      46: this.freeSlots,
      47: this.poolState,
      48: tiles.zeroBricks,
      49: tiles.zeroBricks,
      51: this.growthUniform,
      57: tiles.blockRanks,
      63: tiles.records,
      64: this.tileTable,
      65: tiles.entries,
      66: tiles.brickRecords,
      71: this.solids,
      72: this.solids,
      67: tiles.brickRecords,
      69: tiles.entries,
      ...overrides,
    };
    const bindings = FLUID_KERNELS[name];
    const key = name + ':' + bindings.map((binding) => this.id(resources[binding])).join('|');
    let group = this.bindings.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        label: name,
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => {
          const resource = resources[binding];
          if (!resource) throw new Error(`${name} has nothing bound at ${binding}.`);
          return {
            binding,
            resource: isBufferBinding(binding)
              ? { buffer: resource as GPUBuffer }
              : fluidBindingLayout(binding).sampler
                ? (resource as GPUSampler)
                : (resource as GPUTexture).createView(),
          };
        }),
      });
      this.bindings.set(key, group);
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    const bricks = tiles.capacity * TILE_BRICK_COUNT;
    if (options.workgroups) pass.dispatchWorkgroups(...options.workgroups);
    else if (SMOKE_KERNELS.has(name))
      pass.dispatchWorkgroupsIndirect(
        this.dispatchArgs,
        name === 'zeroSmokeBricks' ? DISPATCH.zeroedBrick : DISPATCH.brick,
      );
    else if (LISTED.has(name)) {
      const tall = name === 'advectVelocityWithBounds' || name === 'correctVelocity';
      pass.dispatchWorkgroupsIndirect(
        this.dispatchArgs,
        onVelocityGrid ? (tall ? DISPATCH.velocityTall : DISPATCH.velocity) : DISPATCH.field,
      );
    } else if (ZEROING.has(name))
      pass.dispatchWorkgroupsIndirect(
        this.dispatchArgs,
        onVelocityGrid ? DISPATCH.zeroedVelocity : DISPATCH.zeroedField,
      );
    else if (name === 'buildBricks') pass.dispatchWorkgroups(Math.ceil(bricks / 256));
    else if (name === 'finishBricks') pass.dispatchWorkgroups(1);
    else if (name === 'buildSolids') {
      // A workgroup per brick: every brick id after the colliders changed, else the new slots.
      if (this.solidsStale)
        pass.dispatchWorkgroups(Math.min(bricks, LIST_SPLIT), Math.ceil(bricks / LIST_SPLIT));
      else pass.dispatchWorkgroupsIndirect(this.dispatchArgs, DISPATCH.zeroedBrick);
    }
    // The rest of the allocator runs a lane per brick id.
    else pass.dispatchWorkgroups(Math.ceil(bricks / 64));
  }
  /** A number naming each bound resource, for bind group reuse. */
  private id(resource: object | undefined): number {
    if (!resource) return -1;
    let id = this.ids.get(resource);
    if (id === undefined) this.ids.set(resource, (id = this.nextId++));
    return id;
  }
  private activityRead(): GPUBuffer {
    const size = READBACK_HEADER + this.tiles.capacity * 4;
    this.activityReads = this.activityReads.filter((buffer) => {
      if (buffer.size === size) return true;
      this.discardActivityRead(buffer);
      return false;
    });
    const cached = this.activityReads.pop();
    if (cached) return cached;
    const buffer = this.device.createBuffer({
      label: 'Activity readback',
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.activityBuffers.add(buffer);
    return buffer;
  }
  private discardActivityRead(buffer: GPUBuffer): void {
    this.activityBuffers.delete(buffer);
    buffer.destroy();
  }
  private releaseActivityReads(): void {
    for (const buffer of this.activityBuffers) buffer.destroy();
    this.activityBuffers.clear();
    this.activityReads = [];
  }
  /** Note the tiles that held content, by the tile coordinates the step ran with, and
   * plan pool growth from the slots the step wanted. */
  private readActivity(read: GPUBuffer): void {
    const step = this.steps;
    const coordinates = this.directory.coordinates;
    const slots = this.poolSlots;
    const epoch = this.epoch;
    read
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        if (epoch !== this.epoch) {
          this.discardActivityRead(read);
          return;
        }
        const range = read.getMappedRange();
        const [bricks] = new Uint32Array(range, 0, 1);
        const [free, failures] = new Int32Array(range, 4, 2);
        const flags = new Uint32Array(range, READBACK_HEADER);
        const tiles: { tile: Vec3; bricks: number }[] = [];
        for (let slot = 0; slot < flags.length; slot++) {
          const coord = coordinates[slot];
          if (flags[slot] && coord) tiles.push({ tile: coord, bricks: flags[slot] });
        }
        read.unmap();
        if (
          this.activityReads.length < 8 &&
          read.size === READBACK_HEADER + this.tiles.capacity * 4
        )
          this.activityReads.push(read);
        else this.discardActivityRead(read);
        if (step < this.contentStep) return;
        this.contentStep = step;
        this.contentTiles = tiles;
        this.activeBrickCount = bricks;
        if (step < this.poolGrownAt) return;
        this.usedSlots = slots - free;
        const limit = this.slotLimit();
        this.poolLimited = failures > 0 && slots >= limit;
        // Grow with room to spare, so spreading content settles it after a few steps.
        if (failures > 0 && slots < limit)
          this.poolTarget = Math.max(
            this.poolTarget,
            Math.ceil((this.usedSlots + failures) * 1.5),
            slots + this.poolSide,
          );
      })
      .catch(() => this.discardActivityRead(read));
  }
  /** The bricks that computed at the latest step, in world space: those with a slot. For
   * the brick debug view. */
  async readBricks(): Promise<{ min: Vec3; max: Vec3 }[]> {
    const coordinates = this.directory.coordinates;
    const count = this.tiles.capacity * TILE_BRICK_COUNT;
    const size = this.brickCells * this.voxelSize;
    const read = this.device.createBuffer({
      label: 'Brick readback',
      size: count * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: 'Read bricks' });
    encoder.copyBufferToBuffer(this.tiles.pages, 0, read, 0, count * 4);
    this.device.queue.submit([encoder.finish()]);
    try {
      await read.mapAsync(GPUMapMode.READ);
      const pages = new Int32Array(read.getMappedRange().slice(0));
      read.unmap();
      const bricks: { min: Vec3; max: Vec3 }[] = [];
      for (let id = 0; id < count; id++) {
        const tile = coordinates[Math.floor(id / TILE_BRICK_COUNT)];
        if (!tile || pages[id] < 0) continue;
        const local = id % TILE_BRICK_COUNT;
        const min = [local & 3, (local >> 2) & 3, local >> 4].map(
          (n, axis) => (tile[axis] * TILE_BRICKS + n) * size,
        ) as Vec3;
        bricks.push({ min, max: min.map((value) => value + size) as Vec3 });
      }
      return bricks;
    } finally {
      read.destroy();
    }
  }
  dispose(): void {
    this.epoch++;
    this.releaseActivityReads();
    this.pressureSolver?.dispose();
    if (this.tiles) {
      this.releasePools();
      this.destroyTiles(this.tiles);
      this.tileTable.destroy();
    }
    this.noise?.texture.destroy();
    for (const buffer of [
      this.fieldUniform,
      this.smokeUniform,
      this.velocityUniform,
      this.growthUniform,
      this.flameUniform,
      this.flameRamps,
      ...this.scene.values(),
      this.dispatchArgs,
    ])
      buffer.destroy();
    this.pipelines.clear();
  }
}
