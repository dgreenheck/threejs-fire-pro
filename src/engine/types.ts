import type { FuelSettings } from './fuel.ts';
import type { FlameSettings } from './flame.ts';

export type Vec3 = [number, number, number];

/** Samples a shadow ray takes toward the sun, a fixed rendering budget. */
export const SHADOW_STEPS = 32;

export const LIGHTING_DIVISORS = [1, 2, 4] as const;
export type LightingDivisor = (typeof LIGHTING_DIVISORS)[number];

export interface RenderSettings {
  fireIntensity: number;
  flameOpacity: number;
  smokeDensity: number;
  scattering: number;
  shadowDensity: number;
  temperature: number;
  smokeColor: Vec3;
  fireColor: Vec3;
  sootGlow: number;
}

/** The simulation's lattice: cubic flame, heat, smoke and fuel cells of `voxelSize`
 * meters, anchored at the world origin. Smoke can use larger cells. */
export interface SimulationGrid {
  voxelSize: number;
  /** Velocity and pressure cells are `velocityDivisor` field cells wide. Defaults to 1. */
  velocityDivisor?: VelocityDivisor;
  /** Fine cells per brick side, independent of velocity resolution. Defaults to 16. */
  brickSize?: BrickSize;
  /** Smoke cells per fine-field cell along each axis. Defaults to 1. */
  smokeDivisor?: SmokeDivisor;
  /** Compile scalar transport with MacCormack correction. Defaults to true. */
  scalarMacCormack?: boolean;
}

export const VELOCITY_DIVISORS = [1, 2, 4] as const;
export type VelocityDivisor = (typeof VELOCITY_DIVISORS)[number];

export const SMOKE_DIVISORS = [1, 2, 4] as const;
export type SmokeDivisor = (typeof SMOKE_DIVISORS)[number];

export const FIXED_DT = 1 / 60;

/** Fine cells per side of a sparse allocation brick. */
export const BRICK_SIZES = [8, 16, 32] as const;
export type BrickSize = (typeof BRICK_SIZES)[number];

/** The device limits that size the brick pools. */
export type PoolLimits = Pick<
  GPUSupportedLimits,
  'maxTextureDimension3D' | 'maxStorageBufferBindingSize' | 'maxBufferSize'
>;
/** WebGPU's default limits, which Three.js requests unless told otherwise. */
export const DEFAULT_POOL_LIMITS: PoolLimits = {
  maxTextureDimension3D: 2048,
  maxStorageBufferBindingSize: 134_217_728,
  maxBufferSize: 268_435_456,
};

/** Most field voxels the pools may hold: every slot's storage buffers and texture
 * atlas fit the device's limits. The largest buffers are pressure (velocity cells
 * in half precision) and solid cells (one bit per field cell, solids.wgsl). */
export function maxPoolVoxels(
  velocityDivisor: VelocityDivisor,
  limits: PoolLimits = DEFAULT_POOL_LIMITS,
  brickSize: BrickSize = 16,
): number {
  const cells = brickSize ** 3;
  const bytes = Math.max((brickSize / velocityDivisor) ** 3 * 2, cells / 8);
  const perAxis = poolSlotsPerAxis(brickSize, limits);
  const side =
    velocityDivisor === 1
      ? Math.min(perAxis, Math.floor(limits.maxTextureDimension3D / (brickSize * 2)))
      : perAxis;
  return (
    Math.min(
      Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / bytes),
      side * side * perAxis,
    ) * cells
  );
}

/** Slots of brick pools along one axis of the largest 3D texture: a field slot is a brick
 * plus a one-texel apron. */
export function poolSlotsPerAxis(
  brickSize: BrickSize,
  limits: PoolLimits = DEFAULT_POOL_LIMITS,
): number {
  return Math.floor(limits.maxTextureDimension3D / (brickSize + 1));
}

/**
 * Pools place slot `s` at (s % side, (s / side) % side, s / side²) slots, and grow by
 * rows of `side` slots within the first layer, then by whole layers. `side` is fixed for
 * the device, so slots keep their texels as the pools grow: it is the smallest that fits
 * the most voxels the device allows within the 3D texture limit.
 */
export function poolSide(
  velocityDivisor: VelocityDivisor,
  limits: PoolLimits = DEFAULT_POOL_LIMITS,
  brickSize: BrickSize = 16,
): number {
  const perAxis = poolSlotsPerAxis(brickSize, limits);
  const slots = Math.floor(
    maxPoolVoxels(velocityDivisor, limits, brickSize) / brickSize ** 3,
  );
  // Compact trace donors need two integer words per velocity cell. At full
  // resolution their slots are twice as wide, without the vector pools' apron.
  const donorSide = velocityDivisor === 1 ? Math.floor(limits.maxTextureDimension3D / (brickSize * 2)) : perAxis;
  return Math.min(perAxis, donorSide, Math.max(4, Math.ceil(Math.sqrt(slots / perAxis))));
}

/** Rows of `side` slots that hold `slots`: whole rows in the first layer, then whole layers. */
export function poolRows(slots: number, side: number): number {
  const rows = Math.max(1, Math.ceil(slots / side));
  return rows <= side ? rows : Math.ceil(rows / side) * side;
}

/** Texels per axis of a pool of `rows` rows of slots `edge` texels on a side. */
export function poolSize(rows: number, side: number, edge: number): Vec3 {
  return [side * edge, Math.min(rows, side) * edge, Math.ceil(rows / side) * edge];
}

/** Slots the pools start with: whole rows holding at least 128, within `limit`. */
export function initialPoolSlots(side: number, limit: number): number {
  return Math.min(limit, side * Math.ceil(128 / side));
}

/** Pressure, its divergence and the multigrid (pressure.wgsl): per pool slot, three
 * half-precision values and a flag byte per velocity cell, coarsened to 2³ per brick;
 * per tile slot, the same for the three tile levels (4³, 2³ and one). */
function pressureMemory(slots: number, tiles: number, velocityCells = 8): number {
  const level = (cells: number) => Math.ceil(cells / 2) * 4 * 3 + Math.ceil(cells / 4) * 4;
  return (
    Array.from({ length: Math.log2(velocityCells) }, (_, i) => (velocityCells / 2 ** i) ** 3)
      .reduce((sum, cells) => sum + level(slots * cells), 0) +
    [64, 8, 1].reduce((sum, cells) => sum + level(tiles * cells), 0)
  );
}

/** Memory of the tile directory with `tiles` tile slots (tiles.wgsl): per tile its
 * record and content flag, and per brick its entry, record, page, state, content,
 * list entries and slot ranks; the hash, at most half full, and dispatch arguments. */
function tileMemory(tiles: number): number {
  let hash = 256;
  while (hash < tiles * 2) hash *= 2;
  return tiles * (128 + 4 + 64 * (4 + 128 + 4 + 4 + 4 + 4 + 16) + 2) + hash * 16 + 164;
}

/** Memory of brick pools with `slots` slots: rows of slots as they are allocated. Three
 * RGBA16F field volumes, one R16F expansion volume and packed RGBA32Uint donor bounds
 * shared between scalar and velocity transport; three RGBA16F velocity volumes and two
 * scratch volumes on a coarser velocity grid (a full-resolution one reuses the field
 * scratch); the free list, and each slot's solid cells. */
function poolMemory(
  slots: number,
  velocityDivisor: VelocityDivisor = 1,
  limits: PoolLimits = DEFAULT_POOL_LIMITS,
  scalarMacCormack = true,
  brickSize: BrickSize = 16,
): number {
  const side = poolSide(velocityDivisor, limits, brickSize);
  const rows = poolRows(slots, side);
  const texels = (edge: number) =>
    poolSize(rows, side, edge).reduce((count, value) => count * value, 1);
  return (
    texels(brickSize + 1) * 26 +
    texels(brickSize / velocityDivisor + 1) * (velocityDivisor === 1 ? 24 : 40) +
    rows *
      side *
      Math.max(brickSize ** 3, 2 * (brickSize / velocityDivisor) ** 3) * 4 *
      (scalarMacCormack ? 4 : 1) +
    Math.min(rows * side, slots) * (4 + brickSize ** 3 / 8) +
    16
  );
}

/** Three R16F smoke pools in coarser modes, plus their grid parameters. */
function smokeMemory(
  slots: number,
  velocityDivisor: VelocityDivisor,
  smokeDivisor: SmokeDivisor,
  brickSize: BrickSize = 16,
): number {
  if (smokeDivisor === 1) return 192;
  const side = poolSide(velocityDivisor, DEFAULT_POOL_LIMITS, brickSize);
  const rows = poolRows(slots, side);
  return (
    poolSize(rows, side, brickSize / smokeDivisor + 1).reduce(
      (n, v) => n * v,
      6,
    ) + 192
  );
}

/** Simulation memory with pools of `slots` slots and `tiles` tile slots, plus the
 * curl-noise texture. */
export function memoryEstimate(
  slots: number,
  velocityDivisor: VelocityDivisor = 1,
  tiles = 16,
  smokeDivisor: SmokeDivisor = 1,
  scalarMacCormack = true,
  brickSize: BrickSize = 16,
): number {
  return (
    poolMemory(slots, velocityDivisor, DEFAULT_POOL_LIMITS, scalarMacCormack, brickSize) +
    smokeMemory(slots, velocityDivisor, smokeDivisor, brickSize) +
    pressureMemory(slots, tiles, brickSize / velocityDivisor) +
    tileMemory(tiles) +
    64 ** 3 * 8 +
    // Field/velocity/smoke/pool growth/flame uniforms and the flame response ramps.
    640
  );
}

/** Renderer memory with pools of `slots` slots: each slot's incident light, at the chosen
 * resolution with an apron (RGBA16F), and the flags of its 8³-cell blocks (RG16F); the
 * page tables of `bricks` bricks in boxes, in rows of 4096; the box table, the blackbody
 * table and the parameters. Smoothed fields borrow the simulation's scalar predictor.
 * Excludes screen and Three.js buffers. */
export function renderMemoryEstimate(
  slots: number,
  brickSize: BrickSize = 16,
  bricks = 0,
  lightingDivisor: LightingDivisor = 4,
): number {
  const cells = brickSize;
  const slot = (cells / lightingDivisor + 1) ** 3 * 8 + (cells / 8) ** 3 * 4;
  return slots * slot + Math.max(1, Math.ceil(bricks / 4096)) * 4096 * 4 + 512 * 16 + 512 * 8 + 80;
}

export interface SimulationSettings {
  buoyancy: number;
  smokeWeight: number;
  velocityDamping: number;
  cooling: number;
  dissipation: number;
  vorticity: number;
  seed: number;
}

/** Positions are in world space. */
export interface SimulationSource {
  id: string;
  shape: 'sphere' | 'box' | 'mesh';
  position: Vec3;
  /** Sphere radii or box half-extents. */
  size: Vec3;
  /** Normalized remaining-lifetime target. */
  flame: number;
  /** Direct heat and smoke rates per second. */
  heatRate: number;
  smokeRate: number;
  fuelRate?: number;
  velocity: Vec3;
  velocityResponse: number;
  /** Radial target speed, evaluated by the original emitter coupling. */
  outwardSpeed: number;
  /** Burst-only initial scalar variation. Continuous sources omit this. */
  variation?: { period: number; strength: number };
  /** Immutable transformed surface samples in world space. Replace the array when the
   * mesh moves. */
  samples?: readonly { position: Vec3; weight: number }[];
  surfaceArea?: number;
}

/** A collider in world space: a sphere or a box, or a mesh's occupancy. */
export type SimulationCollider =
  | {
      shape: 'sphere' | 'box';
      position: Vec3;
      /** Sphere radii or box half-extents along its axes. */
      size: Vec3;
      /** The box's axes in world space, unit vectors: the world axes by default. Spheres
       * ignore them. */
      axes?: readonly [Vec3, Vec3, Vec3];
    }
  | { shape: 'mesh'; occupancy: MeshColliderMask };

/** Acceleration field, optionally masked by the union of emitter spatial footprints. */
export interface SimulationForce {
  type: 'wind' | 'turbulence' | 'vortex' | 'radial';
  vector: Vec3;
  center: Vec3;
  strength: number;
  scale: number;
  lift: number;
  inward: number;
  sources?: readonly SimulationSource[];
}

export interface SimulationFrame {
  simulation: SimulationSettings;
  combustion: FlameSettings;
  fuel?: FuelSettings;
  sources: readonly SimulationSource[];
  forces?: readonly SimulationForce[];
  colliders: readonly SimulationCollider[];
}

/** Cell-center occupancy of a box of lattice cells: cell `lo + (x, y, z)` holds
 * `mask[(z * size.y + y) * size.x + x]`. The lattice's cell 0 starts at the world origin.
 * Immutable: replace it when the geometry moves. */
export interface MeshColliderMask {
  mask: Uint32Array<ArrayBuffer>;
  lo: Vec3;
  size: Vec3;
}
