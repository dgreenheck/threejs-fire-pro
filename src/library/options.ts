import { Color, Mesh, Object3D, type ColorRepresentation } from 'three';
import { DEFAULT_FUEL, type FuelSettings } from '../engine/fuel.ts';
import {
  LIGHTING_DIVISORS,
  type LightingDivisor,
  BRICK_SIZES,
  type BrickSize,
  VELOCITY_DIVISORS,
  SMOKE_DIVISORS,
  type SmokeDivisor,
  type VelocityDivisor,
} from '../engine/types.ts';

export type DeepReadonly<T> = {
  readonly [Key in keyof T]: T[Key] extends object ? DeepReadonly<T[Key]> : T[Key];
};
export type ColorValue = Color | number | `#${string}`;
export type Vec3 = [number, number, number];
export type ReconstructionFilter = 'trilinear' | 'quadratic' | 'cubic';

/** Flame lifetime response: how long flames live, what they produce and how they look. */
export interface FlameOptions {
  /** Seconds for a flame's normalized remaining lifetime to decay from one to zero. */
  lifespan?: number;
  /** Peak relative heat production per second, weighted by the flame's remaining lifetime. */
  heatRate?: number;
  /** Peak relative smoke production per second, weighted by the flame's remaining lifetime. */
  smokeRate?: number;
  /** Gas expansion produced over the flame lifetime, in 1/s; also the expansion for each
   * unit of fuel burned a second. */
  expansionRate?: number;
  /** Exponential heat decay rate in 1/s. */
  cooling?: number;
  color?: ColorValue;
  brightness?: number;
  /** Flame absorption per meter at full glow; also scales its emission. Defaults to 0.55. */
  opacity?: number;
  /** Maximum blackbody temperature, in kelvin, used for flame color. */
  temperature?: number;
  /** Thermal glow from hot smoke. */
  sootGlow?: number;
}
export interface SmokeOptions {
  /** Exponential smoke decay rate in 1/s. */
  dissipation?: number;
  color?: ColorValue;
  /** Light extinction per unit of smoke. */
  density?: number;
  /** Henyey–Greenstein anisotropy from -0.9 to 0.9. */
  scattering?: number;
  /** Extinction multiplier for self-shadowing from directional light. */
  shadowDensity?: number;
}
export interface MotionOptions {
  /** Upward acceleration per unit heat. */
  buoyancy?: number;
  /** Downward acceleration per unit smoke. */
  smokeWeight?: number;
  /** Exponential velocity decay rate in 1/s. */
  damping?: number;
  /** Vorticity-confinement strength; adds small swirling detail. */
  vorticity?: number;
}
/** Optional fuel. Emitters and explosions supply fuel that the flow carries; where it is
 * hot enough it burns as flame, its amount up to 1 the flame's lifetime, and the flame
 * settings give its heat, smoke and expansion. The gas also expands as the fuel burns, by
 * the flame's expansion rate for each unit burned a second. */
export interface FuelOptions {
  enabled?: boolean;
  /** Heat required before fuel burns. */
  ignitionHeat?: number;
  /** First-order fuel consumption in 1/s. */
  burnRate?: number;
}
export interface LightingOptions {
  /** Light nearby scene surfaces from the flames: a point light for each emitter and each
   * burning explosion, at the flames nearest it. With Three.js ClusteredLighting every one
   * is a light of its own; otherwise four lights go to those brightest at the camera. */
  illuminateScene?: boolean;
  intensity?: number;
}
export interface RenderingOptions {
  /** Incident-light cells per fine field cell along each axis: 1, 2 or 4. Defaults to 4. */
  lightingDivisor?: LightingDivisor;
  /** Field reconstruction; trilinear is fastest, cubic is smoothest. */
  filter?: ReconstructionFilter;
  /** Most volume samples a ray takes. Rays take two a voxel, fewer when they are long. */
  raySteps?: number;
  /** March every other pixel on each axis, then finish each pixel's clipped end at
   * full resolution. About a quarter of the cost, with slightly softer detail. */
  halfResolution?: boolean;
}
/** Where the simulation stores and computes cells. It holds only the flame, smoke and
 * fuel above the cutoff, plus the sources, and grows and shrinks with them. Heat alone is
 * invisible and does not hold it. */
export interface GridOptions {
  /** Most cells the simulation may store and compute. When the content needs more, the
   * cells past the budget stay empty. */
  maxVoxels?: number;
  /** Smoke and fuel density, and remaining flame lifetime from 0 to 1, below which
   * they no longer hold cells open. */
  cutoff?: number;
  /** A solid ground below world y = 0. */
  ground?: boolean;
}
/** Settings that can change at any time through `simulation.configure()`. */
export interface SimulationConfigureOptions {
  grid?: GridOptions;
  flame?: FlameOptions;
  smoke?: SmokeOptions;
  motion?: MotionOptions;
  fuel?: FuelOptions;
  lighting?: LightingOptions;
  rendering?: RenderingOptions;
}
/**
 * The simulation runs in world space and has no fixed extent: it holds the content
 * wherever it is, in cells of `voxelSize`.
 */
export interface SimulationOptions extends SimulationConfigureOptions {
  /** Edge length of one flame, heat and fuel cell, in meters. */
  voxelSize?: number;
  /** Velocity and pressure cells are `velocityDivisor` times larger: 1, 2 or 4. */
  velocityDivisor?: VelocityDivisor;
  /** Fine cells per brick side: 8, 16 or 32. Independent of velocity resolution. Defaults to 16. */
  brickSize?: BrickSize;
  /** Smoke simulation/transport cells are 1, 2 or 4 times the voxel size. Defaults to 1. */
  smokeDivisor?: SmokeDivisor;
  /** Bounded MacCormack transport for scalar fields and smoke. Defaults to true. */
  scalarMacCormack?: boolean;
  /** Integer seed for repeatable variation. */
  seed?: number;
}

export interface EmissionOptions {
  /** Normalized flame lifetime supplied by the source, from 0 to 1. */
  flame?: number;
  heatRate?: number;
  smokeRate?: number;
  /** Fuel added per second; requires `fuel.enabled` on the simulation. */
  fuelRate?: number;
}
export interface VelocityOptions {
  direction?: Vec3;
  speed?: number;
}
export type EmitterShape = { type: 'sphere'; radius?: number } | { type: 'mesh'; object: Mesh };
export interface EmitterOptions {
  shape?: EmitterShape;
  active?: boolean;
  emission?: EmissionOptions;
  velocity?: VelocityOptions | null;
}
export interface EmitterPatch extends Omit<EmitterOptions, 'shape'> {
  /** Sphere emitters can change radius; the shape type is construction-only. */
  shape?: { radius?: number };
}
export interface ExplosionOptions {
  radius?: number;
  charge?: { flame?: number; heat?: number; smoke?: number; fuel?: number };
  outwardSpeed?: number;
  /** Initial heat/lifetime variation, sampled once when the burst is injected. */
  variation?: {
    /** Base noise spacing in meters; larger values make broader patches. */
    period?: number;
    /** Maximum fractional reduction of heat/lifetime, from 0 (off) to 1. */
    strength?: number;
  };
}
export type ForceOptions =
  | {
      type: 'wind';
      direction: Vec3;
      /** Acceleration in m/s². */
      strength: number;
      active?: boolean;
    }
  | {
      type: 'turbulence';
      strength: number;
      /** Noise spatial frequency in 1/m. */
      scale: number;
      active?: boolean;
    }
  | {
      type: 'vortex';
      /** Center in world coordinates. */
      center?: Vec3;
      axis?: Vec3;
      /** Tangential acceleration; negative reverses rotation. */
      strength: number;
      radius?: number;
      lift?: number;
      inward?: number;
      active?: boolean;
    }
  | {
      type: 'radial';
      center?: Vec3;
      /** Positive pushes outward; negative pulls inward. */
      strength: number;
      radius?: number;
      active?: boolean;
    };
export type ForcePatch =
  | Partial<Omit<Extract<ForceOptions, { type: 'wind' }>, 'type'>>
  | Partial<Omit<Extract<ForceOptions, { type: 'turbulence' }>, 'type'>>
  | Partial<Omit<Extract<ForceOptions, { type: 'vortex' }>, 'type'>>
  | Partial<Omit<Extract<ForceOptions, { type: 'radial' }>, 'type'>>;
export type ColliderShape =
  { type: 'sphere'; radius: number } | { type: 'box'; size: Vec3 } | { type: 'mesh' };
/** `object` supplies the collider's transform; mesh colliders also use its geometry. */
export type ColliderOptions =
  | { object: Object3D; shape: Exclude<ColliderShape, { type: 'mesh' }> }
  | { object: Mesh; shape: { type: 'mesh' } };
/** What the renderer shows: the final image, or one field. Vorticity is computed from the
 * velocity; expansion is the flames' target divergence; pressure and divergence are the
 * solver's, copied for the view after every step. */
export const DEBUG_FIELDS = [
  'beauty',
  'lifetime',
  'heat',
  'smoke',
  'velocity',
  'flame',
  'fuel',
  'vorticity',
  'expansion',
  'pressure',
  'divergence',
] as const;
export type DebugField = (typeof DEBUG_FIELDS)[number];
/** Each debug view's heat map: it spans values from 0 to `range`, or from -range to range
 * when `signed`, in `unit`. Velocity and vorticity show their magnitudes. */
export const DEBUG_SCALES: Readonly<
  Record<Exclude<DebugField, 'beauty'>, { range: number; signed: boolean; unit: string }>
> = {
  lifetime: { range: 1, signed: false, unit: '' },
  heat: { range: 4, signed: false, unit: '' },
  smoke: { range: 1, signed: false, unit: '' },
  velocity: { range: 4, signed: false, unit: 'm/s' },
  flame: { range: 1, signed: false, unit: '' },
  fuel: { range: 2, signed: false, unit: '' },
  vorticity: { range: 50, signed: false, unit: '1/s' },
  expansion: { range: 5, signed: false, unit: '1/s' },
  pressure: { range: 0.5, signed: true, unit: '' },
  divergence: { range: 10, signed: true, unit: '1/s' },
};
export interface DebugOptions {
  /** Outline the bricks the solver computes. */
  bricks?: boolean;
  field?: DebugField;
}

type ResolvedColor<T> = Omit<Required<T>, 'color'> & { color: string };
export interface ResolvedSimulationOptions {
  voxelSize: number;
  velocityDivisor: VelocityDivisor;
  brickSize: BrickSize;
  smokeDivisor: SmokeDivisor;
  scalarMacCormack: boolean;
  seed: number;
  grid: Required<GridOptions>;
  flame: ResolvedColor<FlameOptions>;
  smoke: ResolvedColor<SmokeOptions>;
  motion: Required<MotionOptions>;
  fuel: FuelSettings;
  lighting: Required<LightingOptions>;
  rendering: Required<RenderingOptions>;
}
export interface ResolvedEmitterOptions {
  shape: { type: 'sphere'; radius: number } | { type: 'mesh' };
  active: boolean;
  emission: Required<EmissionOptions>;
  velocity: Required<VelocityOptions> | null;
}
export interface ResolvedExplosionOptions {
  radius: number;
  charge: { flame: number; heat: number; smoke: number; fuel: number };
  outwardSpeed: number;
  variation: { period: number; strength: number };
}
export type ResolvedForceOptions =
  | Required<Extract<ForceOptions, { type: 'wind' }>>
  | Required<Extract<ForceOptions, { type: 'turbulence' }>>
  | Required<Extract<ForceOptions, { type: 'vortex' }>>
  | Required<Extract<ForceOptions, { type: 'radial' }>>;

/** Invalid patches reject before any state is changed. Unknown keys are errors. */
export function keys(value: unknown, allowed: readonly string[], label: string): void {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${label} must be an options object.`);
  for (const key of Object.keys(value))
    if (!allowed.includes(key)) throw new Error(`Unknown ${label} option: ${key}.`);
}
export function number(value: unknown, label: string, min = 0, max = 1000): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${label} must be a finite number from ${min} to ${max}.`);
  }
  return value;
}
export function boolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean.`);
  return value;
}
export function vector(value: unknown, label: string, min = -1e6, max = 1e6): Vec3 {
  if (!Array.isArray(value) || value.length !== 3)
    throw new Error(`${label} must contain three numbers.`);
  return value.map((item, i) => number(item, `${label}[${i}]`, min, max)) as Vec3;
}
export function direction(value: unknown, label: string): Vec3 {
  const result = vector(value, label);
  const magnitude = Math.hypot(...result);
  if (magnitude === 0) throw new Error(`${label} must be nonzero.`);
  // Preserve already-normalized vectors across JSON exports and repeated patches.
  const divisor = Math.abs(magnitude - 1) < 1e-12 ? 1 : magnitude;
  return result.map((component) => (component === 0 ? 0 : component / divisor)) as Vec3;
}
function color(value: unknown, label: string): string {
  const validString = typeof value === 'string' && /^#[\da-f]{6}$/i.test(value);
  const validNumber =
    typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 0xffffff;
  if (!validString && !validNumber && !(value instanceof Color)) {
    throw new Error(`${label} must be a #RRGGBB string, RGB integer, or THREE.Color.`);
  }
  const result = new Color(value as ColorRepresentation);
  if (
    ![result.r, result.g, result.b].every(
      (component) => Number.isFinite(component) && component >= 0 && component <= 1,
    )
  ) {
    throw new Error(`${label} must have finite RGB channels between zero and one.`);
  }
  return `#${result.getHexString()}`;
}

type ConstructionKey =
  'voxelSize' | 'brickSize' | 'velocityDivisor' | 'smokeDivisor' | 'scalarMacCormack' | 'seed';
type ResolvedSettings = Omit<ResolvedSimulationOptions, ConstructionKey>;
const DEFAULTS: ResolvedSettings = {
  grid: { maxVoxels: 8_000_000, cutoff: 0.05, ground: true },
  flame: {
    lifespan: 1.2,
    heatRate: 4,
    smokeRate: 2,
    expansionRate: 1,
    cooling: 0.62,
    color: '#ffffff',
    brightness: 1.4,
    opacity: 0.55,
    temperature: 3200,
    sootGlow: 1,
  },
  smoke: {
    dissipation: 0.35,
    color: '#25292e',
    density: 2.5,
    scattering: 0.38,
    shadowDensity: 3.5,
  },
  motion: { buoyancy: 1.6, smokeWeight: 0.12, damping: 0.1, vorticity: 3.2 },
  fuel: { ...DEFAULT_FUEL },
  lighting: { illuminateScene: true, intensity: 1 },
  rendering: { filter: 'cubic', raySteps: 256, halfResolution: true, lightingDivisor: 4 },
};
/** Valid [min, max] of every numeric option. Validation and editors both read it. */
export const LIMITS = {
  simulation: { voxelSize: [0.002, 4] },
  grid: { maxVoxels: [4096, 268_435_456], cutoff: [0, 1] },
  flame: {
    lifespan: [0.01, 5],
    heatRate: [0, 20],
    smokeRate: [0, 20],
    expansionRate: [0, 20],
    cooling: [0, 20],
    brightness: [0, 20],
    opacity: [0, 20],
    temperature: [800, 10000],
    sootGlow: [0, 5],
  },
  smoke: {
    dissipation: [0, 20],
    density: [0, 20],
    scattering: [-0.9, 0.9],
    shadowDensity: [0, 20],
  },
  motion: { buoyancy: [0, 20], smokeWeight: [0, 20], damping: [0, 20], vorticity: [0, 20] },
  fuel: { ignitionHeat: [0, 100], burnRate: [0, 100] },
  lighting: { intensity: [0, 20] },
  rendering: { raySteps: [8, 1024] },
  emitter: { radius: [0.01, 16], flame: [0, 1], rate: [0, 100], speed: [0, 50] },
  explosion: {
    radius: [0.01, 16],
    outwardSpeed: [0, 50],
    period: [0.01, 32],
    strength: [0, 1],
    flame: [0, 1],
    amount: [0, 100],
  },
  wind: { strength: [0, 50] },
  turbulence: { strength: [0, 50], scale: [0.01, 32] },
  vortex: { strength: [-50, 50], radius: [0.01, 32], lift: [-50, 50], inward: [0, 50] },
  radial: { strength: [-50, 50], radius: [0.01, 32] },
  collider: { radius: [0.01, 32], size: [0.01, 64] },
} as const;
const FILTERS: readonly ReconstructionFilter[] = ['trilinear', 'quadratic', 'cubic'];
export const GROUPS = [
  'grid',
  'flame',
  'smoke',
  'motion',
  'fuel',
  'lighting',
  'rendering',
] as const;

/** Merge a configurable patch into resolved settings, validating every supplied value. */
export function resolveSettings(
  patch: SimulationConfigureOptions,
  previous: ResolvedSettings = DEFAULTS,
): ResolvedSettings {
  const result = structuredClone(previous);
  for (const group of GROUPS) {
    const block = patch[group] as Record<string, unknown> | undefined;
    if (block === undefined) continue;
    const target = result[group] as unknown as Record<string, unknown>;
    keys(block, Object.keys(target), group);
    for (const [key, value] of Object.entries(block)) {
      if (value === undefined) continue;
      const label = `${group}.${key}`;
      const limits = (LIMITS[group] as Record<string, readonly [number, number]>)[key];
      if (limits) target[key] = number(value, label, ...limits);
      else if (key === 'color') target[key] = color(value, label);
      else if (key === 'lightingDivisor') {
        if (!LIGHTING_DIVISORS.includes(value as LightingDivisor))
          throw new Error(`${label} must be 1, 2 or 4.`);
        target[key] = value;
      } else if (key === 'filter') {
        if (!FILTERS.includes(value as ReconstructionFilter))
          throw new Error(`${label} must be ${FILTERS.join(', ')}.`);
        target[key] = value;
      } else target[key] = boolean(value, label);
    }
  }
  if (!Number.isInteger(result.rendering.raySteps))
    throw new Error('rendering.raySteps must be an integer.');
  if (!Number.isInteger(result.grid.maxVoxels))
    throw new Error('grid.maxVoxels must be an integer.');
  return result;
}
export function resolveSimulation(options: SimulationOptions = {}): ResolvedSimulationOptions {
  keys(
    options,
    ['voxelSize', 'brickSize', 'velocityDivisor', 'smokeDivisor', 'scalarMacCormack', 'seed', ...GROUPS],
    'simulation',
  );
  const seed = number(options.seed ?? 1, 'seed', 0, 0xffffffff);
  if (!Number.isInteger(seed)) throw new Error('seed must be an integer.');
  const voxelSize = number(options.voxelSize ?? 0.02, 'voxelSize', ...LIMITS.simulation.voxelSize);
  const brickSize = options.brickSize ?? 16;
  if (!BRICK_SIZES.includes(brickSize)) throw new Error('brickSize must be 8, 16 or 32.');
  const velocityDivisor = options.velocityDivisor ?? 2;
  if (!VELOCITY_DIVISORS.includes(velocityDivisor))
    throw new Error('velocityDivisor must be 1, 2 or 4.');
  const smokeDivisor = options.smokeDivisor ?? 1;
  if (!SMOKE_DIVISORS.includes(smokeDivisor)) throw new Error('smokeDivisor must be 1, 2 or 4.');
  return {
    voxelSize,
    brickSize,
    velocityDivisor,
    smokeDivisor,
    scalarMacCormack: boolean(options.scalarMacCormack ?? true, 'scalarMacCormack'),
    seed,
    ...resolveSettings(options),
  };
}
export type EmitterKind = 'fire' | 'jet' | 'smoke' | 'emitter';
export function emitterDefaults(kind: EmitterKind): ResolvedEmitterOptions {
  const defaults = {
    fire: { radius: 0.38, flame: 1, heatRate: 1.5, smokeRate: 0, speed: 1.4 },
    jet: { radius: 0.22, flame: 1, heatRate: 2.5, smokeRate: 0, speed: 10 },
    smoke: { radius: 0.4, flame: 0, heatRate: 0.85, smokeRate: 4, speed: 1.2 },
    emitter: { radius: 0.3, flame: 0, heatRate: 0, smokeRate: 0, speed: 0 },
  }[kind];
  return {
    shape: { type: 'sphere', radius: defaults.radius },
    active: true,
    emission: {
      flame: defaults.flame,
      heatRate: defaults.heatRate,
      smokeRate: defaults.smokeRate,
      fuelRate: 0,
    },
    velocity: kind === 'emitter' ? null : { direction: [0, 1, 0], speed: defaults.speed },
  };
}
export function resolveEmitter(
  patch: EmitterPatch | EmitterOptions,
  previous: ResolvedEmitterOptions,
  constructing = false,
): ResolvedEmitterOptions {
  keys(patch, ['shape', 'active', 'emission', 'velocity'], 'emitter');
  const result = structuredClone(previous);
  if (patch.active !== undefined) result.active = boolean(patch.active, 'active');
  if (patch.shape !== undefined) {
    keys(patch.shape, constructing ? ['type', 'radius', 'object'] : ['radius'], 'shape');
    if (constructing) {
      if (!('type' in patch.shape) || !['sphere', 'mesh'].includes(patch.shape.type))
        throw new Error('shape.type must be sphere or mesh.');
      if (patch.shape.type === 'mesh') {
        if (!(patch.shape.object instanceof Mesh))
          throw new Error('shape.object must be a THREE.Mesh.');
        if ('radius' in patch.shape) throw new Error('Mesh source does not accept radius.');
        result.shape = { type: 'mesh' };
      } else {
        if ('object' in patch.shape) throw new Error('Sphere source does not accept object.');
        result.shape = {
          type: 'sphere',
          radius: number(patch.shape.radius ?? 0.3, 'shape.radius', ...LIMITS.emitter.radius),
        };
      }
    } else if ('radius' in patch.shape && patch.shape.radius !== undefined) {
      if (result.shape.type !== 'sphere')
        throw new Error('Mesh emitter shape cannot be reconfigured.');
      result.shape.radius = number(patch.shape.radius, 'shape.radius', ...LIMITS.emitter.radius);
    }
  }
  if (patch.emission !== undefined) {
    keys(patch.emission, ['flame', 'heatRate', 'smokeRate', 'fuelRate'], 'emission');
    for (const name of ['flame', 'heatRate', 'smokeRate', 'fuelRate'] as const) {
      if (patch.emission[name] !== undefined)
        result.emission[name] = number(
          patch.emission[name],
          `emission.${name}`,
          ...(name === 'flame' ? LIMITS.emitter.flame : LIMITS.emitter.rate),
        );
    }
  }
  if (patch.velocity === null) result.velocity = null;
  else if (patch.velocity !== undefined) {
    keys(patch.velocity, ['direction', 'speed'], 'velocity');
    result.velocity = {
      direction:
        patch.velocity.direction !== undefined
          ? direction(patch.velocity.direction, 'velocity.direction')
          : (result.velocity?.direction ?? [0, 1, 0]),
      speed: number(
        patch.velocity.speed ?? result.velocity?.speed ?? 0,
        'velocity.speed',
        ...LIMITS.emitter.speed,
      ),
    };
  }
  return result;
}
export function resolveExplosion(
  patch: ExplosionOptions = {},
  previous: ResolvedExplosionOptions = {
    radius: 0.5,
    charge: { flame: 1, heat: 2, smoke: 0, fuel: 0 },
    outwardSpeed: 8,
    variation: { period: 0.5, strength: 0 },
  },
): ResolvedExplosionOptions {
  keys(patch, ['radius', 'charge', 'outwardSpeed', 'variation'], 'explosion');
  const result = structuredClone(previous);
  if (patch.radius !== undefined)
    result.radius = number(patch.radius, 'radius', ...LIMITS.explosion.radius);
  if (patch.outwardSpeed !== undefined)
    result.outwardSpeed = number(
      patch.outwardSpeed,
      'outwardSpeed',
      ...LIMITS.explosion.outwardSpeed,
    );
  if (patch.variation !== undefined) {
    keys(patch.variation, ['period', 'strength'], 'explosion.variation');
    if (patch.variation.period !== undefined)
      result.variation.period = number(
        patch.variation.period,
        'variation.period',
        ...LIMITS.explosion.period,
      );
    if (patch.variation.strength !== undefined)
      result.variation.strength = number(
        patch.variation.strength,
        'variation.strength',
        ...LIMITS.explosion.strength,
      );
  }
  if (patch.charge !== undefined) {
    keys(patch.charge, ['flame', 'heat', 'smoke', 'fuel'], 'charge');
    for (const name of ['flame', 'heat', 'smoke', 'fuel'] as const)
      if (patch.charge[name] !== undefined)
        result.charge[name] = number(
          patch.charge[name],
          `charge.${name}`,
          ...(name === 'flame' ? LIMITS.explosion.flame : LIMITS.explosion.amount),
        );
  }
  return result;
}
export function resolveForce(
  patch: ForceOptions | ForcePatch,
  previous?: ResolvedForceOptions,
): ResolvedForceOptions {
  const type = previous?.type ?? ('type' in patch ? patch.type : undefined);
  const allowed =
    type === 'wind'
      ? ['direction', 'strength', 'active']
      : type === 'turbulence'
        ? ['strength', 'scale', 'active']
        : type === 'vortex'
          ? ['center', 'axis', 'strength', 'radius', 'lift', 'inward', 'active']
          : ['center', 'strength', 'radius', 'active'];
  keys(patch, previous ? allowed : ['type', ...allowed], 'force');
  const p = patch as Record<string, unknown>;
  const active = boolean(p.active ?? previous?.active ?? true, 'force.active');
  if (type === 'wind') {
    const old = previous?.type === 'wind' ? previous : undefined;
    return {
      type,
      active,
      direction: direction(p.direction ?? old?.direction, 'wind.direction'),
      strength: number(p.strength ?? old?.strength, 'wind.strength', ...LIMITS.wind.strength),
    };
  }
  if (type === 'turbulence') {
    const old = previous?.type === 'turbulence' ? previous : undefined;
    return {
      type,
      active,
      strength: number(
        p.strength ?? old?.strength,
        'turbulence.strength',
        ...LIMITS.turbulence.strength,
      ),
      scale: number(p.scale ?? old?.scale, 'turbulence.scale', ...LIMITS.turbulence.scale),
    };
  }
  if (type === 'vortex' || type === 'radial') {
    const old = previous?.type === type ? previous : undefined;
    const common = {
      active,
      center: vector(p.center ?? (old && 'center' in old ? old.center : [0, 0, 0]), 'force.center'),
      strength: number(p.strength ?? old?.strength, 'force.strength', ...LIMITS[type].strength),
      radius: number(
        p.radius ?? (old && 'radius' in old ? old.radius : 1),
        'force.radius',
        ...LIMITS[type].radius,
      ),
    };
    if (type === 'radial') return { type, ...common };
    const vortex = previous?.type === 'vortex' ? previous : undefined;
    return {
      type,
      ...common,
      axis: direction(p.axis ?? vortex?.axis ?? [0, 1, 0], 'vortex.axis'),
      lift: number(p.lift ?? vortex?.lift ?? 0, 'vortex.lift', ...LIMITS.vortex.lift),
      inward: number(p.inward ?? vortex?.inward ?? 0, 'vortex.inward', ...LIMITS.vortex.inward),
    };
  }
  throw new Error('force.type must be wind, turbulence, vortex or radial.');
}
export function resolveCollider(options: ColliderOptions): ColliderOptions {
  keys(options, ['object', 'shape'], 'collider');
  if (!(options.object instanceof Object3D))
    throw new Error('collider.object must be a THREE.Object3D.');
  const shape = options.shape as Record<string, unknown>;
  if (!shape || typeof shape !== 'object' || Array.isArray(shape))
    throw new Error('collider.shape must be an options object.');
  if (shape.type === 'sphere') {
    keys(shape, ['type', 'radius'], 'collider.shape');
    return {
      object: options.object,
      shape: {
        type: 'sphere',
        radius: number(shape.radius, 'collider.shape.radius', ...LIMITS.collider.radius),
      },
    };
  }
  if (shape.type === 'box') {
    keys(shape, ['type', 'size'], 'collider.shape');
    return {
      object: options.object,
      shape: {
        type: 'box',
        size: vector(shape.size, 'collider.shape.size', ...LIMITS.collider.size),
      },
    };
  }
  if (shape.type === 'mesh') {
    keys(shape, ['type'], 'collider.shape');
    if (!(options.object instanceof Mesh))
      throw new Error('Mesh collider object must be a THREE.Mesh.');
    return { object: options.object, shape: { type: 'mesh' } };
  }
  throw new Error('collider.shape.type must be sphere, box or mesh.');
}
