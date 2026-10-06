import {
  resolveSimulation,
  resolveForce,
  resolveExplosion,
  emitterDefaults,
  resolveEmitter,
  LIMITS,
  type ResolvedSimulationOptions,
  type ForceOptions,
  type EmitterOptions,
  type ExplosionOptions,
  type Vec3,
} from '../src/library/options.ts';

export const RECIPES = [
  'empty',
  'emitter-forces',
  'flaming-letters',
  'fireplace',
  'campfire',
  'projectiles',
  'flamethrower',
  'fire-tornado',
  'magical-fire',
  'delayed-ignition',
  'industrial-smoke',
  'fuel-fireball',
  'moving-emitter',
] as const;
export type Recipe = (typeof RECIPES)[number];
export interface Transform {
  position: Vec3;
  rotation: Vec3;
}
export interface SourceDocument extends Transform {
  id: string;
  name: string;
  mode: 'continuous' | 'burst';
  shape: 'sphere' | 'box' | 'disk' | 'torus' | 'letters';
  radius: number;
  size: Vec3;
  options: Omit<EmitterOptions, 'shape'>;
  burst: ExplosionOptions;
}
export interface ForceDocument {
  id: string;
  name: string;
  options: ForceOptions;
  targets: null | string[];
}
export interface ColliderDocument extends Transform {
  id: string;
  name: string;
  shape: 'sphere' | 'box';
  size: Vec3;
  radius: number;
}
/** A preset: the preview scene, and one simulation in world space with its emitters,
 * forces and colliders. */
export interface SimulationDocument {
  format: 'fire-pro';
  version: 11;
  id: string;
  name: string;
  scene: {
    recipe: Recipe;
    sky: boolean;
    floor: { roughness: number; metalness: number };
  };
  camera: { position: Vec3; target: Vec3 };
  simulation: ResolvedSimulationOptions;
  emitters: SourceDocument[];
  forces: ForceDocument[];
  colliders: ColliderDocument[];
}
/** Selection ID of the simulation's own settings; every other object has a generated ID. */
export const SIMULATION_ID = 'simulation';
export const uid = () => crypto.randomUUID();
export const transform = (): Transform => ({ position: [0, 0, 0], rotation: [0, 0, 0] });
export function newSource(mode: SourceDocument['mode'] = 'continuous'): SourceDocument {
  return {
    id: uid(),
    name: 'Emitter',
    mode,
    shape: 'sphere',
    radius: 0.35,
    size: [1, 1, 1],
    ...transform(),
    position: [0, 0.5, 0],
    options: {
      active: true,
      emission: { flame: 1, heatRate: 1, smokeRate: 0.02, fuelRate: 0 },
      velocity: null,
    },
    burst: {
      radius: 0.5,
      charge: { flame: 1, heat: 2, smoke: 0.5, fuel: 0 },
      outwardSpeed: 2,
      variation: { period: 0.4, strength: 0.3 },
    },
  };
}
export function newForce(type: ForceOptions['type'] = 'turbulence'): ForceDocument {
  const options: ForceOptions =
    type === 'wind'
      ? { type, direction: [1, 0, 0], strength: 1 }
      : type === 'turbulence'
        ? { type, strength: 2, scale: 1 }
        : type === 'vortex'
          ? {
              type,
              strength: 3,
              center: [0, 1, 0],
              axis: [0, 1, 0],
              radius: 1,
              lift: 1,
              inward: 0.2,
            }
          : { type, strength: 2, center: [0, 1, 0], radius: 1 };
  return {
    id: uid(),
    name: type[0].toUpperCase() + type.slice(1),
    options: resolveForce(options),
    targets: null,
  };
}
export function newDocument(): SimulationDocument {
  return {
    format: 'fire-pro',
    version: 11,
    id: uid(),
    name: 'Untitled simulation',
    scene: {
      recipe: 'empty',
      sky: false,
      floor: { roughness: 0.58, metalness: 0.08 },
    },
    camera: { position: [6, 4.5, 7], target: [0, 1.7, 0] },
    simulation: resolveSimulation(),
    emitters: [],
    forces: [],
    colliders: [],
  };
}
function object(v: unknown, path: string, allowed: string[]): asserts v is Record<string, any> {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw Error(`${path}: expected an object.`);
  for (const k of Object.keys(v))
    if (!allowed.includes(k)) throw Error(`${path}.${k}: unknown property.`);
}
/** Ranges of document fields that the library API does not validate itself. */
export const DOCUMENT_LIMITS = {
  floor: { roughness: [0, 1], metalness: [0, 1] },
  emitter: { size: [0.01, 16] },
} as const;
function num(v: unknown, path: string, min = -1e5, max = 1e5): asserts v is number {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max)
    throw Error(`${path}: expected a finite number from ${min} to ${max}.`);
}
function vec(v: unknown, path: string, min = -1e5, max = 1e5): asserts v is Vec3 {
  if (!Array.isArray(v) || v.length !== 3) throw Error(`${path}: expected X, Y, Z.`);
  v.forEach((x, i) => num(x, `${path}[${i}]`, min, max));
}
function text(v: unknown, path: string): asserts v is string {
  if (typeof v !== 'string' || !v.trim() || v.length > 120)
    throw Error(`${path}: use 1–120 characters.`);
}
function list(v: unknown, path: string): asserts v is any[] {
  if (!Array.isArray(v)) throw Error(`${path}: expected a list.`);
}
function bool(v: unknown, path: string) {
  if (typeof v !== 'boolean') throw Error(`${path}: expected true or false.`);
}
function pose(v: Record<string, any>, path: string) {
  vec(v.position, `${path}.position`);
  vec(v.rotation, `${path}.rotation`);
}
/** Strict current-format validation, shared by commands and all import/export paths. */
export function validateDocument(input: unknown): SimulationDocument {
  input = structuredClone(input);
  // Check the format first, so an obsolete preset gets this message, not a property error.
  const header = input as { format?: unknown; version?: unknown } | null;
  if (header?.format !== 'fire-pro' || header.version !== 11)
    throw Error('Unsupported preset format. Export a current Fire Pro preset.');
  object(input, 'Simulation', [
    'format',
    'version',
    'id',
    'name',
    'scene',
    'camera',
    'simulation',
    'emitters',
    'forces',
    'colliders',
  ]);
  const ids = new Set<string>();
  const identity = (v: Record<string, any>, path: string) => {
    text(v.id, `${path}.id`);
    text(v.name, `${path}.name`);
    if (ids.has(v.id)) throw Error(`${path}: duplicate ID ${v.id}.`);
    ids.add(v.id);
  };
  identity(input, 'Simulation');
  object(input.scene, 'scene', ['recipe', 'sky', 'floor']);
  const s = input.scene;
  if (!RECIPES.includes(s.recipe)) throw Error('scene.recipe: unknown built-in scene.');
  bool(s.sky, 'scene.sky');
  object(s.floor, 'floor', ['roughness', 'metalness']);
  num(s.floor.roughness, 'floor.roughness', ...DOCUMENT_LIMITS.floor.roughness);
  num(s.floor.metalness, 'floor.metalness', ...DOCUMENT_LIMITS.floor.metalness);
  object(input.camera, 'camera', ['position', 'target']);
  vec(input.camera.position, 'camera.position');
  vec(input.camera.target, 'camera.target');
  num(input.simulation?.brickSize, 'simulation.brickSize', 8, 32);
  input.simulation = resolveSimulation(input.simulation);
  list(input.emitters, 'emitters');
  list(input.forces, 'forces');
  list(input.colliders, 'colliders');
  for (const e of input.emitters) {
    object(e, 'emitter', [
      'id',
      'name',
      'mode',
      'shape',
      'radius',
      'size',
      'position',
      'rotation',
      'options',
      'burst',
    ]);
    identity(e, 'emitter');
    pose(e, 'emitter');
    if (!['continuous', 'burst'].includes(e.mode)) throw Error('Unknown emitter mode.');
    if (!['sphere', 'box', 'disk', 'torus', 'letters'].includes(e.shape))
      throw Error('Unknown emitter shape.');
    if (e.shape === 'letters' && s.recipe !== 'flaming-letters')
      throw Error('Letter geometry requires the Flaming letters scene.');
    num(e.radius, 'emitter.radius', ...LIMITS.emitter.radius);
    vec(e.size, 'emitter.size', ...DOCUMENT_LIMITS.emitter.size);
    const resolvedEmitter = resolveEmitter(
      { ...e.options, shape: { type: 'sphere', radius: e.radius } },
      emitterDefaults('emitter'),
      true,
    );
    const { shape: _shape, ...settings } = resolvedEmitter;
    e.options = settings;
    e.burst = resolveExplosion(e.burst);
  }
  const targets = new Set(
    input.emitters
      .filter((e: SourceDocument) => e.mode !== 'burst')
      .map((e: SourceDocument) => e.id),
  );
  for (const f of input.forces) {
    object(f, 'force', ['id', 'name', 'options', 'targets']);
    identity(f, 'force');
    f.options = resolveForce(f.options);
    if (f.targets !== null) {
      list(f.targets, 'force.targets');
      if (
        new Set(f.targets).size !== f.targets.length ||
        f.targets.some((id: string) => !targets.has(id))
      )
        throw Error('Force targets must be unique continuous emitters.');
    }
  }
  for (const c of input.colliders) {
    object(c, 'collider', ['id', 'name', 'shape', 'size', 'radius', 'position', 'rotation']);
    identity(c, 'collider');
    pose(c, 'collider');
    if (!['sphere', 'box'].includes(c.shape)) throw Error('Invalid collider shape.');
    vec(c.size, 'collider.size', ...LIMITS.collider.size);
    num(c.radius, 'collider.radius', ...LIMITS.collider.radius);
  }
  return structuredClone(input) as SimulationDocument;
}
/** A copy with new IDs for the preset and every object in it. */
export function cloneDocument(document: SimulationDocument): SimulationDocument {
  const copy = structuredClone(document),
    ids = new Map(copy.emitters.map((e) => [e.id, uid()]));
  copy.id = uid();
  copy.emitters.forEach((e) => (e.id = ids.get(e.id)!));
  copy.forces.forEach((f) => {
    f.id = uid();
    f.targets = f.targets?.map((id) => ids.get(id)!) ?? null;
  });
  copy.colliders.forEach((c) => (c.id = uid()));
  return copy;
}
