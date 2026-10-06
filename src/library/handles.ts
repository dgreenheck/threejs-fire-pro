import { Box3, Matrix4, Mesh, Object3D, Vector3 } from 'three';
import type {
  MeshColliderMask,
  SimulationCollider,
  SimulationSource,
  SimulationForce,
} from '../engine/types.ts';
import { MeshCollider } from './MeshCollider.ts';
import { MeshSurface } from './MeshSurface.ts';
import {
  keys,
  resolveEmitter,
  resolveExplosion,
  resolveForce,
  vector,
  type ColliderOptions,
  type DeepReadonly,
  type EmitterOptions,
  type EmitterPatch,
  type ExplosionOptions,
  type ForceOptions,
  type ForcePatch,
  type ResolvedEmitterOptions,
  type ResolvedExplosionOptions,
  type ResolvedForceOptions,
  type Vec3,
} from './options.ts';

export interface SimulationHost {
  assertAlive(): void;
  assertReady(): void;
  detach(handle: Emitter | Explosion | Force | Collider): void;
  queueExplosion(source: Explosion, position: Vec3, options: ResolvedExplosionOptions): void;
}

export function snapshot<T>(value: T): DeepReadonly<T> {
  const copy = structuredClone(value);
  const freeze = (object: unknown) => {
    if (!object || typeof object !== 'object') return;
    Object.values(object).forEach(freeze);
    Object.freeze(object);
  };
  freeze(copy);
  return copy as DeepReadonly<T>;
}

/** Rigid transforms preserve declared dimensions, speed units and source values. */
function assertRigid(object: Object3D, label: string): void {
  object.updateWorldMatrix(true, false);
  const elements = object.matrixWorld.elements;
  if (!elements.every(Number.isFinite)) throw new Error(`${label} has a nonfinite transform.`);
  const axes = [0, 4, 8].map(
    (offset) => new Vector3(elements[offset], elements[offset + 1], elements[offset + 2]),
  );
  const invalidLength = axes.some((axis) => Math.abs(axis.lengthSq() - 1) > 1e-5);
  const invalidAngle =
    Math.abs(axes[0].dot(axes[1])) +
      Math.abs(axes[0].dot(axes[2])) +
      Math.abs(axes[1].dot(axes[2])) >
    1e-5;
  if (invalidLength || invalidAngle || object.matrixWorld.determinant() < 0) {
    throw new Error(
      `${label} requires unit world scale without reflection or shear. Set dimensions through options.`,
    );
  }
}

abstract class OwnedHandle {
  protected removed = false;
  protected readonly host: SimulationHost;
  /** @internal */
  constructor(host: SimulationHost) {
    this.host = host;
  }
  protected assertAlive(): void {
    this.host.assertAlive();
    if (this.removed) throw new Error('This Fire Pro handle has been removed.');
  }
}

export class Emitter extends OwnedHandle {
  readonly object = new Object3D();
  /** @internal */
  readonly forces = new Set<Force>();
  private options: ResolvedEmitterOptions;
  private readonly mesh?: Mesh;
  private readonly surface?: MeshSurface;
  private transformCache?: number[];
  private cachedSamples?: readonly { position: Vec3; weight: number }[];
  private readonly cellSize: number;
  /** @internal */
  readonly id: string;

  /** @internal */
  constructor(
    host: SimulationHost,
    id: string,
    options: EmitterOptions,
    defaults: ResolvedEmitterOptions,
    cellSize = 0.1,
    seed = 1,
  ) {
    super(host);
    this.id = id;
    this.cellSize = cellSize;
    this.options = resolveEmitter(options, defaults, true);
    this.object.name = `Fire Pro ${id}`;
    if (options.shape?.type === 'mesh') {
      this.mesh = options.shape.object;
      assertRigid(this.mesh, 'Mesh source');
      this.surface = new MeshSurface(this.mesh, cellSize, seed);
      this.mesh.add(this.object);
    }
  }
  getOptions(): DeepReadonly<ResolvedEmitterOptions> {
    this.assertAlive();
    return snapshot(this.options);
  }
  configure(patch: EmitterPatch): this {
    this.assertAlive();
    this.options = resolveEmitter(patch, this.options);
    return this;
  }
  start(): this {
    this.assertAlive();
    this.options.active = true;
    return this;
  }
  stop(): this {
    this.assertAlive();
    this.options.active = false;
    return this;
  }
  remove(): void {
    if (this.removed) return;
    for (const force of this.forces) force.detachTarget(this);
    this.forces.clear();
    this.removed = true;
    this.object.removeFromParent();
    this.host.detach(this);
  }
  /** @internal The source in world space. */
  source(includeInactive = false): SimulationSource | undefined {
    this.assertAlive();
    if (!this.options.active && !includeInactive) return;
    assertRigid(this.object, 'Emitter');
    const transform = this.object.matrixWorld;
    const position = new Vector3().setFromMatrixPosition(transform);
    const velocity = new Vector3();
    if (this.options.velocity) {
      velocity
        .fromArray(this.options.velocity.direction)
        .transformDirection(transform)
        .multiplyScalar(this.options.velocity.speed);
    }
    if (this.mesh) this.surface!.assertUnchanged(this.mesh);
    // Samples are reused, with their coverage, while the emitter holds still.
    const transformChanged =
      !this.transformCache ||
      transform.elements.some((value, index) => value !== this.transformCache![index]);
    if (this.mesh && transformChanged) {
      this.transformCache = [...transform.elements];
      this.cachedSamples = this.surface!.samples.map((sample) => ({
        position: sample.position.clone().applyMatrix4(transform).toArray(),
        weight: sample.weight,
      }));
    }
    const size: Vec3 =
      this.options.shape.type === 'sphere'
        ? [this.options.shape.radius, this.options.shape.radius, this.options.shape.radius]
        : [this.cellSize, this.cellSize, this.cellSize];
    return {
      id: this.id,
      shape: this.options.shape.type,
      position: position.toArray(),
      size,
      ...this.options.emission,
      velocity: velocity.toArray(),
      velocityResponse: this.options.velocity ? 60 : 0,
      outwardSpeed: 0,
      samples: this.cachedSamples,
      surfaceArea: this.surface?.surfaceArea,
    };
  }
}

export class Explosion extends OwnedHandle {
  readonly object = new Object3D();
  private options: ResolvedExplosionOptions;
  /** @internal */
  readonly id: string;
  /** @internal */
  constructor(host: SimulationHost, id: string, options: ExplosionOptions = {}) {
    super(host);
    this.id = id;
    this.options = resolveExplosion(options);
    this.object.name = `Fire Pro ${id}`;
  }
  configure(patch: ExplosionOptions): this {
    this.assertAlive();
    this.options = resolveExplosion(patch, this.options);
    return this;
  }
  getOptions(): DeepReadonly<ResolvedExplosionOptions> {
    this.assertAlive();
    return snapshot(this.options);
  }
  trigger(options: { worldPosition?: Vec3 } = {}): this {
    this.assertAlive();
    this.host.assertReady();
    keys(options, ['worldPosition'], 'trigger');
    assertRigid(this.object, 'Explosion');
    const position =
      options.worldPosition !== undefined
        ? vector(options.worldPosition, 'worldPosition')
        : new Vector3().setFromMatrixPosition(this.object.matrixWorld).toArray();
    this.host.queueExplosion(this, position, structuredClone(this.options));
    return this;
  }
  remove(): void {
    if (this.removed) return;
    this.removed = true;
    this.object.removeFromParent();
    this.host.detach(this);
  }
}

export class Force extends OwnedHandle {
  private options: ResolvedForceOptions;
  /** @internal */
  private targets?: Emitter[];
  /** @internal */
  constructor(host: SimulationHost, options: ForceOptions, targets?: readonly Emitter[]) {
    super(host);
    this.targets = targets === undefined ? undefined : [...new Set(targets)];
    this.options = resolveForce(options);
  }
  configure(patch: ForcePatch): this {
    this.assertAlive();
    this.options = resolveForce(patch, this.options);
    return this;
  }
  getOptions(): DeepReadonly<ResolvedForceOptions> {
    this.assertAlive();
    return snapshot(this.options);
  }
  /** @internal */
  frame(): SimulationForce | undefined {
    if (!this.options.active) return;
    const options = this.options;
    const center = new Vector3().fromArray('center' in options ? options.center : [0, 0, 0]);
    const axis = new Vector3()
      .fromArray(
        options.type === 'wind'
          ? options.direction
          : options.type === 'vortex'
            ? options.axis
            : [0, 1, 0],
      )
      .normalize();
    return {
      type: options.type,
      sources: this.targets?.map((target) => target.source(true)!),
      center: center.toArray(),
      vector: axis.toArray(),
      strength: options.strength,
      scale:
        options.type === 'turbulence' ? options.scale : 'radius' in options ? options.radius : 1,
      lift: options.type === 'vortex' ? options.lift : 0,
      inward: options.type === 'vortex' ? options.inward : 0,
    };
  }
  /** @internal Removing a target never turns an empty field into one that applies everywhere. */
  detachTarget(target: Emitter): void {
    this.targets = this.targets?.filter((entry) => entry !== target);
    target.forces.delete(this);
  }
  remove(): void {
    if (this.removed) return;
    this.removed = true;
    this.targets?.forEach((target) => target.forces.delete(this));
    this.host.detach(this);
  }
}

export class Collider extends OwnedHandle {
  private readonly options: ColliderOptions;
  private readonly meshCollider?: MeshCollider;
  private maskKey = '';
  private cachedMask!: MeshColliderMask;
  /** @internal */
  constructor(host: SimulationHost, options: ColliderOptions) {
    super(host);
    this.options = options;
    assertRigid(options.object, 'Collider');
    if (options.shape.type === 'mesh') this.meshCollider = new MeshCollider(options.object as Mesh);
  }
  /** Occupancy of a box of lattice cells around the mesh, with cells of `spacing` from
   * the world origin. The result is reused while the mesh holds still. */
  private meshMask(spacing: Vec3): MeshColliderMask {
    assertRigid(this.options.object, 'Mesh collider');
    const mesh = this.options.object as Mesh;
    const transform = mesh.matrixWorld;
    const key = [...transform.elements, ...spacing].join(',');
    if (key === this.maskKey) return this.cachedMask;
    mesh.geometry.computeBoundingBox();
    const bounds = new Box3().copy(mesh.geometry.boundingBox!).applyMatrix4(transform);
    // One cell of margin keeps the voxelization's cell centers inside the box.
    const lo = [0, 1, 2].map(
      (axis) => Math.floor(bounds.min.getComponent(axis) / spacing[axis]) - 1,
    ) as Vec3;
    const size = [0, 1, 2].map(
      (axis) => Math.ceil(bounds.max.getComponent(axis) / spacing[axis]) + 1 - lo[axis],
    ) as Vec3;
    // The voxelizer's box is centered in X/Z and starts at its floor.
    const toBox = new Matrix4()
      .makeTranslation(
        -(lo[0] + size[0] / 2) * spacing[0],
        -lo[1] * spacing[1],
        -(lo[2] + size[2] / 2) * spacing[2],
      )
      .multiply(transform);
    const domain = size.map((n, axis) => n * spacing[axis]) as Vec3;
    this.cachedMask = { mask: this.meshCollider!.voxelize(mesh, toBox, domain, size), lo, size };
    this.maskKey = key;
    return this.cachedMask;
  }
  /** @internal The collider in world space. A mesh collider gives its occupancy of
   * lattice cells of `spacing`. */
  frame(spacing: Vec3): SimulationCollider {
    const shape = this.options.shape;
    if (shape.type === 'mesh') return { shape: 'mesh', occupancy: this.meshMask(spacing) };
    assertRigid(this.options.object, 'Collider');
    const transform = this.options.object.matrixWorld;
    const axes = [0, 1, 2].map((column) =>
      new Vector3().setFromMatrixColumn(transform, column).toArray(),
    ) as [Vec3, Vec3, Vec3];
    const size: Vec3 =
      shape.type === 'sphere'
        ? [shape.radius, shape.radius, shape.radius]
        : (shape.size.map((value) => value * 0.5) as Vec3);
    return {
      shape: shape.type,
      position: new Vector3().setFromMatrixPosition(transform).toArray(),
      size,
      axes,
    };
  }
  remove(): void {
    if (!this.removed) {
      this.removed = true;
      this.host.detach(this);
    }
  }
}
