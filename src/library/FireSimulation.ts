import {
  BufferAttribute,
  BufferGeometry,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Object3D,
  Vector3,
  type WebGPURenderer,
} from 'three/webgpu';
import { FluidSimulation } from '../engine/FluidSimulation.ts';
import { DEFAULT_FLAME_SETTINGS } from '../engine/flame.ts';
import {
  FIXED_DT,
  DEFAULT_POOL_LIMITS,
  initialPoolSlots,
  memoryEstimate,
  poolSide,
  renderMemoryEstimate,
  type SimulationFrame,
} from '../engine/types.ts';
import { Collider, Emitter, Explosion, Force, snapshot, type SimulationHost } from './handles.ts';
import { SceneVolumeRenderer } from './SceneVolumeRenderer.ts';
import type { LightAnchor } from './SceneLights.ts';
import {
  DEBUG_FIELDS,
  GROUPS,
  boolean,
  emitterDefaults,
  keys,
  number,
  resolveCollider,
  resolveSettings,
  resolveSimulation,
  type DeepReadonly,
  type ColliderOptions,
  type DebugField,
  type DebugOptions,
  type EmitterKind,
  type EmitterOptions,
  type ExplosionOptions,
  type ForceOptions,
  type ResolvedExplosionOptions,
  type ResolvedSimulationOptions,
  type SimulationConfigureOptions,
  type SimulationOptions,
  type Vec3,
} from './options.ts';

/** The twelve edges of a unit box, as pairs of corners. */
const BOX_EDGES: [Vec3, Vec3][] = [
  [
    [0, 0, 0],
    [1, 0, 0],
  ],
  [
    [0, 1, 0],
    [1, 1, 0],
  ],
  [
    [0, 0, 1],
    [1, 0, 1],
  ],
  [
    [0, 1, 1],
    [1, 1, 1],
  ],
  [
    [0, 0, 0],
    [0, 1, 0],
  ],
  [
    [1, 0, 0],
    [1, 1, 0],
  ],
  [
    [0, 0, 1],
    [0, 1, 1],
  ],
  [
    [1, 0, 1],
    [1, 1, 1],
  ],
  [
    [0, 0, 0],
    [0, 0, 1],
  ],
  [
    [1, 0, 0],
    [1, 0, 1],
  ],
  [
    [0, 1, 0],
    [0, 1, 1],
  ],
  [
    [1, 1, 0],
    [1, 1, 1],
  ],
];

interface PendingExplosion {
  owner: Explosion;
  position: Vec3;
  options: ResolvedExplosionOptions;
  id: number;
}
export interface SimulationStats {
  /** Seconds simulated since initialization or the last reset. */
  simulationTime: number;
  /** Seconds of `update()` time not simulated: each call runs at most one step. */
  droppedTime: number;
  /** Requested GPU bytes for owned simulation and volume-rendering resources, including
   * screen-sized volume slices. Before initialization, an allocation estimate. Excludes
   * Three.js scene targets/geometry, driver alignment, and transient growth overlap. */
  estimatedMemoryBytes: number;
  /** The content needs more than `grid.maxVoxels`; bricks it reaches past the budget stay
   * empty. */
  gridLimited: boolean;
  /** Flame, heat, smoke and fuel cells the solver computed on the latest measured step:
   * the bricks near content and sources. */
  activeVoxels: number;
}
/** The solver field a debug view copies out of the solver after every step, if any. */
function solverField(field: DebugField): 'pressure' | 'divergence' | undefined {
  return field === 'pressure' || field === 'divergence' ? field : undefined;
}
/** A real-time flame simulation in a Three.js scene. It runs in world space, wherever its
 * emitters are: the ground is world y = 0, and storage follows the fire and smoke. Keep its
 * own transform at the identity. */
export class FireSimulation extends Object3D {
  readonly isFireSimulation = true;
  private options: ResolvedSimulationOptions;
  private readonly host: SimulationHost = {
    assertAlive: () => this.assertAlive(),
    assertReady: () => this.assertReady(),
    detach: (handle) => this.detach(handle),
    queueExplosion: (source, position, options) => this.queueExplosion(source, position, options),
  };
  private readonly emitters = new Set<Emitter>();
  private readonly explosions = new Set<Explosion>();
  private readonly forces = new Set<Force>();
  private readonly colliders = new Set<Collider>();
  private pending: PendingExplosion[] = [];
  private owner?: WebGPURenderer;
  private initialization?: Promise<void>;
  private fluid?: FluidSimulation;
  private surface?: SceneVolumeRenderer;
  /** Explosions whose flames still light the scene, by their burst. */
  private bursts: { owner: Explosion; anchor: LightAnchor }[] = [];
  /** Outlines of the bricks that hold storage, while the brick view is on, and when they
   * were last read back. */
  private brickView?: LineSegments<BufferGeometry, LineBasicMaterial>;
  private brickRead = { at: -Infinity, pending: false };
  private debugField: DebugField = 'beauty';
  private accumulator = 0;
  private droppedTime = 0;
  private disposed = false;
  private nextHandle = 1;
  private nextEvent = 1;

  constructor(options: SimulationOptions = {}) {
    super();
    this.name = 'Fire Simulation';
    this.options = resolveSimulation(options);
  }
  private assertAlive(): void {
    if (this.disposed) throw new Error('This FireSimulation has been disposed.');
  }
  private assertReady(): void {
    this.assertAlive();
    if (!this.fluid || !this.surface)
      throw new Error(
        'Await simulation.initialize(renderer) before updating or triggering effects.',
      );
  }
  initialize(renderer: WebGPURenderer): Promise<void> {
    this.assertAlive();
    if (this.owner && this.owner !== renderer)
      return Promise.reject(
        new Error('FireSimulation is already associated with another renderer.'),
      );
    if (this.initialization) return this.initialization;
    this.owner = renderer;
    this.initialization = this.initializeResources(renderer);
    return this.initialization;
  }
  private async initializeResources(renderer: WebGPURenderer): Promise<void> {
    if (renderer.reversedDepthBuffer || renderer.logarithmicDepthBuffer)
      throw new Error(
        'FireSimulation requires conventional depth; reversed and logarithmic depth are not supported.',
      );
    if (typeof renderer.init !== 'function')
      throw new Error('FireSimulation requires THREE.WebGPURenderer.');
    await renderer.init();
    this.assertAlive();
    const device = (renderer.backend as unknown as { device?: GPUDevice }).device;
    if (!device)
      throw new Error('FireSimulation requires a WebGPURenderer running the WebGPU backend.');
    let fluid: FluidSimulation | undefined;
    let surface: SceneVolumeRenderer | undefined;
    try {
      fluid = await FluidSimulation.create(device, {
        voxelSize: this.options.voxelSize,
        velocityDivisor: this.options.velocityDivisor,
        brickSize: this.options.brickSize,
        smokeDivisor: this.options.smokeDivisor,
        scalarMacCormack: this.options.scalarMacCormack,
      });
      this.assertAlive();
      fluid.showSolverField(solverField(this.debugField));
      surface = await SceneVolumeRenderer.create(
        renderer,
        fluid,
        this,
        () => this.options,
        () => this.lightAnchors(),
        this.debugField,
      );
      this.assertAlive();
      this.fluid = fluid;
      this.surface = surface;
      this.add(surface);
    } catch (error) {
      surface?.dispose();
      fluid?.dispose();
      throw error;
    }
  }
  addFire(options: EmitterOptions = {}): Emitter {
    return this.addContinuous('fire', options);
  }
  addFlameJet(options: EmitterOptions = {}): Emitter {
    return this.addContinuous('jet', options);
  }
  addSmoke(options: EmitterOptions = {}): Emitter {
    return this.addContinuous('smoke', options);
  }
  addEmitter(options: EmitterOptions = {}): Emitter {
    return this.addContinuous('emitter', options);
  }
  private addContinuous(kind: EmitterKind, options: EmitterOptions): Emitter {
    this.assertAlive();
    const emitter = new Emitter(
      this.host,
      `emitter-${this.nextHandle}`,
      options,
      emitterDefaults(kind),
      this.options.voxelSize,
      (this.options.seed + this.nextHandle) >>> 0,
    );
    if (!emitter.object.parent) this.add(emitter.object);
    this.emitters.add(emitter);
    this.nextHandle++;
    return emitter;
  }
  addExplosion(options: ExplosionOptions = {}): Explosion {
    this.assertAlive();
    const explosion = new Explosion(this.host, `explosion-${this.nextHandle}`, options);
    this.add(explosion.object);
    this.explosions.add(explosion);
    this.nextHandle++;
    return explosion;
  }
  /** A force everywhere, or only within the union of the target emitters' regions. */
  addForce(options: ForceOptions, targets?: readonly Emitter[]): Force {
    this.assertAlive();
    if (targets?.some((target) => !this.emitters.has(target)))
      throw new Error('Force target must be a live emitter owned by this simulation.');
    const force = new Force(this.host, options, targets);
    this.forces.add(force);
    targets?.forEach((target) => target.forces.add(force));
    return force;
  }
  addCollider(options: ColliderOptions): Collider {
    this.assertAlive();
    const collider = new Collider(this.host, resolveCollider(options));
    this.colliders.add(collider);
    return collider;
  }
  /** Change any settings except voxelSize, brickSize, velocityDivisor, smokeDivisor, scalarMacCormack and seed without clearing the simulation. */
  configure(patch: SimulationConfigureOptions): this {
    this.assertAlive();
    keys(patch, GROUPS, 'simulation.configure');
    const next = { ...this.options, ...resolveSettings(patch, this.options) };
    this.options = next;
    this.surface?.invalidate();
    this.surface?.update();
    return this;
  }
  getOptions(): DeepReadonly<ResolvedSimulationOptions> {
    this.assertAlive();
    return snapshot(this.options);
  }
  get stats(): DeepReadonly<SimulationStats> {
    this.assertAlive();
    const fluid = this.fluid;
    const divisor = this.options.velocityDivisor;
    const cells = this.options.brickSize;
    // Memory follows the pools' slots and the tiles, which grow with the content.
    const slots =
      fluid?.poolSlots ??
      initialPoolSlots(
        poolSide(divisor, DEFAULT_POOL_LIMITS, cells),
        Math.max(1, Math.floor(this.options.grid.maxVoxels / cells ** 3)),
      );
    const simulationMemory = fluid
      ? fluid.memoryBytes + fluid.meshMemoryBytes + fluid.forceMemoryBytes
      : memoryEstimate(
          slots,
          divisor,
          16,
          this.options.smokeDivisor,
          this.options.scalarMacCormack,
          cells,
        ) +
        // Element minima and growth reserve of the eight initial scene buffers.
        544;
    const boxBricks = (fluid?.renderBricks.boxes ?? []).reduce(
      (count, box) => count + box.hi.reduce((n, hi, axis) => n * (hi - box.lo[axis] + 1), 1),
      0,
    );
    return snapshot({
      simulationTime: fluid?.time ?? 0,
      droppedTime: this.droppedTime,
      gridLimited: Boolean(fluid?.poolLimited),
      activeVoxels: (fluid?.activeBrickCount ?? 0) * cells ** 3,
      estimatedMemoryBytes:
        simulationMemory +
        (this.surface?.memoryBytes ?? renderMemoryEstimate(
          slots, cells, boxBricks, this.options.rendering.lightingDivisor,
        )),
    });
  }
  /** View a single simulation field, or outline the bricks the solver computes. */
  debug(options: DebugOptions): this {
    this.assertAlive();
    keys(options, ['bricks', 'field'], 'debug');
    const bricks =
      options.bricks === undefined
        ? (this.brickView?.visible ?? false)
        : boolean(options.bricks, 'debug.bricks');
    const field = options.field ?? this.debugField;
    if (!DEBUG_FIELDS.includes(field)) throw new Error('Unknown debug field.');
    if (bricks && !this.brickView) {
      // Start with an empty position attribute: the material builds its shader for the
      // first geometry it draws, and every later geometry must match it.
      this.brickView = new LineSegments(
        new BufferGeometry().setAttribute('position', new BufferAttribute(new Float32Array(0), 3)),
        new LineBasicMaterial({ color: 0x5fd4c4, transparent: true, opacity: 0.6 }),
      );
      this.brickView.name = 'Fire Pro bricks';
      this.add(this.brickView);
    }
    if (this.brickView) this.brickView.visible = bricks;
    this.brickRead.at = -Infinity;
    this.debugField = field;
    this.fluid?.showSolverField(solverField(field));
    this.surface?.setDebug(field);
    // A paused simulation shows its bricks too.
    this.refreshBricks();
    return this;
  }
  /** Read the bricks back for the brick view, a few times a second. */
  private refreshBricks(): void {
    const view = this.brickView;
    const now = performance.now();
    if (!view?.visible || !this.fluid || this.brickRead.pending || now - this.brickRead.at < 250)
      return;
    this.brickRead = { at: now, pending: true };
    this.fluid
      .readBricks()
      .then((bricks) => {
        if (this.disposed || view !== this.brickView) return;
        // Twelve edges, two points each, per brick.
        const positions = new Float32Array(bricks.length * 72);
        let i = 0;
        for (const { min, max } of bricks)
          for (const [a, b] of BOX_EDGES) {
            positions.set(
              a.map((n, axis) => (n ? max : min)[axis]),
              i,
            );
            positions.set(
              b.map((n, axis) => (n ? max : min)[axis]),
              i + 3,
            );
            i += 6;
          }
        view.geometry.dispose();
        view.geometry = new BufferGeometry().setAttribute(
          'position',
          new BufferAttribute(positions, 3),
        );
      })
      .catch(() => {})
      .finally(() => (this.brickRead.pending = false));
  }
  /** Where flames light the scene from: every emitter, and every explosion while it burns
   * (SceneLights). */
  private lightAnchors(): readonly LightAnchor[] {
    const position = new Vector3();
    return [
      ...[...this.emitters].map((emitter) => {
        emitter.object.updateWorldMatrix(true, false);
        return {
          id: emitter.id,
          position: position.setFromMatrixPosition(emitter.object.matrixWorld).toArray(),
          transient: false,
        };
      }),
      ...this.bursts.map((burst) => burst.anchor),
    ];
  }
  private queueExplosion(
    owner: Explosion,
    position: Vec3,
    options: ResolvedExplosionOptions,
  ): void {
    this.pending.push({ owner, position: [...position], options, id: this.nextEvent++ });
  }
  private detach(handle: Emitter | Explosion | Force | Collider): void {
    if (handle instanceof Emitter) this.emitters.delete(handle);
    else if (handle instanceof Explosion) {
      this.explosions.delete(handle);
      this.pending = this.pending.filter((event) => event.owner !== handle);
      this.bursts = this.bursts.filter((burst) => burst.owner !== handle);
    } else if (handle instanceof Force) this.forces.delete(handle);
    else this.colliders.delete(handle);
  }
  /** Advance by `deltaSeconds` of real time in fixed 1/60 s steps, at most one a call: a
   * frame slower than a step drops the time it cannot keep up with (`stats.droppedTime`),
   * so the simulation slows down instead of running more steps every frame. */
  update(deltaSeconds: number): void {
    this.assertReady();
    number(deltaSeconds, 'update deltaSeconds', 0, Number.MAX_VALUE);
    assertWorldFrame(this);
    this.accumulator += deltaSeconds;
    if (this.accumulator + 1e-12 >= FIXED_DT) {
      this.accumulator = Math.max(0, this.accumulator - FIXED_DT);
      // Whole steps beyond this one are dropped; the remainder carries over.
      const behind = Math.floor((this.accumulator + 1e-12) / FIXED_DT) * FIXED_DT;
      this.accumulator = Math.max(0, this.accumulator - behind);
      this.droppedTime += behind;
      this.step();
    }
    const surface = this.surface!;
    surface.update();
    if (this.options.lighting.illuminateScene)
      this.bursts = this.bursts.filter((burst) => !surface.lightFinished(burst.anchor.id));
    else this.bursts = [];
    this.refreshBricks();
  }
  /** One fixed step of the emitters, explosions, forces and colliders as they are now. */
  private step(): void {
    const fluid = this.fluid!;
    fluid.cutoff = this.options.grid.cutoff;
    fluid.voxelBudget = this.options.grid.maxVoxels;
    fluid.ground = this.options.grid.ground;
    const sources = [...this.emitters].flatMap((emitter) => {
      const source = emitter.source();
      return source ? [source] : [];
    });
    const forceTargets = [...this.forces].flatMap((force) => {
      const value = force.frame();
      return value ? [value] : [];
    });
    // Mesh colliders voxelize on the lattice.
    const voxel = this.options.voxelSize;
    const colliders = [...this.colliders].map((collider) => collider.frame([voxel, voxel, voxel]));
    for (const event of this.pending) {
      const radius = event.options.radius;
      const id = `${event.owner.id}-burst-${event.id}`;
      if (this.options.lighting.illuminateScene)
        this.bursts.push({
          owner: event.owner,
          anchor: { id, position: [...event.position], transient: true },
        });
      sources.push({
        id,
        shape: 'sphere',
        position: [...event.position],
        size: [radius, radius, radius],
        flame: event.options.charge.flame,
        heatRate: event.options.charge.heat / FIXED_DT,
        smokeRate: event.options.charge.smoke / FIXED_DT,
        fuelRate: event.options.charge.fuel / FIXED_DT,
        velocity: [0, 0, 0],
        velocityResponse: event.options.outwardSpeed > 0 ? 60 : 0,
        outwardSpeed: event.options.outwardSpeed,
        variation: event.options.variation,
      });
    }
    const { flame, motion } = this.options;
    const frame: SimulationFrame = {
      simulation: {
        buoyancy: motion.buoyancy,
        smokeWeight: motion.smokeWeight,
        velocityDamping: motion.damping,
        cooling: flame.cooling,
        dissipation: this.options.smoke.dissipation,
        vorticity: motion.vorticity,
        seed: this.options.seed,
      },
      combustion: {
        lifespan: flame.lifespan,
        smokeRate: flame.smokeRate,
        heatRate: flame.heatRate,
        expansionRate: flame.expansionRate,
        ramps: DEFAULT_FLAME_SETTINGS.ramps,
      },
      fuel: this.options.fuel,
      sources,
      forces: forceTargets,
      colliders,
    };
    fluid.step(frame, FIXED_DT);
    this.pending = [];
  }
  reset(): void {
    this.assertReady();
    this.fluid!.reset();
    this.surface!.reset();
    this.pending = [];
    this.bursts = [];
    this.accumulator = 0;
    this.droppedTime = 0;
    this.nextEvent = 1;
  }
  dispose(): void {
    if (this.disposed) return;
    for (const emitter of this.emitters) emitter.remove();
    for (const explosion of this.explosions) explosion.remove();
    for (const force of this.forces) force.remove();
    for (const collider of this.colliders) collider.remove();
    this.disposed = true;
    this.pending = [];
    this.surface?.dispose();
    this.fluid?.dispose();
    this.brickView?.removeFromParent();
    this.brickView?.geometry.dispose();
    this.brickView?.material.dispose();
    this.removeFromParent();
  }
}

const IDENTITY = new Matrix4();
/** The simulation runs in world space, so its own transform must stay at the identity. */
function assertWorldFrame(object: Object3D): void {
  object.updateWorldMatrix(true, false);
  if (!object.matrixWorld.equals(IDENTITY))
    throw new Error(
      'FireSimulation runs in world space. Add it to the scene without moving, rotating or scaling it or its parents.',
    );
}
