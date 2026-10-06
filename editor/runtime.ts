export { demoActions } from './demo-actions';
import * as THREE from 'three/webgpu';
import { ClusteredLighting } from 'three/addons/lighting/ClusteredLighting.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { FireSimulation } from '../src/library/FireSimulation';
import { Emitter, Explosion, type Force, type Collider } from '../src/library/handles';
import { createLetterGeometry } from './scenes/letters';
import { createPreviewSky } from './scenes/sky';
import { disposeScene } from './scenes/primitives';
import { PropAssets } from './scenes/props';
import { recipes } from './recipes';
import { wanderPosition } from './scenes/motion';
import type {
  DebugField,
  SimulationOptions,
  SimulationConfigureOptions,
} from '../src/library/options';
import type { SimulationDocument, SourceDocument, Transform } from './document';
import type { DemoScene } from './scenes/types';
import { createGpuTimings, type GpuTimingsSnapshot } from './gpu-timings';
import { FrameTimingHistory, type FrameTiming } from './frame-timing';

/** The simulation and the preview objects built for it. A change to a setting that needs a
 * new simulation replaces all of it. */
interface LiveSimulation {
  simulation: FireSimulation;
  structural: string;
  sources: Map<string, Emitter | Explosion>;
  objects: Map<string, THREE.Object3D>;
  guides: Map<string, THREE.Mesh>;
  forces: Force[];
  forceKey: string;
  sourceKeys: Map<string, string>;
  sourceResources: Map<string, THREE.Mesh[]>;
  sourceColliders: Map<string, Collider>;
  resources: THREE.Mesh[];
}
export interface RuntimeStats {
  fps: number | null;
  frameTimeMs: number | null;
  /** The most voxels the simulation may store and compute. */
  voxelBudget: number;
  /** Voxels the solver computed: the bricks near content and sources. */
  activeVoxelCount: number;
  /** The content needs more than the voxel budget. */
  gridLimited: boolean;
  estimatedMemoryBytes: number;
}
interface RuntimeOptions {
  select?(id: string): void;
  transform?(id: string, value: Transform): void;
  error?(error: Error): void;
  camera?(position: [number, number, number], target: [number, number, number]): void;
  editor?: boolean;
}
/** Sphere, disk and torus guides scale with radius; box and letter guides use size. */
/** How far, in meters, the moving-emitter recipe wanders from its authored position. */
const WANDER_REACH = [1.4, 1.9, 1.4];
const radialGuide = (e: SourceDocument) =>
  e.mode === 'burst' || e.shape === 'sphere' || e.shape === 'disk' || e.shape === 'torus';
/** Disk and torus sources lie flat, facing up, like a fire bed. */
const shapeGeometry = (shape: 'disk' | 'torus', radius: number) =>
  (shape === 'disk'
    ? new THREE.CircleGeometry(radius, 40)
    : new THREE.TorusGeometry(radius, radius * 0.4, 12, 40)
  ).rotateX(-Math.PI / 2);
const pose = (object: THREE.Object3D, value: Transform) => {
  object.position.fromArray(value.position);
  object.rotation.set(...value.rotation);
};
export class SimulationRuntime {
  readonly renderer = new THREE.WebGPURenderer({ antialias: true });
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(44, 1, 0.05, 250);
  live?: LiveSimulation;
  readonly controls: OrbitControls;
  readonly gizmo: TransformControls;
  private sky?: Awaited<ReturnType<typeof createPreviewSky>>;
  private setup?: DemoScene;
  private props = new PropAssets();
  private observer: ResizeObserver;
  private alive = true;
  private frame = 0;
  private last = 0;
  private readonly frameHistory = new FrameTimingHistory();
  private elapsed = 0;
  private document: SimulationDocument;
  private selected = '';
  private transformMode: 'translate' | 'rotate' | null = null;
  private gizmosVisible = false;
  private bricksVisible = false;
  private field: DebugField = 'beauty';
  private forceGuide?: THREE.ArrowHelper;
  private revision = 0;
  private queue = Promise.resolve();
  private shots: { mesh: THREE.Mesh; emitter: Emitter }[] = [];
  private firstShot = false;
  private sun = new THREE.DirectionalLight(0xffffff, 3);
  private fill = new THREE.DirectionalLight(0xc5d5e6, 1.5);
  private ambient = new THREE.HemisphereLight(0xc5d5e6, 0x32313b, 0.7);
  paused = false;
  /** GPU pass timings, measured only while the profiler is on. */
  private timings?: ReturnType<typeof createGpuTimings>;
  private profiling = false;
  constructor(
    private container: HTMLElement,
    document: SimulationDocument,
    private options: RuntimeOptions = {},
  ) {
    this.document = document;
    // Every flame light is a point light, and adding or removing them recompiles nothing.
    this.renderer.lighting = new ClusteredLighting();
    // Volume marching is pixel-bound; keep the interactive viewport at CSS resolution.
    // DOM controls remain native-resolution and MSAA still smooths scene geometry.
    this.renderer.setPixelRatio(1);
    container.append(this.renderer.domElement);
    this.renderer.domElement.setAttribute('aria-label', 'Simulation viewport');
    this.camera.position.fromArray(document.camera.position);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.fromArray(document.camera.target);
    this.controls.enableDamping = true;
    this.controls.maxDistance = 60;
    this.controls.update();
    this.controls.addEventListener('end', () =>
      this.options.camera?.(this.camera.position.toArray(), this.controls.target.toArray()),
    );
    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSpace('local');
    this.scene.add(this.gizmo.getHelper());
    this.gizmo.addEventListener('dragging-changed', (e) => {
      this.controls.enabled = !e.value;
      if (!e.value && this.gizmo.object)
        this.options.transform?.(this.selected, {
          position: this.gizmo.object.position.toArray(),
          rotation: [
            this.gizmo.object.rotation.x,
            this.gizmo.object.rotation.y,
            this.gizmo.object.rotation.z,
          ],
        });
    });
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    let down = [0, 0];
    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      down = [e.clientX, e.clientY];
    });
    this.renderer.domElement.addEventListener('pointerup', (e) => {
      if (
        !options.editor ||
        this.gizmo.dragging ||
        Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4
      )
        return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      const ray = new THREE.Raycaster();
      ray.setFromCamera(
        new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          (-(e.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        this.camera,
      );
      ray.params.Line.threshold = 0.06;
      const live = this.live;
      const hits = ray.intersectObjects(
        (live ? [...live.guides.values()] : []).filter((object) => object.visible),
      );
      if (hits[0]) this.options.select?.(hits[0].object.userData.id);
    });
  }
  async initialize() {
    if (!navigator.gpu)
      throw Error('WebGPU is unavailable. You can still edit, import, and export the simulation.');
    await this.renderer.init();
    if (!this.alive) {
      // Three skips native cleanup until init finishes. A preset change may have
      // disposed this runtime while the adapter/device was still being requested.
      this.renderer.dispose();
      return;
    }
    this.timings = createGpuTimings(this.renderer);
    this.timings.setActive(this.profiling);
    this.sky = await createPreviewSky();
    if (!this.alive) {
      this.sky.dispose();
      return;
    }
    this.scene.backgroundNode = this.sky.node;
    this.scene.environment = this.sky.environment;
    this.scene.environmentIntensity = 1;
    this.sun.position.copy(this.sky.sunDirection).multiplyScalar(10);
    this.fill.position.set(4, 7, 6);
    this.scene.add(this.sun, this.sun.target, this.fill, this.ambient);
    await this.props.load(
      ['campfire', 'fireplace', 'fire-tornado'].includes(this.document.scene.recipe),
    );
    if (!this.alive) {
      this.props.dispose();
      return;
    }
    this.scene.userData.props = this.props;
    this.setup = recipes[this.document.scene.recipe].buildScene({
      scene: this.scene,
      camera: this.camera,
      renderer: this.renderer,
    });
    if (this.setup.objects.letters) (this.setup.objects.letters as THREE.Object3D).visible = false;
    await this.apply(this.document);
    if (!this.alive) return;
    this.last = performance.now();
    this.animate(this.last);
  }
  private resize() {
    const w = Math.max(1, this.container.clientWidth),
      h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
  private structural(d: SimulationDocument) {
    return JSON.stringify([
      d.simulation.voxelSize,
      d.simulation.velocityDivisor,
      d.simulation.brickSize,
      d.simulation.smokeDivisor,
      d.simulation.scalarMacCormack,
      d.simulation.seed,
      d.colliders,
    ]);
  }
  apply(document: SimulationDocument, allowRebuild = true) {
    this.document = document;
    const rev = ++this.revision;
    const next = this.queue
      .catch(() => {})
      .then(async () => {
        if (!this.alive || rev !== this.revision || !this.setup) return;
        this.sky!.blend.value = Number(document.scene.sky);
        this.sun.intensity = document.scene.sky ? 3 : 0;
        this.fill.intensity = document.scene.sky ? 1.5 : 0;
        this.ambient.intensity = document.scene.sky ? 0.7 : 0;
        this.scene.environmentIntensity = Number(document.scene.sky);
        let live = this.live;
        const structural = this.structural(document);
        if (!live || live.structural !== structural) {
          if (allowRebuild) {
            const fresh = await this.build(document, structural);
            if (!this.alive || rev !== this.revision) {
              this.disposeLive(fresh);
              return;
            }
            if (live) this.disposeLive(live);
            this.scene.add(fresh.simulation);
            this.live = live = fresh;
          } else live = undefined;
        }
        if (live) this.update(live, document, allowRebuild);
        this.select(this.selected);
        this.updateNozzle();
        this.updateMovingSphere();
        this.timings?.reset();
      });
    this.queue = next;
    return next;
  }
  /** Bring the live simulation, its sources and its forces in line with the document. */
  private update(live: LiveSimulation, document: SimulationDocument, allowRebuild: boolean) {
    const {
      voxelSize,
      brickSize,
      velocityDivisor,
      smokeDivisor,
      scalarMacCormack,
      seed,
      ...patch
    } = document.simulation;
    live.simulation.configure(patch as SimulationConfigureOptions);
    if (allowRebuild)
      for (const id of live.sources.keys())
        if (!document.emitters.some((e) => e.id === id)) this.removeSource(live, id);
    for (const e of document.emitters) {
      if (live.sourceKeys.get(e.id) !== this.sourceKey(e)) {
        if (!allowRebuild) continue;
        this.removeSource(live, e.id);
        this.addSource(live, e);
      }
      const source = live.sources.get(e.id)!;
      if (source instanceof Emitter)
        source.configure({
          ...e.options,
          ...(e.shape === 'sphere' ? { shape: { radius: e.radius } } : {}),
        });
      else source.configure(e.burst);
      const obj = live.objects.get(e.id)!;
      if (this.gizmo.object !== obj || !this.gizmo.dragging) pose(obj, e);
      const guide = live.guides.get(e.id);
      if (guide && radialGuide(e))
        guide.scale.setScalar(e.mode === 'burst' ? e.burst.radius! : e.radius);
    }
    const key = JSON.stringify(document.forces);
    if (key !== live.forceKey) {
      live.forces.forEach((f) => f.remove());
      live.forces = document.forces.map((f) =>
        live.simulation.addForce(
          f.options,
          f.targets === null ? undefined : f.targets.map((id) => live.sources.get(id) as Emitter),
        ),
      );
      live.forceKey = key;
    }
  }
  private sourceKey(e: SourceDocument) {
    return JSON.stringify([
      e.mode,
      e.shape,
      e.shape === 'box' ? e.size : e.shape === 'sphere' ? null : e.radius,
    ]);
  }
  private removeSource(live: LiveSimulation, id: string) {
    live.sources.get(id)?.remove();
    live.sourceColliders.get(id)?.remove();
    for (const mesh of live.sourceResources.get(id) ?? []) {
      mesh.removeFromParent();
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
      live.resources = live.resources.filter((item) => item !== mesh);
    }
    live.sources.delete(id);
    live.objects.delete(id);
    live.guides.delete(id);
    live.sourceKeys.delete(id);
    live.sourceResources.delete(id);
    live.sourceColliders.delete(id);
    live.forceKey = '';
  }
  private addSource(live: LiveSimulation, e: SourceDocument) {
    const firstResource = live.resources.length;
    let obj: THREE.Object3D;
    let source: Emitter | Explosion;
    if (e.mode === 'burst') {
      source = live.simulation.addExplosion(e.burst);
      obj = source.object;
    } else if (e.shape === 'sphere') {
      source = live.simulation.addEmitter({
        ...e.options,
        shape: { type: 'sphere', radius: e.radius },
      });
      obj = source.object;
    } else {
      const geo =
        e.shape === 'letters'
          ? createLetterGeometry()
          : e.shape === 'box'
            ? new THREE.BoxGeometry(...e.size)
            : shapeGeometry(e.shape, e.radius);
      const mesh = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({ color: 0x6b7076, metalness: 0.4, roughness: 0.35 }),
      );
      // Box, disk and torus only shape the emission; the letters are also the scene prop.
      // Hide the material, not the mesh, so the guide attached to it stays visible.
      mesh.material.visible = e.shape === 'letters';
      pose(mesh, e);
      live.simulation.add(mesh);
      live.resources.push(mesh);
      source = live.simulation.addEmitter({ ...e.options, shape: { type: 'mesh', object: mesh } });
      obj = mesh;
      if (e.shape === 'letters')
        live.sourceColliders.set(
          e.id,
          live.simulation.addCollider({ object: mesh, shape: { type: 'mesh' } }),
        );
    }
    pose(obj, e);
    if (
      this.document.scene.recipe === 'moving-emitter' &&
      e.mode === 'continuous' &&
      e.shape === 'sphere'
    ) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(1, 40, 24),
        new THREE.MeshStandardMaterial({
          color: 0x30140a,
          emissive: 0xff550c,
          emissiveIntensity: 1.8,
          metalness: 0.5,
          roughness: 0.38,
        }),
      );
      sphere.name = 'moving-sphere';
      sphere.scale.setScalar(e.radius * 0.78);
      obj.add(sphere);
      live.resources.push(sphere);
    }
    live.sources.set(e.id, source);
    live.objects.set(e.id, obj);
    const geo =
      e.mode === 'continuous' && (e.shape === 'disk' || e.shape === 'torus')
        ? shapeGeometry(e.shape, 1)
        : radialGuide(e)
          ? new THREE.SphereGeometry(1, 20, 12)
          : new THREE.BoxGeometry(...e.size);
    const guide = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        color: 0xed925e,
        wireframe: true,
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      }),
    );
    guide.userData.id = e.id;
    if (radialGuide(e)) guide.scale.setScalar(e.mode === 'burst' ? e.burst.radius! : e.radius);
    obj.add(guide);
    live.guides.set(e.id, guide);
    live.resources.push(guide);

    live.sourceKeys.set(e.id, this.sourceKey(e));
    live.sourceResources.set(e.id, live.resources.slice(firstResource));
    live.forceKey = '';
  }
  private async build(d: SimulationDocument, structural: string): Promise<LiveSimulation> {
    const simulation = new FireSimulation(d.simulation as SimulationOptions);
    simulation.debug({ bricks: this.bricksVisible, field: this.field });
    const result: LiveSimulation = {
      simulation,
      structural,
      sources: new Map(),
      objects: new Map(),
      guides: new Map(),
      forces: [],
      forceKey: '',
      sourceKeys: new Map(),
      sourceResources: new Map(),
      sourceColliders: new Map(),
      resources: [],
    };
    try {
      for (const e of d.emitters) this.addSource(result, e);
      for (const c of d.colliders) {
        const mesh = new THREE.Mesh(
          c.shape === 'sphere'
            ? new THREE.SphereGeometry(c.radius, 20, 12)
            : new THREE.BoxGeometry(...c.size),
          new THREE.MeshStandardMaterial({ color: 0x59636b }),
        );
        pose(mesh, c);
        simulation.add(mesh);
        result.objects.set(c.id, mesh);
        result.resources.push(mesh);
        simulation.addCollider({
          object: mesh,
          shape:
            c.shape === 'sphere'
              ? { type: 'sphere', radius: c.radius }
              : { type: 'box', size: c.size },
        });
      }
      // Existing environment props remain colliders in their original authored scenes.
      const o = this.setup!.objects as Record<string, any>;
      if (d.scene.recipe === 'fireplace')
        for (const [key, size] of [
          ['left', [0.52, 2.1, 1.1]],
          ['right', [0.52, 2.1, 1.1]],
          ['mantel', [3.5, 0.3, 1.5]],
        ] as const)
          simulation.addCollider({ object: o[key], shape: { type: 'box', size: [...size] } });
      if (d.scene.recipe === 'campfire')
        for (const rock of o.rocks)
          simulation.addCollider({ object: rock, shape: { type: 'sphere', radius: 0.25 } });
      if (d.scene.recipe === 'projectiles')
        simulation.addCollider({ object: o.target, shape: { type: 'box', size: [0.3, 2.5, 2.8] } });
      await simulation.initialize(this.renderer);
      return result;
    } catch (error) {
      this.disposeLive(result);
      throw error;
    }
  }
  /** Show the final image, or a single simulation field. */
  showField(field: DebugField) {
    this.field = field;
    this.live?.simulation.debug({ field });
  }
  /** Show or hide the outlines of the bricks the solver computes. */
  showBricks(visible: boolean) {
    this.bricksVisible = visible;
    this.live?.simulation.debug({ bricks: visible });
  }
  select(
    id: string,
    mode: 'translate' | 'rotate' | null = this.transformMode,
    gizmosVisible = this.gizmosVisible,
  ) {
    this.selected = id;
    this.transformMode = mode;
    this.gizmosVisible = gizmosVisible;
    const show = Boolean(this.options.editor && gizmosVisible);
    this.forceGuide?.removeFromParent();
    this.forceGuide?.dispose();
    this.forceGuide = undefined;
    this.gizmo.detach();
    this.gizmo.enabled = show;
    let selected: THREE.Object3D | undefined;
    const live = this.live;
    if (live) {
      const force = this.document.forces.find((f) => f.id === id);
      if (force && show && (force.options.type === 'wind' || force.options.type === 'vortex')) {
        const options = force.options;
        const direction = new THREE.Vector3()
          .fromArray(options.type === 'wind' ? options.direction : (options.axis ?? [0, 1, 0]))
          .normalize();
        const center = new THREE.Vector3().fromArray(
          options.type === 'vortex' ? (options.center ?? [0, 0, 0]) : [0, 0, 0],
        );
        this.forceGuide = new THREE.ArrowHelper(direction, center, 1, 0xed925e, 0.15, 0.08);
        live.simulation.add(this.forceGuide);
      }
      for (const [eid, guide] of live.guides) {
        const target =
          eid === id || (force && (force.targets === null || force.targets.includes(eid)));
        guide.visible = Boolean(show && target);
        (guide.material as THREE.MeshBasicMaterial).opacity = target ? 0.65 : 0.08;
      }
      if (live.objects.has(id)) selected = live.objects.get(id);
    }
    if (show && mode && selected) {
      this.gizmo.setMode(mode);
      this.gizmo.attach(selected);
    }
  }
  trigger(id?: string) {
    for (const [sid, source] of this.live?.sources ?? [])
      if (source instanceof Explosion && (!id || sid === id)) source.trigger();
  }
  setEmitting(id: string, active: boolean) {
    const source = this.live?.sources.get(id);
    if (source instanceof Emitter) source.configure({ active });
  }
  launch() {
    if (this.shots.length >= 6) return;
    const live = this.live;
    if (!live) return;
    const objects = this.setup!.objects as any;
    if (this.document.scene.recipe !== 'projectiles' || !objects.muzzle) return;
    const mesh = new THREE.Mesh(objects.projectileGeometry, objects.projectileMaterial);
    objects.muzzle.getWorldPosition(mesh.position);
    this.scene.add(mesh);
    const emitter = live.simulation.addFire({
      shape: { type: 'sphere', radius: 0.16 },
      emission: { flame: 0.8, heatRate: 1 },
      velocity: null,
    });
    mesh.add(emitter.object);
    this.shots.push({ mesh, emitter });
  }
  restart() {
    this.shots.forEach((s) => {
      s.emitter.remove();
      s.mesh.removeFromParent();
    });
    this.shots = [];
    this.elapsed = 0;
    this.firstShot = false;
    this.live?.simulation.reset();
    this.updateNozzle();
    this.updateMovingSphere();
    this.timings?.reset();
  }
  private updateMovingSphere() {
    if (this.document.scene.recipe !== 'moving-emitter') return;
    const path = wanderPosition(this.elapsed);
    const live = this.live;
    if (!live) return;
    for (const source of this.document.emitters) {
      if (source.mode !== 'continuous' || source.shape !== 'sphere') continue;
      const object = live.objects.get(source.id);
      if (!object) continue;
      const position = source.position.map(
        (center, axis) => center + path[axis] * WANDER_REACH[axis],
      );
      object.position.fromArray(position);
      object.getObjectByName('moving-sphere')?.scale.setScalar(source.radius * 0.78);
    }
  }
  private updateNozzle() {
    if (this.document.scene.recipe !== 'flamethrower' || !this.setup) return;
    const source = this.document.emitters.find((e) => e.mode === 'continuous');
    const object = source && this.live?.objects.get(source.id);
    if (!source || !object) return;
    const yaw = Math.sin(this.elapsed * 0.85) * 0.38;
    pose(object, source);
    object.rotation.y += yaw;
    // Rotate about the swivel, keeping the emitter precisely at the moving muzzle.
    object.position.x += 0.39 * (Math.cos(yaw) - 1);
    object.position.z -= 0.39 * Math.sin(yaw);
    object.updateWorldMatrix(true, false);
    const nozzle = this.setup.objects.nozzle as THREE.Group;
    object.getWorldQuaternion(nozzle.quaternion);
    object.getWorldPosition(nozzle.position);
    nozzle.position.add(new THREE.Vector3(-0.39, 0, 0).applyQuaternion(nozzle.quaternion));
  }
  /** Advance authored behavior and the simulation by a supplied timestep. */
  advance(dt: number) {
    this.elapsed += dt;
    this.updateNozzle();
    this.updateMovingSphere();
    if (this.document.scene.recipe === 'projectiles') {
      if (!this.firstShot && this.elapsed > 0.6) {
        this.firstShot = true;
        this.launch();
      }
      for (let i = this.shots.length - 1; i >= 0; i--) {
        const s = this.shots[i];
        s.mesh.position.x += dt * 5;
        if (s.mesh.position.x >= 3.25) {
          this.trigger();
          s.emitter.remove();
          s.mesh.removeFromParent();
          this.shots.splice(i, 1);
        }
      }
    }
    const live = this.live;
    if (live) {
      live.simulation.update(dt);
    }
  }
  /** Draw the scene. */
  render() {
    this.renderer.render(this.scene, this.camera);
  }
  /** Measure GPU pass timings for the stats panel. */
  setProfiling(active: boolean) {
    this.profiling = active;
    this.timings?.setActive(active);
  }
  get profile(): GpuTimingsSnapshot | null {
    return this.timings?.snapshot() ?? null;
  }
  get frameTiming(): FrameTiming {
    return this.alive
      ? this.frameHistory.sample(performance.now())
      : { fps: null, frameTimeMs: null };
  }
  get stats(): RuntimeStats {
    const stats = this.live?.simulation.stats;
    const { fps, frameTimeMs } = this.frameTiming;
    return {
      fps,
      frameTimeMs,
      voxelBudget: this.live?.simulation.getOptions().grid.maxVoxels ?? 0,
      activeVoxelCount: stats?.activeVoxels ?? 0,
      gridLimited: stats?.gridLimited ?? false,
      estimatedMemoryBytes: stats?.estimatedMemoryBytes ?? 0,
    };
  }
  private animate = (now: number) => {
    if (!this.alive) return;
    this.frame = requestAnimationFrame(this.animate);
    // Measure the actual frame interval, not the clamped simulation time step.
    const interval = now - this.last;
    const dt = Math.max(0, Math.min(interval / 1000, 0.1));
    this.last = now;
    this.controls.update();
    try {
      if (this.profiling) this.timings?.beginFrame(now);
      if (!this.paused) this.advance(dt);
      this.render();
      this.frameHistory.record(now, interval);
    } catch (error) {
      cancelAnimationFrame(this.frame);
      this.options.error?.(error as Error);
    } finally {
      if (this.profiling) this.timings?.endFrame();
    }
  };
  private disposeLive(v: LiveSimulation) {
    this.gizmo.detach();
    v.simulation.dispose();
    v.resources.forEach((m) => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
  }
  dispose() {
    this.alive = false;
    this.revision++;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.timings?.dispose();
    this.shots.forEach((s) => {
      s.emitter.remove();
      s.mesh.removeFromParent();
    });
    if (this.live) this.disposeLive(this.live);
    this.live = undefined;
    this.forceGuide?.dispose();
    this.gizmo.dispose();
    this.controls.dispose();
    this.setup?.dispose?.();
    this.sky?.dispose();
    disposeScene(this.scene);
    this.props.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
/** Standalone exported applications share exactly the editor's runtime. */
export async function startSimulation(container: HTMLElement, document: SimulationDocument) {
  const runtime = new SimulationRuntime(container, document);
  try {
    await runtime.initialize();
    return runtime;
  } catch (error) {
    runtime.dispose();
    throw error;
  }
}
