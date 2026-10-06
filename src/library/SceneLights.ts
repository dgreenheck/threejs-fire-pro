import { Color, PointLight, Vector3, type Object3D, type WebGPURenderer } from 'three/webgpu';
import { ClusteredLighting } from 'three/addons/lighting/ClusteredLighting.js';
import { beginComputePass } from '../engine/GpuProfiler.ts';
import type { Vec3 } from '../engine/types.ts';
import type { RenderSource, VolumeLighting } from '../engine/VolumeLighting.ts';
import { SCENE_LIGHT_KERNELS, renderPipeline } from '../engine/renderKernels.ts';
import commonSource from '../engine/shaders/volume-common.wgsl?raw';
import emissionSource from '../engine/shaders/fire-emission.wgsl?raw';
import bricksSource from '../engine/shaders/volume-bricks.wgsl?raw';
import listSource from '../engine/shaders/bricks.wgsl?raw';
import poolSource from '../engine/shaders/pool.wgsl?raw';
import reductionSource from '../engine/shaders/scene-lights.wgsl?raw';

/** Floats summed per anchor (scene-lights.wgsl). */
const SUM_FLOATS = 8;
/** A light ends where its illuminance falls to this. */
const LIGHT_CUTOFF = 0.01;
/** Without ClusteredLighting, Three.js recompiles every lit material when lights are added
 * or removed: this many lights stay in the scene, handed to the anchors brightest at the
 * camera. */
const FIXED_LIGHTS = 4;
/** An explosion's light ends after this many readings once its brightness falls below
 * `FADED` of its peak. */
const SETTLE_READINGS = 10;
const FADED = 0.01;

/** Where flames light the scene from: an emitter, or an explosion while it burns. */
export interface LightAnchor {
  id: string;
  position: Vec3;
  /** An explosion's light, which ends once its flames die down. */
  transient: boolean;
}
/** An anchor's light as last measured. */
interface Measured {
  position: Vector3;
  color: Color;
  intensity: number;
  transient: boolean;
  peak: number;
  readings: number;
  /** The anchor's own light, with ClusteredLighting. */
  light?: PointLight;
}
/** A collection in flight. */
interface Request {
  simulation: RenderSource;
  lighting: VolumeLighting;
  params: GPUBuffer;
  sampler: GPUSampler;
  lut: GPUTexture;
  anchors: readonly LightAnchor[];
  intensity: number;
  voxelSize: number;
}

/**
 * Lights for the scene's surfaces: one per anchor, at the centroid of the flames nearest
 * it, colored and as bright as they are. Every brick's flames go to its nearest anchor on
 * the GPU (scene-lights.wgsl); the totals come back asynchronously, so the application's
 * update never waits.
 */
export class SceneLights {
  private readonly measured = new Map<string, Measured>();
  private readonly fixed: PointLight[] = [];
  private pipeline!: GPUComputePipeline;
  private anchors?: GPUBuffer;
  private sums?: GPUBuffer;
  private readback?: GPUBuffer;
  private pending = false;
  private deferred?: Request;
  private disposed = false;
  private generation = 0;
  /** Where the camera that last drew the simulation stood. */
  readonly viewer = new Vector3();

  private constructor(
    private readonly device: GPUDevice,
    private readonly parent: Object3D,
    private readonly renderer: WebGPURenderer,
  ) {}
  static async create(
    device: GPUDevice,
    parent: Object3D,
    renderer: WebGPURenderer,
  ): Promise<SceneLights> {
    const lights = new SceneLights(device, parent, renderer);
    const module = device.createShaderModule({
      label: 'Scene lights',
      code: [
        emissionSource,
        poolSource,
        bricksSource,
        listSource,
        commonSource,
        reductionSource,
      ].join('\n'),
    });
    lights.pipeline = await renderPipeline(
      device,
      module,
      'collectLights',
      SCENE_LIGHT_KERNELS.collectLights,
    );
    return lights;
  }
  /** Whether the application lights its scene with ClusteredLighting, which takes any
   * number of point lights without recompiling materials. */
  private get clustered(): boolean {
    return this.renderer.lighting instanceof ClusteredLighting;
  }
  /** Native reduction and readback buffers, including growth headroom. */
  get memoryBytes(): number {
    return (this.anchors?.size ?? 0) + (this.sums?.size ?? 0) + (this.readback?.size ?? 0);
  }
  /** Whether an explosion's light has died down, so its anchor can go. */
  finished(id: string): boolean {
    const measured = this.measured.get(id);
    return Boolean(
      measured?.transient &&
      measured.readings >= SETTLE_READINGS &&
      measured.intensity <= measured.peak * FADED,
    );
  }
  /** Collect the light of the simulation's latest step by anchor. A collection still in
   * flight defers this one until it lands. */
  sample(request: Request): void {
    if (this.disposed) return;
    if (this.pending) {
      this.deferred = request;
      return;
    }
    const { simulation, lighting, anchors } = request;
    const bricks = simulation.renderBricks;
    if (!anchors.length || !bricks.boxes.length) {
      this.apply(request, new Float32Array(anchors.length * SUM_FLOATS));
      return;
    }
    lighting.sync(simulation);
    const device = this.device;
    const sumBytes = anchors.length * SUM_FLOATS * 4;
    const anchorBytes = (anchors.length + 1) * 16;
    if (!this.anchors || this.anchors.size < anchorBytes) {
      this.anchors?.destroy();
      this.anchors = device.createBuffer({
        label: 'Scene light anchors',
        size: anchorBytes * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (!this.sums || this.sums.size < sumBytes) {
      this.sums?.destroy();
      this.readback?.destroy();
      this.sums = device.createBuffer({
        label: 'Scene light totals',
        size: sumBytes * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.readback = device.createBuffer({
        label: 'Scene light readback',
        size: sumBytes * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    const data = new Float32Array(anchorBytes / 4);
    data[0] = anchors.length;
    anchors.forEach((anchor, i) => data.set(anchor.position, (i + 1) * 4));
    device.queue.writeBuffer(this.anchors, 0, data);
    const resources: Record<number, GPUBindingResource> = {
      0: { buffer: request.params },
      1: simulation.field.createView(),
      8: simulation.smoke.createView(),
      2: request.sampler,
      11: request.lut.createView(),
      12: { buffer: bricks.pages },
      13: { buffer: lighting.tiles },
      14: { buffer: bricks.active },
      15: { buffer: this.anchors },
      16: { buffer: this.sums },
    };
    const group = device.createBindGroup({
      label: 'Scene lights',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: SCENE_LIGHT_KERNELS.collectLights.map((binding) => ({
        binding,
        resource: resources[binding],
      })),
    });
    const encoder = device.createCommandEncoder({ label: 'Collect scene light' });
    encoder.clearBuffer(this.sums, 0, sumBytes);
    const pass = beginComputePass(device, encoder, { label: 'Scene lights' }, 'lighting');
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(bricks.dispatch, bricks.activeOffset);
    pass.end();
    const readback = this.readback!;
    encoder.copyBufferToBuffer(this.sums, 0, readback, 0, sumBytes);
    device.queue.submit([encoder.finish()]);
    this.pending = true;
    const generation = this.generation;
    readback
      .mapAsync(GPUMapMode.READ, 0, sumBytes)
      .then(() => {
        const sums = new Float32Array(readback.getMappedRange(0, sumBytes).slice(0));
        readback.unmap();
        if (!this.disposed && generation === this.generation) this.apply(request, sums);
      })
      .catch(() => {
        // Device loss or disposal can reject an in-flight readback; leave no stale light.
        if (!this.disposed) this.reset();
      })
      .finally(() => {
        this.pending = false;
        const deferred = this.deferred;
        this.deferred = undefined;
        if (deferred && !this.disposed) this.sample(deferred);
      });
  }
  /** Measure each anchor's light from its totals, then light the scene. */
  private apply(request: Request, sums: Float32Array): void {
    const { anchors, intensity } = request;
    // Each sample stands for a cube a quarter of a brick wide (scene-lights.wgsl).
    const sample = (request.simulation.brickCells / 4) * request.voxelSize;
    const live = new Set<string>();
    anchors.forEach((anchor, i) => {
      live.add(anchor.id);
      const [mass, x, y, z, r, g, b] = sums.subarray(i * SUM_FLOATS, i * SUM_FLOATS + 7);
      let measured = this.measured.get(anchor.id);
      if (!measured) {
        measured = {
          position: new Vector3(),
          color: new Color(),
          intensity: 0,
          transient: anchor.transient,
          peak: 0,
          readings: 0,
        };
        this.measured.set(anchor.id, measured);
      }
      const energy = Math.max(0, mass) * sample ** 3;
      measured.position.set(...anchor.position);
      if (mass > 0) measured.position.add(new Vector3(x, y, z).divideScalar(mass));
      const peak = Math.max(r, g, b, 1e-8);
      measured.color.setRGB(r / peak, g / peak, b / peak);
      measured.intensity = Math.min(400, energy * 2 * intensity);
      measured.peak = Math.max(measured.peak, measured.intensity);
      measured.readings++;
    });
    for (const [id, measured] of this.measured)
      if (!live.has(id)) {
        this.release(measured);
        this.measured.delete(id);
      }
    this.place();
  }
  /** Give the measured lights to point lights: one each with ClusteredLighting, otherwise
   * the fixed few to the anchors that light the camera's surroundings most. */
  private place(): void {
    const lit = [...this.measured.values()].filter((measured) => measured.intensity > 0);
    if (this.clustered) {
      for (const light of this.fixed.splice(0)) this.remove(light);
      for (const measured of this.measured.values()) {
        if (measured.intensity <= 0) {
          this.release(measured);
          continue;
        }
        measured.light ??= this.add();
        this.copy(measured, measured.light);
      }
      return;
    }
    for (const measured of this.measured.values()) this.release(measured);
    while (this.fixed.length < FIXED_LIGHTS) this.fixed.push(this.add());
    const brightest = lit
      .map((measured) => ({
        measured,
        light:
          measured.intensity / Math.max(1e-6, measured.position.distanceToSquared(this.viewer)),
      }))
      .sort((a, b) => b.light - a.light);
    this.fixed.forEach((light, i) => {
      const entry = brightest[i];
      if (entry) this.copy(entry.measured, light);
      else light.intensity = 0;
    });
  }
  private copy(measured: Measured, light: PointLight): void {
    light.position.copy(measured.position);
    light.color.copy(measured.color);
    light.intensity = measured.intensity;
    light.distance = Math.sqrt(measured.intensity / LIGHT_CUTOFF);
  }
  private add(): PointLight {
    const light = new PointLight(0xffffff, 0);
    light.name = 'Fire Pro flame light';
    light.castShadow = false;
    light.userData.fireProOwnedLight = true;
    this.parent.add(light);
    return light;
  }
  private remove(light: PointLight): void {
    light.removeFromParent();
    light.dispose();
  }
  /** Remove an anchor's own light. */
  private release(measured: Measured): void {
    if (!measured.light) return;
    this.remove(measured.light);
    measured.light = undefined;
  }
  /** Turn every light off, and drop measurements in flight. */
  reset(): void {
    this.generation++;
    this.deferred = undefined;
    for (const measured of this.measured.values()) this.release(measured);
    this.measured.clear();
    for (const light of this.fixed) light.intensity = 0;
  }
  dispose(): void {
    if (this.disposed) return;
    this.reset();
    this.disposed = true;
    for (const light of this.fixed.splice(0)) this.remove(light);
    this.anchors?.destroy();
    this.sums?.destroy();
    this.readback?.destroy();
  }
}
