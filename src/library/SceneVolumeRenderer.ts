import { BufferUploads } from '../engine/BufferUploads.ts';
import {
  BackSide,
  BoxGeometry,
  Camera,
  Color,
  CustomBlending,
  DepthTexture,
  ExternalTexture,
  HalfFloatType,
  FloatType,
  NearestFilter,
  RedFormat,
  LinearFilter,
  Matrix4,
  Mesh,
  NodeMaterial,
  Object3D,
  OneFactor,
  OneMinusSrcAlphaFactor,
  Texture,
  Vector2,
  Vector3,
  Vector4,
  WebGPURenderer,
  type Data3DTexture,
  type DirectionalLight,
  type AmbientLight,
  type Node,
  type Scene,
} from 'three/webgpu';
import {
  sampler,
  screenUV,
  texture,
  texture3D,
  uniform,
  viewportDepthTexture,
  wgsl,
  wgslFn,
} from 'three/tsl';
import { FluidSimulation } from '../engine/FluidSimulation.ts';
import { VolumeLighting } from '../engine/VolumeLighting.ts';
import { SLICE_PARAMS, VolumeSlices } from '../engine/VolumeSlices.ts';
import { createBlackbodyTexture } from '../engine/BlackbodyLut.ts';
import { SHADOW_STEPS, type RenderSettings } from '../engine/types.ts';
import { SceneLights, type LightAnchor } from './SceneLights.ts';
import {
  DEBUG_FIELDS,
  DEBUG_SCALES,
  type DebugField,
  type ResolvedSimulationOptions,
  type Vec3,
} from './options.ts';
import source from '../engine/shaders/scene-volume.wgsl?raw';
import emissionSource from '../engine/shaders/fire-emission.wgsl?raw';
import reconstructionSource from '../engine/shaders/volume-reconstruction.wgsl?raw';
import marchSource from '../engine/shaders/volume-march.wgsl?raw';
import bricksSource from '../engine/shaders/volume-bricks.wgsl?raw';
import poolSource from '../engine/shaders/pool.wgsl?raw';

const fireEmission = wgsl(emissionSource);
const shadeVolume = wgslFn(source.slice(source.indexOf('fn fireProSceneVolume')), [
  fireEmission,
  wgsl(poolSource),
  wgsl(bricksSource),
  wgsl(reconstructionSource),
  wgsl(marchSource),
]);

/** Draws into the application's render pipeline with premultiplied volume radiance. */
export class SceneVolumeRenderer extends Mesh<BoxGeometry, NodeMaterial> {
  private readonly params: GPUBuffer;
  private readonly values = new Float32Array(20);
  private readonly gpuSampler: GPUSampler;
  private readonly blackbody: GPUTexture;
  private lighting!: VolumeLighting;
  private slicer!: VolumeSlices;
  /** Placeholder bound while the half-resolution march is off. */
  private readonly emptySlices: GPUTexture;
  private readonly sliceParams = new Float32Array(SLICE_PARAMS);
  private uploads?: BufferUploads;
  private sceneLights?: SceneLights;
  private readonly wrappers = new Map<GPUTexture, ExternalTexture>();
  // TSL's declarations name Data3DTexture; runtime values are 3D ExternalTexture
  // wrappers, assigned before rendering.
  private readonly field = texture3D(null as unknown as Data3DTexture);
  /** The boxes' page tables and the boxes (VolumeLighting). */
  private readonly pages = texture(null as unknown as Texture);
  private readonly boxes = texture(null as unknown as Texture);
  private readonly smokeField = texture3D(null as unknown as Data3DTexture);
  private readonly smokeDivisor = uniform(1);
  private readonly velocityDivisor = uniform(1);
  private readonly velocity = texture3D(null as unknown as Data3DTexture);
  /** The expansion, or the solver field the simulation copies out for a debug view. */
  private readonly solverField = texture3D(null as unknown as Data3DTexture);
  private debugField: DebugField = 'beauty';
  private readonly incident = texture3D(null as unknown as Data3DTexture);
  private readonly occupancy = texture3D(null as unknown as Data3DTexture);
  private readonly slices = texture3D(null as unknown as Data3DTexture);
  private readonly halfResolution = uniform(0);
  /** Voxel size, log2 of the field brick size, and the number of boxes. */
  private readonly grid = uniform(new Vector4());
  private readonly inverseVP = uniform(new Matrix4());
  private readonly optical = uniform(new Vector4());
  private readonly light = uniform(new Vector4());
  private readonly smoke = uniform(new Vector4());
  private readonly fire = uniform(new Vector4());
  private readonly sootGlow = uniform(0.25);
  private readonly sampling = uniform(new Vector2());
  private readonly viewport = uniform(new Vector2());
  private readonly sceneLight = uniform(new Vector4());
  private readonly ambient = uniform(new Vector4());
  private readonly displayField = uniform(0);
  /** The debug view's heat map: its range, and 1 when it is signed (DEBUG_SCALES). */
  private readonly debugScale = uniform(new Vector2(1, 0));
  private readonly reconstructionFilter = uniform(2);
  private readonly depth = new DepthTexture();
  private readonly direction = new Vector3();
  private disposed = false;
  private lastLightStep = -1;
  /** The field pool of the previous frame: the field and its scratch alternate. */
  private previousField?: GPUTexture;
  private previousSmoke?: GPUTexture;
  /** The version of the boxes the mesh was fitted to. */
  private fitted = -1;

  private constructor(
    private readonly owner: WebGPURenderer,
    private readonly simulation: FluidSimulation,
    private readonly options: () => ResolvedSimulationOptions,
    private readonly anchors: () => readonly LightAnchor[],
  ) {
    // The mesh only rasterizes the pixels the boxes cover: rays run in world space.
    const geometry = new BoxGeometry(1, 1, 1);
    const material = new NodeMaterial();
    Object.assign(material, {
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: BackSide,
      blending: CustomBlending,
      blendSrc: OneFactor,
      blendDst: OneMinusSrcAlphaFactor,
      blendSrcAlpha: OneFactor,
      blendDstAlpha: OneMinusSrcAlphaFactor,
    });
    super(geometry, material);
    this.name = 'Fire Pro volume rendering';
    this.visible = false;
    const device = simulation.device;
    this.params = device.createBuffer({
      label: 'Scene volume parameters',
      size: this.values.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gpuSampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    this.blackbody = createBlackbodyTexture(device);
    this.emptySlices = device.createTexture({
      label: 'Empty depth slices',
      size: [1, 1, 1],
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    material.fragmentNode = shadeVolume({
      fields: this.field,
      smokeField: this.smokeField,
      smokeDivisor: this.smokeDivisor,
      velocityDivisor: this.velocityDivisor,
      pages: this.pages,
      boxes: this.boxes,
      grid: this.grid,
      reconstructionFilter: this.reconstructionFilter,
      velocityField: this.velocity,
      solverField: this.solverField,
      linearSampler: sampler(this.field),
      lighting: this.incident,
      occupancy: this.occupancy,
      lut: texture(this.wrap(this.blackbody)),
      slices: this.slices,
      halfResolution: this.halfResolution,
      inverseViewProjection: this.inverseVP,
      optical: this.optical,
      light: this.light,
      smoke: this.smoke,
      fire: this.fire,
      sootGlow: this.sootGlow,
      sampling: this.sampling,
      viewport: this.viewport,
      sceneDepth: (
        viewportDepthTexture as unknown as (uv: Node, level: null, depth: DepthTexture) => Node
      )(screenUV, null, this.depth),
      uv: screenUV,
      sceneLight: this.sceneLight,
      ambientLight: this.ambient,
      displayField: this.displayField,
      debugScale: this.debugScale,
    });
    this.onBeforeRender = (renderer, scene, camera) => {
      if (this.disposed) return;
      if ((renderer as unknown) !== this.owner)
        throw new Error('Render FireSimulation with its initialization renderer.');
      this.prepare(camera, scene);
    };
  }
  /** `anchors` are where flames light the scene from (SceneLights). */
  static async create(
    renderer: WebGPURenderer,
    simulation: FluidSimulation,
    parent: Object3D,
    options: () => ResolvedSimulationOptions,
    anchors: () => readonly LightAnchor[],
    field: DebugField,
  ): Promise<SceneVolumeRenderer> {
    const result = new SceneVolumeRenderer(renderer, simulation, options, anchors);
    try {
      result.lighting = await VolumeLighting.create(
        simulation.device,
        result.params,
        result.gpuSampler,
        result.blackbody,
      );
      result.slicer = await VolumeSlices.create(simulation.device, result.gpuSampler);
      result.sceneLights = await SceneLights.create(simulation.device, parent, renderer);
      result.setDebug(field);
      result.update();
      return result;
    } catch (error) {
      result.dispose();
      throw error;
    }
  }
  /** Fit the mesh around every box, after they changed; hide it without any. */
  private fit(): void {
    this.lighting.sync(this.simulation);
    if (this.lighting.version === this.fitted) return;
    this.fitted = this.lighting.version;
    const boxes = this.lighting.boxList;
    this.visible = boxes.length > 0;
    if (!boxes.length) return;
    const size = this.simulation.brickCells * this.simulation.voxelSize;
    const lo = [0, 1, 2].map((axis) => Math.min(...boxes.map((box) => box.lo[axis])) * size);
    const hi = [0, 1, 2].map((axis) => (Math.max(...boxes.map((box) => box.hi[axis])) + 1) * size);
    this.position.set(...(lo.map((n, axis) => (n + hi[axis]) / 2) as Vec3));
    this.scale.set(...(hi.map((n, axis) => n - lo[axis]) as Vec3));
    this.updateMatrix();
  }
  /** Which field to show. */
  setDebug(field: DebugField): void {
    this.debugField = field;
    this.displayField.value = DEBUG_FIELDS.indexOf(field);
    if (field !== 'beauty') {
      const { range, signed } = DEBUG_SCALES[field];
      this.debugScale.value.set(range, Number(signed));
    }
  }
  invalidate(): void {
    this.lighting.invalidate();
    this.lastLightStep = -1;
  }
  reset(): void {
    this.sceneLights?.reset();
    this.invalidate();
  }
  private wrap(source: GPUTexture): ExternalTexture {
    let wrapper = this.wrappers.get(source);
    if (!wrapper) {
      // 32-bit floats are read without filtering.
      const float = source.format === 'r32float' || source.format === 'rgba32float';
      wrapper = new ExternalTexture(source);
      Object.assign(wrapper, {
        is3DTexture: source.dimension === '3d',
        image: { width: source.width, height: source.height, depth: source.depthOrArrayLayers },
        type: float ? FloatType : HalfFloatType,
        ...(['r32float', 'r16float'].includes(source.format) ? { format: RedFormat } : {}),
        minFilter: float ? NearestFilter : LinearFilter,
        magFilter: float ? NearestFilter : LinearFilter,
        generateMipmaps: false,
      });
      wrapper.needsUpdate = true;
      this.wrappers.set(source, wrapper);
    }
    return wrapper;
  }
  private renderSettings(): RenderSettings {
    const { flame, smoke } = this.options();
    return {
      fireIntensity: flame.brightness,
      flameOpacity: flame.opacity,
      smokeDensity: smoke.density,
      scattering: smoke.scattering,
      shadowDensity: smoke.shadowDensity,
      temperature: flame.temperature,
      smokeColor: new Color(smoke.color).toArray() as Vec3,
      fireColor: new Color(flame.color).toArray() as Vec3,
      sootGlow: flame.sootGlow,
    };
  }
  private readSceneLighting(scene?: Scene): void {
    this.direction.set(0, 1, 0);
    this.sceneLight.value.set(0, 0, 0, 0);
    this.ambient.value.set(0, 0, 0, 0);
    let brightest = 0;
    scene?.traverseVisible((object) => {
      if (object.userData.fireProOwnedLight) return;
      if ('isAmbientLight' in object) {
        const light = object as AmbientLight;
        this.ambient.value.x += light.color.r * light.intensity;
        this.ambient.value.y += light.color.g * light.intensity;
        this.ambient.value.z += light.color.b * light.intensity;
      }
      if ('isDirectionalLight' in object) {
        const light = object as DirectionalLight;
        if (light.intensity <= brightest) return;
        light.updateWorldMatrix(true, false);
        light.target.updateWorldMatrix(true, false);
        this.direction
          .setFromMatrixPosition(light.matrixWorld)
          .sub(new Vector3().setFromMatrixPosition(light.target.matrixWorld));
        if (this.direction.lengthSq() < 1e-12) this.direction.set(0, 1, 0);
        brightest = light.intensity;
        this.sceneLight.value.set(light.color.r, light.color.g, light.color.b, 0);
      }
    });
    this.light.value.set(this.direction.x, this.direction.y, this.direction.z, brightest);
  }
  /** The compute passes' parameters (volume-common.wgsl RenderParams). */
  private writeParameters(settings: RenderSettings): void {
    const simulation = this.simulation;
    this.values.set([
      simulation.voxelSize,
      simulation.brickShift,
      SHADOW_STEPS,
      settings.flameOpacity,
      settings.smokeDensity,
      settings.fireIntensity,
      settings.shadowDensity,
      settings.scattering,
      ...this.light.value.toArray(),
      ...settings.smokeColor,
      settings.sootGlow,
      ...settings.fireColor,
      settings.temperature,
    ]);
    this.uploads ??= new BufferUploads(simulation.device.queue);
    this.uploads.write(this.params, this.values);
  }
  /** Keep the mesh around the boxes, and light the scene from the latest step, whether or
   * not the camera sees the simulation. */
  update(): void {
    if (this.disposed || !this.sceneLights) return;
    this.slicer.beginFrame(this.owner.info.frame, this.options().rendering.halfResolution);
    this.fit();
    const simulation = this.simulation;
    const lighting = this.options().lighting;
    if (!lighting.illuminateScene) {
      this.sceneLights.reset();
      this.lastLightStep = -1;
      return;
    }
    if (simulation.steps === this.lastLightStep) return;
    this.writeParameters(this.renderSettings());
    this.sceneLights.sample({
      simulation,
      lighting: this.lighting,
      params: this.params,
      sampler: this.gpuSampler,
      lut: this.blackbody,
      anchors: this.anchors(),
      intensity: lighting.intensity,
      voxelSize: simulation.voxelSize,
    });
    this.lastLightStep = simulation.steps;
  }
  /** Whether an explosion's light has died down (SceneLights). */
  lightFinished(id: string): boolean {
    return this.sceneLights?.finished(id) ?? false;
  }
  /** Native renderer allocations, including lighting, slice targets and readback.
   * Borrowed solver scratch is counted by the simulation. Three.js-managed scene
   * targets, geometry and backend bookkeeping are outside this total. */
  get memoryBytes(): number {
    const textureBytes = (texture: GPUTexture) =>
      texture.width * texture.height * texture.depthOrArrayLayers * 8;
    return (
      this.params.size +
      textureBytes(this.blackbody) +
      textureBytes(this.emptySlices) +
      (this.lighting?.memoryBytes ?? 0) +
      (this.slicer?.memoryBytes ?? 0) +
      (this.sceneLights?.memoryBytes ?? 0)
    );
  }
  private prepare(camera?: Camera, scene?: Scene): void {
    const simulation = this.simulation;
    const settings = this.renderSettings();
    const { filter, raySteps, halfResolution, lightingDivisor } = this.options().rendering;
    this.slicer.beginFrame(this.owner.info.frame, halfResolution);
    this.reconstructionFilter.value = ['trilinear', 'quadratic', 'cubic'].indexOf(filter);
    this.fit();
    this.readSceneLighting(scene);
    this.writeParameters(settings);
    const encoder = simulation.device.createCommandEncoder({ label: 'Prepare volume lighting' });
    // Only the light direction and whether shadows are used affect the cache.
    // Ambient/direct colors are applied by the camera shader.
    const sceneKey = [
      this.light.value.x,
      this.light.value.y,
      this.light.value.z,
      Number(this.light.value.w > 0),
    ].join(',');
    if (this.lighting.encode(encoder, simulation, settings, sceneKey, lightingDivisor))
      simulation.device.queue.submit([encoder.finish()]);
    const lighting = this.lighting;
    // Debug views show the solver's actual values; beauty uses rendering-only smoke.
    this.field.value = this.wrap(
      this.debugField === 'beauty' ? lighting.fields : simulation.field,
    ) as unknown as Data3DTexture;
    this.smokeField.value = this.wrap(
      this.debugField === 'beauty' ? lighting.smoke : simulation.smoke,
    ) as unknown as Data3DTexture;
    this.smokeDivisor.value = simulation.smokeDivisor;
    this.velocityDivisor.value = simulation.velocityDivisor;
    this.pages.value = this.wrap(lighting.pages);
    this.boxes.value = this.wrap(lighting.boxes);
    this.velocity.value = this.wrap(simulation.velocity) as unknown as Data3DTexture;
    const solverField =
      this.debugField === 'expansion' ? simulation.expansionRate : simulation.solverFieldTexture;
    this.solverField.value = this.wrap(solverField) as unknown as Data3DTexture;
    this.incident.value = this.wrap(lighting.light) as unknown as Data3DTexture;
    this.occupancy.value = this.wrap(lighting.occupancy) as unknown as Data3DTexture;
    this.grid.value.set(
      simulation.voxelSize,
      simulation.brickShift,
      lighting.boxList.length,
      0,
    );
    this.optical.value.set(
      settings.smokeDensity,
      settings.fireIntensity,
      settings.shadowDensity,
      settings.scattering,
    );
    // Camera/slice smoke.w carries flame opacity; soot glow has its own uniform.
    this.smoke.value.set(...settings.smokeColor, settings.flameOpacity);
    this.fire.value.set(...settings.fireColor, settings.temperature);
    this.sootGlow.value = settings.sootGlow;
    this.sampling.value.set(raySteps, simulation.time);
    if (camera) {
      this.inverseVP.value.copy(camera.matrixWorld).multiply(camera.projectionMatrixInverse);
      this.owner.getDrawingBufferSize(this.viewport.value);
      this.sceneLights?.viewer.setFromMatrixPosition(camera.matrixWorld);
    }
    let slices = this.emptySlices;
    // Size the slices to this render's target. Targets already rendered at reduced
    // resolution, such as reflections, keep the direct march.
    const target = this.owner.getRenderTarget();
    const size = target ? new Vector2(target.width, target.height) : this.viewport.value;
    if (
      halfResolution &&
      camera &&
      size.x * size.y >= this.viewport.value.x * this.viewport.value.y
    ) {
      const p = this.sliceParams;
      p.set(this.inverseVP.value.elements, 0);
      for (const [offset, value] of [
        [16, this.optical.value],
        [20, this.light.value],
        [24, this.smoke.value],
        [28, this.fire.value],
        [32, this.grid.value],
        [36, this.sceneLight.value],
        [40, this.ambient.value],
      ] as const)
        p.set([value.x, value.y, value.z, value.w], offset);
      p.set([raySteps, simulation.time, this.reconstructionFilter.value, settings.sootGlow], 44);
      slices = this.slicer.march({
        frame: this.owner.info.frame,
        width: size.x,
        height: size.y,
        params: p,
        fields: lighting.fields,
        smokeField: lighting.smoke,
        pages: lighting.pages,
        boxes: lighting.boxes,
        lighting: lighting.light,
        occupancy: lighting.occupancy,
        lut: this.blackbody,
      });
    }
    this.halfResolution.value = Number(slices !== this.emptySlices);
    this.slices.value = this.wrap(slices) as unknown as Data3DTexture;
    const live = new Set([
      simulation.field,
      simulation.smoke,
      ...(this.previousSmoke ? [this.previousSmoke] : []),
      ...(this.previousField ? [this.previousField] : []),
      simulation.velocity,
      solverField,
      this.blackbody,
      lighting.pages,
      lighting.boxes,
      lighting.light,
      lighting.fields,
      lighting.smoke,
      lighting.occupancy,
      this.emptySlices,
      ...this.slicer.textures(),
    ]);
    for (const [gpu, wrapper] of this.wrappers)
      if (!live.has(gpu)) {
        wrapper.dispose();
        this.wrappers.delete(gpu);
      }
    this.previousField = simulation.field;
    this.previousSmoke = simulation.smoke;
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
    this.depth.dispose();
    for (const wrapper of this.wrappers.values()) wrapper.dispose();
    this.wrappers.clear();
    this.sceneLights?.dispose();
    this.lighting?.dispose();
    this.slicer?.dispose();
    this.blackbody.destroy();
    this.emptySlices.destroy();
    this.params.destroy();
  }
}
