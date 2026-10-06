import commonSource from './shaders/volume-common.wgsl?raw';
import emissionSource from './shaders/fire-emission.wgsl?raw';
import bricksSource from './shaders/volume-bricks.wgsl?raw';
import listSource from './shaders/bricks.wgsl?raw';
import poolSource from './shaders/pool.wgsl?raw';
import lightingSource from './shaders/lighting.wgsl?raw';
import type { BrickBox, FluidSimulation } from './FluidSimulation.ts';
import { LIGHTING_KERNELS, renderPipeline } from './renderKernels.ts';
import type { LightingDivisor, RenderSettings } from './types.ts';
import { beginComputePass } from './GpuProfiler.ts';

/** Boxes per row of the box texture, two texels each (volume-bricks.wgsl). */
const BOX_ROW = 256;
/** Entries per row of the page table texture (volume-bricks.wgsl FIRE_PRO_PAGE_ROW). */
const PAGE_ROW = 4096;
/** Workgroups per row of a dispatch over every brick id. */
const DISPATCH_ROW = 65535;

/** What the renderer reads of a simulation. */
export type RenderSource = Pick<
  FluidSimulation,
  'field' | 'smoke' | 'renderScratch' | 'renderSmokeScratch' | 'steps' | 'brickCells' | 'renderBricks'
>;

/**
 * The renderer's camera-independent state: the boxes it marches and their page tables,
 * and in every brick slot the flags of its 8-cell blocks and its incident light
 * at the selected resolution (lighting.wgsl). Slots keep both as long as they keep their bricks.
 */
export class VolumeLighting {
  /** The boxes (rgba32float, `BOX_ROW` a row, two texels each), and their page tables
   * (r32float, `PAGE_ROW` entries a row). */
  boxes!: GPUTexture;
  pages!: GPUTexture;
  /** Every tile slot's coordinates and box (RenderBricks.tiles). */
  tiles!: GPUBuffer;
  /** Flags of every slot's 8-cell blocks (rg16float), and its incident light with an
   * apron (rgba16float), laid out slot by slot like the pools. */
  occupancy!: GPUTexture;
  light!: GPUTexture;
  /** Fine heat/flame/fuel. Full-resolution smoke uses a smoothed RGBA scratch copy;
   * coarse smoke keeps the original fine fields and borrows its own R16 predictor.
   * The simulation owns both borrowed textures. */
  fields!: GPUTexture;
  smoke!: GPUTexture;
  /** The boxes as last uploaded, in bricks, and the version of the simulation's bricks
   * they came with (RenderBricks). */
  boxList: readonly BrickBox[] = [];
  version = -1;
  private pipelines!: Record<string, GPUComputePipeline>;
  /** The boxes changed since the last `encode`: their page tables are empty, and bricks
   * that were outside every box need their occupancy and light. */
  private stale = true;
  private lastStep = -1;
  private lastLightStep = -1;
  /** The step the occupancy was last built for, or -1 when it is out of date. */
  private occupancyStep = -1;
  private lastSettings = '';
  /** Bind groups for each field pool, which alternate with their scratch. */
  private groups = new Map<GPUTexture, Map<GPUTexture, Record<string, GPUBindGroup>>>();

  private constructor(
    private device: GPUDevice,
    private uniform: GPUBuffer,
    private sampler: GPUSampler,
    private blackbodyTexture: GPUTexture,
  ) {}

  static async create(
    device: GPUDevice,
    uniform: GPUBuffer,
    sampler: GPUSampler,
    blackbodyTexture: GPUTexture,
  ): Promise<VolumeLighting> {
    const cache = new VolumeLighting(device, uniform, sampler, blackbodyTexture);
    const module = device.createShaderModule({
      label: 'Volume lighting and occupancy',
      code: [
        emissionSource,
        poolSource,
        bricksSource,
        listSource,
        commonSource,
        lightingSource,
      ].join('\n'),
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === 'error');
    if (errors.length)
      throw new Error(
        errors.map((message) => `Lighting WGSL ${message.lineNum}: ${message.message}`).join('\n'),
      );
    const names = Object.keys(LIGHTING_KERNELS);
    const pipelines = await Promise.all(
      names.map((name) => renderPipeline(device, module, name, LIGHTING_KERNELS[name])),
    );
    cache.pipelines = Object.fromEntries(names.map((name, i) => [name, pipelines[i]]));
    return cache;
  }

  invalidate(): void {
    this.lastSettings = '';
  }

  /** Bytes of the native textures/buffer this cache owns, from their actual sizes.
   * The simulation accounts for the borrowed predictor textures; the caller owns the
   * uniform and blackbody table. Driver padding is not exposed by WebGPU. */
  get memoryBytes(): number {
    const bytes = (texture: GPUTexture | undefined, texel: number) =>
      texture ? texture.width * texture.height * texture.depthOrArrayLayers * texel : 0;
    return (
      bytes(this.boxes, 16) +
      bytes(this.pages, 4) +
      bytes(this.occupancy, 4) +
      bytes(this.light, 8) +
      (this.tiles?.size ?? 0)
    );
  }

  /**
   * Upload the simulation's boxes, empty page tables for them and its tile table, after
   * they changed. The next `encode` fills the page tables. True when they changed.
   */
  sync(simulation: RenderSource): boolean {
    const bricks = simulation.renderBricks;
    if (bricks.version === this.version) return false;
    this.version = bricks.version;
    this.boxList = bricks.boxes;
    const device = this.device;
    const rows = Math.max(1, Math.ceil(bricks.boxes.length / BOX_ROW));
    if (this.boxes?.height !== rows) {
      this.boxes?.destroy();
      this.boxes = device.createTexture({
        label: 'Render boxes',
        size: [BOX_ROW * 2, rows],
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    }
    // Each box: its first brick and first page table entry, then its bricks per axis.
    const data = new Float32Array(BOX_ROW * 8 * rows);
    let entries = 0;
    bricks.boxes.forEach((box, i) => {
      const size = box.hi.map((n, axis) => n - box.lo[axis] + 1);
      data.set([...box.lo, entries, ...size, 0], i * 8);
      entries += size[0] * size[1] * size[2];
    });
    device.queue.writeTexture({ texture: this.boxes }, data, { bytesPerRow: BOX_ROW * 32 }, [
      BOX_ROW * 2,
      rows,
    ]);
    const pageRows = Math.max(1, Math.ceil(entries / PAGE_ROW));
    if (pageRows > device.limits.maxTextureDimension2D)
      throw new Error('The simulation spans more bricks than the renderer can address.');
    if (!this.pages || this.pages.height < pageRows) {
      this.pages?.destroy();
      this.pages = device.createTexture({
        label: 'Render page tables',
        size: [PAGE_ROW, pageRows],
        format: 'r32float',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_DST,
      });
    }
    // Bricks outside every tile have no slot.
    device.queue.writeTexture(
      { texture: this.pages },
      new Float32Array(PAGE_ROW * pageRows).fill(-1),
      { bytesPerRow: PAGE_ROW * 4 },
      [PAGE_ROW, pageRows],
    );
    if (!this.tiles || this.tiles.size < bricks.tiles.byteLength) {
      this.tiles?.destroy();
      this.tiles = device.createBuffer({
        label: 'Render tiles',
        size: Math.max(16, bricks.tiles.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this.tiles, 0, bricks.tiles);
    this.groups.clear();
    this.stale = true;
    return true;
  }

  /** Size the rendering pools to the field pool's slots; true when their layout changed. */
  private sizePools(simulation: RenderSource, lightingDivisor: LightingDivisor): boolean {
    const { field, brickCells } = simulation;
    const edge = brickCells + 1;
    const slots = [field.width, field.height, field.depthOrArrayLayers].map((n) => n / edge);
    const make = (label: string, texel: number, format: GPUTextureFormat) =>
      this.device.createTexture({
        label,
        dimension: '3d',
        size: slots.map((n) => n * texel),
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
    const fits = (texture: GPUTexture | undefined, texel: number) =>
      texture?.width === slots[0] * texel &&
      texture.height === slots[1] * texel &&
      texture.depthOrArrayLayers === slots[2] * texel;
    let replaced = false;
    const coarse = simulation.smoke !== field;
    const fields = coarse ? field : simulation.renderScratch;
    const smoke = coarse ? simulation.renderSmokeScratch : fields;
    if (this.fields !== fields || this.smoke !== smoke) {
      // Transport finishes before rendering on the same queue. Predictors are dead
      // after evolution; their interiors and absent-neighbor aprons are restored
      // before corrected transport reads them on the next timestep.
      // The raw fine fields alternate every coarse-smoke step. That changes the
      // bind group, not the layout or the two-step lighting update cadence.
      const sameSize = (a: GPUTexture | undefined, b: GPUTexture) =>
        a?.width === b.width &&
        a.height === b.height &&
        a.depthOrArrayLayers === b.depthOrArrayLayers;
      replaced =
        !sameSize(this.fields, fields) ||
        !sameSize(this.smoke, smoke) ||
        (this.fields === this.smoke) !== (fields === smoke);
      if (this.smoke !== smoke) this.groups.clear();
      this.fields = fields;
      this.smoke = smoke;
    }
    const blocks = brickCells / 8;
    if (!fits(this.occupancy, blocks)) {
      this.occupancy?.destroy();
      this.occupancy = make('Brick block flags', blocks, 'rg16float');
      replaced = true;
    }
    const light = brickCells / lightingDivisor + 1;
    if (!fits(this.light, light)) {
      this.light?.destroy();
      this.light = make('Incident volume lighting', light, 'rgba16float');
      replaced = true;
    }
    if (replaced) this.groups.clear();
    return replaced;
  }

  /**
   * Bring the page tables, the occupancy and the lighting up to date with the simulation's
   * latest step: the page tables and occupancy every step. Incident light is smooth in
   * time, so it is built every second step, the slots given out in between at once; new
   * settings, boxes or pools build it whole at once. True when it encoded work.
   */
  encode(
    encoder: GPUCommandEncoder,
    simulation: RenderSource,
    settings: RenderSettings,
    sceneKey = '',
    lightingDivisor: LightingDivisor = 4,
  ): boolean {
    this.sync(simulation);
    const resized = this.sizePools(simulation, lightingDivisor);
    const changed = this.stale;
    const bricks = simulation.renderBricks;
    const steps = simulation.steps;
    const settingsKey = [
      settings.smokeDensity,
      settings.shadowDensity,
      settings.fireIntensity,
      settings.flameOpacity,
      settings.temperature,
      ...settings.fireColor,
      ...settings.smokeColor,
      settings.sootGlow,
      sceneKey,
    ].join(',');
    const stepped = steps !== this.lastStep;
    const rebuild = changed || resized;
    const smoothSmoke = stepped || rebuild;
    const exportPages = stepped || changed;
    const buildOccupancy = this.occupancyStep !== steps || rebuild;
    const lightDue = steps < this.lastLightStep || steps - this.lastLightStep >= 2;
    const buildLight = (stepped && lightDue) || rebuild || settingsKey !== this.lastSettings;
    const lightFresh = stepped && !buildLight;
    if (
      !bricks.boxes.length ||
      (!smoothSmoke && !exportPages && !buildOccupancy && !buildLight && !lightFresh)
    )
      return false;
    const coarse = simulation.smoke !== simulation.field;
    let bySmoke = this.groups.get(simulation.field);
    if (!bySmoke) {
      bySmoke = new Map();
      this.groups.set(simulation.field, bySmoke);
    }
    let groups = bySmoke.get(simulation.smoke);
    if (!groups) {
      const resources: Record<number, GPUBindingResource> = {
        0: { buffer: this.uniform },
        1: simulation.field.createView(),
        2: this.sampler,
        3: this.light.createView(),
        4: this.occupancy.createView(),
        5: this.pages.createView(),
        6: { buffer: bricks.zeroed },
        7: this.fields.createView(),
        8: simulation.smoke.createView(),
        9: this.pages.createView(),
        10: this.boxes.createView(),
        11: this.blackbodyTexture.createView(),
        12: { buffer: bricks.pages },
        13: { buffer: this.tiles },
        14: { buffer: bricks.active },
        17: this.smoke.createView(),
      };
      groups = Object.fromEntries(
        Object.entries(LIGHTING_KERNELS)
          .filter(([name]) => name !== (coarse ? 'smoothSmoke' : 'smoothCoarseSmoke'))
          .map(([name, bindings]) => [
            name,
            this.device.createBindGroup({
              label: name,
              layout: this.pipelines[name].getBindGroupLayout(0),
              entries: bindings.map((binding) => ({
                binding,
                resource:
                  binding === 1 && !name.startsWith('smooth')
                    ? this.fields.createView()
                    : binding === 8 && !name.startsWith('smooth')
                      ? this.smoke.createView()
                      : resources[binding],
              })),
            }),
          ]),
      );
      bySmoke.set(simulation.smoke, groups);
    }
    const pass = beginComputePass(
      this.device,
      encoder,
      { label: 'Volume lighting and occupancy' },
      'lighting',
    );
    const run = (name: string) => {
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, groups[name]);
    };
    if (exportPages) {
      run('exportPages');
      const workgroups = Math.ceil(bricks.pages.size / 4 / 64);
      pass.dispatchWorkgroups(
        Math.min(workgroups, DISPATCH_ROW),
        Math.ceil(workgroups / DISPATCH_ROW),
      );
    }
    if (smoothSmoke) {
      run(coarse ? 'smoothCoarseSmoke' : 'smoothSmoke');
      pass.dispatchWorkgroupsIndirect(
        bricks.dispatch, coarse ? bricks.smokeOffset : bricks.fieldOffset,
      );
    }
    if (buildOccupancy) {
      run('buildOccupancy');
      pass.dispatchWorkgroupsIndirect(bricks.dispatch, bricks.activeOffset);
      this.occupancyStep = steps;
    }
    if (buildLight) {
      run('buildLighting');
      pass.dispatchWorkgroupsIndirect(bricks.dispatch, bricks.activeOffset);
      this.lastLightStep = steps;
      this.lastSettings = settingsKey;
    } else if (lightFresh) {
      run('lightFresh');
      pass.dispatchWorkgroupsIndirect(bricks.dispatch, bricks.zeroedOffset);
    }
    pass.end();
    this.lastStep = steps;
    this.stale = false;
    return true;
  }

  dispose(): void {
    this.boxes?.destroy();
    this.pages?.destroy();
    this.tiles?.destroy();
    this.light?.destroy();
    this.occupancy?.destroy();
    this.groups.clear();
  }
}
