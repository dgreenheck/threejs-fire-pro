import emissionSource from './shaders/fire-emission.wgsl?raw';
import bricksSource from './shaders/volume-bricks.wgsl?raw';
import poolSource from './shaders/pool.wgsl?raw';
import reconstructionSource from './shaders/volume-reconstruction.wgsl?raw';
import marchSource from './shaders/volume-march.wgsl?raw';
import slicesSource from './shaders/volume-slices.wgsl?raw';
import { beginComputePass } from './GpuProfiler.ts';

/** Depths per pixel at which the half-resolution march stores its progress. */
const DEPTH_SLICES = 8;
/** Floats of the parameters, in SliceParams order (volume-slices.wgsl). */
export const SLICE_PARAMS = 48;

export interface SliceInputs {
  /** Three.js renderer.info.frame; all cameras drawn in one frame share this ID. */
  frame: number;
  /** Render target size in pixels; the slices use half of it on each axis. */
  width: number;
  height: number;
  /** `SLICE_PARAMS` floats in SliceParams order (volume-slices.wgsl). */
  params: Float32Array;
  /** The field pool, and the boxes and their page tables (VolumeLighting). */
  fields: GPUTexture;
  smokeField: GPUTexture;
  pages: GPUTexture;
  boxes: GPUTexture;
  lighting: GPUTexture;
  occupancy: GPUTexture;
  lut: GPUTexture;
}

/** Half-resolution camera march, stored at evenly spaced depths through the volume. */
export class VolumeSlices {
  private pipeline!: GPUComputePipeline;
  private readonly uniform: GPUBuffer;
  /** Current/recent camera sizes coexist; obsolete resize targets are retired. */
  private readonly targets = new Map<string, { texture: GPUTexture; frame: number }>();
  private retireBefore = -Infinity;
  private cleanupScheduled = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly sampler: GPUSampler,
  ) {
    this.uniform = device.createBuffer({
      label: 'Half-resolution volume parameters',
      size: SLICE_PARAMS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  static async create(device: GPUDevice, sampler: GPUSampler): Promise<VolumeSlices> {
    const slices = new VolumeSlices(device, sampler);
    const module = device.createShaderModule({
      label: 'Half-resolution volume march',
      code: [
        emissionSource,
        poolSource,
        bricksSource,
        reconstructionSource,
        marchSource,
        slicesSource,
      ].join('\n'),
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === 'error');
    if (errors.length)
      throw new Error(
        errors.map((message) => `Slices WGSL ${message.lineNum}: ${message.message}`).join('\n'),
      );
    slices.pipeline = await device.createComputePipelineAsync({
      label: 'buildSlices',
      layout: 'auto',
      compute: { module, entryPoint: 'buildSlices' },
    });
    return slices;
  }

  /** Submits the march immediately so it precedes the render that samples it. */
  march(inputs: SliceInputs): GPUTexture {
    this.beginFrame(inputs.frame);
    const width = Math.max(1, Math.ceil(inputs.width / 2));
    const height = Math.max(1, Math.ceil(inputs.height / 2));
    const key = `${width}x${height}`;
    let entry = this.targets.get(key);
    if (!entry) {
      const texture = this.device.createTexture({
        label: `Volume depth slices ${key}`,
        size: [width, height, DEPTH_SLICES],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      entry = { texture, frame: inputs.frame };
      this.targets.set(key, entry);
    }
    entry.frame = inputs.frame;
    const target = entry.texture;
    this.device.queue.writeBuffer(this.uniform, 0, inputs.params);
    const encoder = this.device.createCommandEncoder({ label: 'Half-resolution volume' });
    const pass = beginComputePass(
      this.device,
      encoder,
      { label: 'Half-resolution volume march' },
      'lighting',
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: inputs.fields.createView() },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: inputs.lighting.createView() },
          { binding: 4, resource: inputs.occupancy.createView() },
          { binding: 5, resource: inputs.lut.createView() },
          { binding: 6, resource: target.createView() },
          { binding: 7, resource: inputs.pages.createView() },
          { binding: 8, resource: inputs.boxes.createView() },
          { binding: 9, resource: inputs.smokeField.createView() },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return target;
  }

  /** Retain current and previous frame sizes for multi-camera rendering. Defer
   * destruction until the synchronous Three.js render stack (including reentrant
   * views) has submitted its draws: onBeforeRender runs before that submission. */
  beginFrame(frame: number, enabled = true): void {
    this.retireBefore = enabled ? frame - 1 : Infinity;
    if (this.cleanupScheduled) return;
    this.cleanupScheduled = true;
    queueMicrotask(() => {
      this.cleanupScheduled = false;
      for (const [key, entry] of this.targets)
        if (entry.frame < this.retireBefore) {
          entry.texture.destroy();
          this.targets.delete(key);
        }
    });
  }

  *textures(): IterableIterator<GPUTexture> {
    for (const entry of this.targets.values()) yield entry.texture;
  }

  /** Current native allocation, including retained target sizes after a resize. */
  get memoryBytes(): number {
    let bytes = this.uniform.size;
    for (const { texture } of this.targets.values())
      bytes += texture.width * texture.height * texture.depthOrArrayLayers * 8;
    return bytes;
  }

  dispose(): void {
    for (const { texture } of this.targets.values()) texture.destroy();
    this.targets.clear();
    this.uniform.destroy();
  }
}
