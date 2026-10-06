import source from './shaders/curl-noise.wgsl?raw';
import { beginComputePass } from './GpuProfiler.ts';

/** Periodic analytic curl field, generated once instead of hashing every live voxel. */
export async function createCurlNoise(
  device: GPUDevice,
): Promise<{ texture: GPUTexture; sampler: GPUSampler }> {
  const module = device.createShaderModule({ label: 'Periodic curl noise', code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length)
    throw new Error(
      errors.map((message) => `Noise WGSL ${message.lineNum}: ${message.message}`).join('\n'),
    );
  const pipeline = await device.createComputePipelineAsync({
    label: 'Generate curl noise',
    layout: 'auto',
    compute: { module, entryPoint: 'buildNoise' },
  });
  const texture = device.createTexture({
    label: 'Periodic curl noise (2 MiB)',
    dimension: '3d',
    size: [64, 64, 64],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  try {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: texture.createView() }],
    });
    const encoder = device.createCommandEncoder();
    const pass = beginComputePass(device, encoder, { label: 'Generate curl noise' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(8, 16, 16);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const sampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      addressModeW: 'repeat',
    });
    return { texture, sampler };
  } catch (error) {
    texture.destroy();
    throw error;
  }
}
