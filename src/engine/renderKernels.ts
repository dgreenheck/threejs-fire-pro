/** The bindings each of the renderer's compute kernels reads (lighting.wgsl,
 * volume-common.wgsl), for their explicit layouts. */
export const LIGHTING_KERNELS: Record<string, readonly number[]> = {
  exportPages: [5, 10, 12, 13],
  smoothSmoke: [0, 1, 2, 7, 8, 9, 10, 12, 13, 14],
  smoothCoarseSmoke: [0, 1, 2, 8, 9, 10, 12, 13, 14, 17],
  buildOccupancy: [0, 1, 4, 8, 9, 10, 12, 13, 14],
  buildLighting: [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14],
  lightFresh: [0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13],
};
/** The scene lights' kernel (scene-lights.wgsl). */
export const SCENE_LIGHT_KERNELS: Record<string, readonly number[]> = {
  collectLights: [0, 1, 2, 8, 11, 12, 13, 14, 15, 16],
};

const READ_ONLY_STORAGE = new Set([6, 12, 13, 14, 15]);
const WRITE_ONLY_TEXTURES: Record<number, [GPUTextureFormat, GPUTextureViewDimension]> = {
  3: ['rgba16float', '3d'],
  4: ['rg16float', '3d'],
  5: ['r32float', '2d'],
  7: ['rgba16float', '3d'],
  17: ['r16float', '3d'],
};

/** The layout entry a render compute binding needs. */
function renderBindingLayout(binding: number): GPUBindGroupLayoutEntry {
  const entry: GPUBindGroupLayoutEntry = { binding, visibility: GPUShaderStage.COMPUTE };
  if (binding === 0) entry.buffer = { type: 'uniform' };
  else if (binding === 16) entry.buffer = { type: 'storage' };
  else if (READ_ONLY_STORAGE.has(binding)) entry.buffer = { type: 'read-only-storage' };
  else if (binding === 2) entry.sampler = { type: 'filtering' };
  else if (binding in WRITE_ONLY_TEXTURES) {
    const [format, viewDimension] = WRITE_ONLY_TEXTURES[binding];
    entry.storageTexture = { access: 'write-only', format, viewDimension };
  } else if (binding === 1 || binding === 8) entry.texture = { viewDimension: '3d' };
  // The 32-bit page table and boxes are read without filtering.
  else if (binding === 9 || binding === 10)
    entry.texture = { viewDimension: '2d', sampleType: 'unfilterable-float' };
  else if (binding === 11) entry.texture = { viewDimension: '2d' };
  else throw new Error(`Unknown render binding ${binding}.`);
  return entry;
}

/** A compute pipeline with an explicit layout of `bindings`. */
export function renderPipeline(
  device: GPUDevice,
  module: GPUShaderModule,
  entryPoint: string,
  bindings: readonly number[],
): Promise<GPUComputePipeline> {
  const layout = device.createBindGroupLayout({
    label: entryPoint,
    entries: bindings.map(renderBindingLayout),
  });
  return device.createComputePipelineAsync({
    label: entryPoint,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint },
  });
}
