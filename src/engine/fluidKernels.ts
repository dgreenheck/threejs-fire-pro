/**
 * The fluid solver's kernels and the bindings each one uses (the fluid modules and the files
 * ShaderSources.ts joins to it). Pipelines use explicit layouts built from these lists.
 */
const LISTED_KERNELS = [
  'advectVelocityWithBounds',
  'correctVelocity',
  'computeCurl',
  'applyForces',
  'computeDivergence',
  'project',
  'evolveFields',
  'advectScalars',
  'exportScalars',
];
/** Kernels that read mesh emission, which tiles list emitters to, and solid cells. */
const MESH_SOURCE_KERNELS = ['applyForces', 'evolveFields'];
const EMITTER_KERNELS = ['applyForces', 'evolveFields'];
const SOLID_KERNELS = [
  'advectVelocityWithBounds',
  'correctVelocity',
  'applyForces',
  'project',
  'evolveFields',
];
/** The tile directory's textures (tiles.wgsl): tile records and their hash, and each
 * brick's entry and record. */
const DIRECTORY = [63, 64, 65, 66];

function kernels(): Record<string, number[]> {
  const used: Record<string, number[]> = {
    // Velocity and scalar advection reuse the packed donor pool (13); correction reads it (5).
    advectVelocityWithBounds: [0, 1, 6, 8, 13, 19, 29],
    correctVelocity: [0, 1, 3, 5, 6, 8, 20, 21],
    computeCurl: [0, 1, 6],
    // Applying forces evaluates them too.
    applyForces: [0, 1, 2, 3, 6, 8, 11, 12, 30, 31, 73],
    computeDivergence: [0, 1, 7, 8, 15],
    project: [0, 1, 4, 6],
    // Evolution corrects the transported fields first (MacCormack), then evolves them.
    evolveFields: [0, 1, 2, 3, 5, 6, 8, 16, 22, 23, 36],
    advectScalars: [0, 1, 2, 6, 8, 13],
    exportScalars: [0, 4, 6],
  };
  for (const name of SOLID_KERNELS) used[name].push(71);
  for (const name of EMITTER_KERNELS) used[name].push(9, 70);
  for (const name of MESH_SOURCE_KERNELS) used[name].push(26, 27);
  // Every listed kernel reads or writes the brick pools, or the pressure in their slots,
  // through the directory.
  for (const name of LISTED_KERNELS) used[name].push(37, ...DIRECTORY);
  const zeroing = [0, 49, ...DIRECTORY];
  Object.assign(used, {
    advectSmoke: [0, 1, 8, 13, 16, 37, 71, 73, ...DIRECTORY],
    evolveSmoke: [0, 1, 2, 3, 5, 8, 16, 36, 37, 71, 73, ...DIRECTORY],
    zeroSmokeBricks: [0, 16, 49, 74, 75, ...DIRECTORY],
    buildBricks: [0, 9, 26, 36, 39, 45, 57, 63, 64, 70],
    freeBricks: [0, 39, 45, 46, 47, 48, 57],
    allocateBricks: [0, 36, 39, 41, 45, 46, 47, 48, 57, 69],
    linkBricks: [0, 63, 64, 65, 67],
    buildSolids: [0, 10, 25, 49, 63, 65, 70, 72],
    finishBricks: [0, 41, 43, 47, 48, 57],
    zeroVelocityBricks: [...zeroing, 6, 7, 58, 59, 60, 61],
    zeroFieldBricks: [...zeroing, 6, 16, 58, 59],
    growFreeList: [46, 47, 51],
  });
  for (const list of Object.values(used)) list.sort((a, b) => a - b);
  return used;
}

/** Bindings per kernel. */
export const FLUID_KERNELS: Readonly<Record<string, readonly number[]>> = kernels();

/** Only these entry points use the scalar correction specialization. */
export const SCALAR_TRANSPORT_KERNELS = new Set([
  'advectScalars',
  'evolveFields',
  'advectSmoke',
  'evolveSmoke',
]);

/** Kernels dispatched over the listed active bricks. */
export const LISTED = new Set(LISTED_KERNELS);

/** Kernels dispatched over the bricks zeroed this step. */
export const ZEROING = new Set(['zeroVelocityBricks', 'zeroFieldBricks']);

/** Kernels that write the velocity grid. Every other kernel writes the field grid. */
export const VELOCITY_KERNELS = new Set([
  'advectVelocityWithBounds',
  'correctVelocity',
  'computeCurl',
  'applyForces',
  'computeDivergence',
  'project',
  'exportScalars',
]);

export const SMOKE_KERNELS = new Set([
  'advectSmoke',
  'evolveSmoke',
  'zeroSmokeBricks',
]);

const UNIFORM = new Set([0, 22, 51]);
const READ_ONLY_STORAGE = new Set([4, 9, 10, 23, 25, 26, 27, 30, 31, 37, 49, 70, 71]);
const STORAGE = new Set([7, 36, 39, 41, 43, 45, 46, 47, 48, 57, 60, 61, 72]);
const SAMPLERS = new Set([8, 12]);
const WRITE_ONLY_TEXTURES: Record<number, GPUTextureFormat> = {
  6: 'rgba16float',
  13: 'rgba32uint',
  16: 'r16float',
  19: 'rgba16float',
  29: 'rgba16float',
  58: 'rgba16float',
  59: 'rgba16float',
  67: 'rgba32sint',
  69: 'r32sint',
  74: 'r16float',
  75: 'r16float',
};
/** The tile directory's 2D integer textures (tiles.wgsl), and the ones the allocator
 * writes. */
const DIRECTORY_TEXTURES = new Set([...DIRECTORY, 67, 69]);
/** Unsigned integer 3D textures: where the advection kernels' traces landed. */
const UINT_TEXTURES = new Set([5]);

/** The layout entry a fluid binding needs. */
export function fluidBindingLayout(binding: number): GPUBindGroupLayoutEntry {
  const entry: GPUBindGroupLayoutEntry = { binding, visibility: GPUShaderStage.COMPUTE };
  if (UNIFORM.has(binding)) entry.buffer = { type: 'uniform' };
  else if (READ_ONLY_STORAGE.has(binding)) entry.buffer = { type: 'read-only-storage' };
  else if (STORAGE.has(binding)) entry.buffer = { type: 'storage' };
  else if (SAMPLERS.has(binding)) entry.sampler = { type: 'filtering' };
  else if (binding in WRITE_ONLY_TEXTURES)
    entry.storageTexture = {
      access: 'write-only',
      format: WRITE_ONLY_TEXTURES[binding],
      viewDimension: DIRECTORY_TEXTURES.has(binding) ? '2d' : '3d',
    };
  else if (DIRECTORY_TEXTURES.has(binding))
    entry.texture = { viewDimension: '2d', sampleType: 'sint' };
  else if (UINT_TEXTURES.has(binding)) entry.texture = { viewDimension: '3d', sampleType: 'uint' };
  else entry.texture = { viewDimension: '3d', sampleType: 'float' };
  return entry;
}

/** Bindings that hold buffers rather than textures or samplers. */
export function isBufferBinding(binding: number): boolean {
  return UNIFORM.has(binding) || READ_ONLY_STORAGE.has(binding) || STORAGE.has(binding);
}
