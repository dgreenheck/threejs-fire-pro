/**
 * The multigrid's kernels and the bindings each one uses (pressure.wgsl). Pipelines use
 * explicit layouts built from these lists; scripts/check-shaders.mjs checks them against
 * the shader.
 */
export const PRESSURE_KERNELS: Readonly<Record<string, readonly number[]>> = {
  initializePressure: [0, 2, 4, 6, 8, 12, 63, 64, 65, 66],
  relaxBricks: [0, 1, 2, 4, 6, 8, 12, 63, 64, 65, 66],
  relaxCoarseBricks: [0, 1, 2, 4, 6, 8, 12, 63, 64, 65, 66],
  relaxTiles: [0, 1, 2, 4, 6, 8, 63, 64, 65, 66],
  relaxCoarse: [0, 1, 2, 4, 6, 8, 63, 64, 65, 66],
  restrictBricks: [0, 1, 2, 4, 6, 8, 9, 11, 12, 63, 64, 65, 66],
  restrictTiles: [0, 1, 2, 4, 6, 8, 9, 11, 63, 64, 65, 66],
  prolongateBricks: [0, 1, 3, 4, 6, 8, 9, 12, 63, 64, 65, 66],
  prolongateCoarseBricks: [0, 1, 3, 4, 6, 8, 9, 12, 63, 64, 65, 66],
  prolongateTiles: [0, 1, 3, 4, 6, 8, 9, 63, 64, 65, 66],
  buildBrickTopology: [0, 6, 10, 12, 63, 64, 65, 66, 71],
  buildTileTopology: [0, 6, 10, 63, 64, 65, 66, 71],
};
/** The layout entry of a multigrid binding. */
export function pressureBindingLayout(binding: number): GPUBindGroupLayoutEntry {
  const entry: GPUBindGroupLayoutEntry = { binding, visibility: GPUShaderStage.COMPUTE };
  if ([0, 6].includes(binding)) entry.buffer = { type: 'uniform' };
  else if ([4, 10, 11].includes(binding)) entry.buffer = { type: 'storage' };
  // The tile directory's integer textures (tiles.wgsl).
  else if (binding >= 63 && binding <= 66)
    entry.texture = { sampleType: 'sint', viewDimension: '2d' };
  else entry.buffer = { type: 'read-only-storage' };
  return entry;
}
