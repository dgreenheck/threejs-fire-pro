import forceFields from './shaders/force-fields.wgsl?raw';
import flameCommon from './shaders/flame-common.wgsl?raw';
import smoke from './shaders/smoke.wgsl?raw';
import fluidPressure from './shaders/fluid-pressure-buffers.wgsl?raw';
import bricks from './shaders/bricks.wgsl?raw';
import pool from './shaders/pool.wgsl?raw';
import tiles from './shaders/tiles.wgsl?raw';
import solids from './shaders/solids.wgsl?raw';

// WGSL has no include syntax. Vite imports the fragments and joins them in this explicit
// phase order.
const FLUID_CORE_PARTS = [
  'fluid-common.wgsl',
  'fluid-sampling.wgsl',
  'fluid-sources.wgsl',
  'fluid-advection.wgsl',
  'fluid-velocity-transport.wgsl',
  'fluid-forces.wgsl',
  'fluid-scalar-transport.wgsl',
  'fluid-field-evolution.wgsl',
  'fluid-content.wgsl',
  'fluid-allocation.wgsl',
  'fluid-zeroing.wgsl',
  'fluid-pool-growth.wgsl',
];
const modules = import.meta.glob<string>('./shaders/fluid-*.wgsl', {
  eager: true,
  query: '?raw',
  import: 'default',
});
const fluidCoreSource = FLUID_CORE_PARTS.map((name) => modules[`./shaders/${name}`]).join('\n');

/** The fluid solver's kernels. The f16 directive in the pressure part comes first. */
export const fluidSource =
  fluidPressure +
  bricks +
  pool +
  tiles +
  solids +
  flameCommon +
  fluidCoreSource +
  smoke +
  forceFields;

/** Without scalar correction, only velocity uses donor scratch and needs R32Uint. */
export const fluidFirstOrderSource = fluidSource.replace(
  'var outputScalarDonors: texture_storage_3d<rgba32uint, write>',
  'var outputScalarDonors: texture_storage_3d<r32uint, write>',
);
