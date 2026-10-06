// Shared fluid records, resource bindings, and numerical helpers.
// Assembled with the other fluid modules by ShaderSources.ts.
// Specialized at pipeline creation: disabled scalar kernels omit correction work.
override SCALAR_MACCORMACK: bool = true;
// 0.1% of one authored scalar unit. A joint smoke, heat and fuel floor clears numerical
// tails without cutting active flames or preventing weak sources accumulating.
const DORMANT_CUTOFF: f32 = 0.001;
fn clearDormant(v: vec4f, injected: bool) -> vec4f {
  if !injected && v.z == 0.0 && all(v.xyw < vec3f(DORMANT_CUTOFF)) {
    return vec4f(0);
  }
  return v;
}

// Host records and resource bindings
// All host-visible records use vec4 slots: no implicit vec3 padding.
// The grids are lattices of world space: cell `c` of a grid with cell size `s` spans
// `c * s` to `(c + 1) * s` meters. Velocity and pressure may use a coarser grid than the
// flame, heat, smoke and fuel fields. `grid` is the grid a kernel writes; the other two
// name both grids.
struct Params {
  grid: vec4f, // cell size in meters, dt
  forces: vec4f, // buoyancy, cooling, smoke weight, velocity damping
  dynamics: vec4f, // dissipation, vorticity, content cutoff, lifetime known to be zero
  noise: vec4f, // time, seed, smoke divisor, log2 smoke cells per brick
  counts: vec4f, // solids of every brick (buildSolids), colliders, mesh source records, ground
  velocityGrid: vec4f, // velocity and pressure cell size, log2 velocity cells per brick
  fieldGrid: vec4f, // flame, heat, smoke and fuel cell size
  pool: vec4f, // slots per row and rows per layer of the brick pools (pool.wgsl), held-back slots, log2 of the field brick size
  bricks: vec4f, // brick ids (tile slots * 64)
  fuel: vec4f, // ignition heat, burn rate (1/s), fuel enabled, predictor borrowed by rendering
};

struct Emitter {
  position: vec4f,
  size: vec4f,
  rates: vec4f,
  velocity: vec4f,
  variation: vec4f, // Perlin period (meters), strength, fuel rate, reserved
};

// A collider in world space. `center.w` is its kind: 0 an ellipsoid of radii `extent`, 1 a
// box of half extents `extent` along the columns of `axes`, 2 mesh occupancy of `size`
// lattice cells from cell `lo`, whose first word in `colliderMasks` is `lo.w`.
struct Collider {
  center: vec4f,
  extent: vec4f,
  axes: mat3x3f,
  lo: vec4i,
  size: vec4i
};

struct MeshSource {
  cell: vec4i,
  scalar: vec4f,
  motion: vec4f
}; // cell: lattice cell
@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var velocity: texture_3d<f32>;
// Scalar channels: X smoke, Y heat, Z remaining flame lifetime, W fuel.
// With coarse smoke, X stages source production; smokeDensity owns the density.
@group(0) @binding(2) var fields: texture_3d<f32>;
// Per-pass input: velocity predictor for correction, scalar predictor for
// evolution, or curl XYZ / magnitude W for forces (FluidSimulation.step).
@group(0) @binding(3) var auxiliary: texture_3d<f32>;
@group(0) @binding(6) var outputVector: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var linearSampler: sampler;
@group(0) @binding(9) var<storage, read> emitters: array<Emitter>;
@group(0) @binding(10) var<storage, read> colliders: array<Collider>;
@group(0) @binding(11) var turbulenceNoise: texture_3d<f32>;
@group(0) @binding(12) var repeatSampler: sampler;
@group(0) @binding(15) var expansionRate: texture_3d<f32>;
@group(0) @binding(16) var outputExpansion: texture_storage_3d<r16float, write>;
@group(0) @binding(19) var outputVelocityMax: texture_storage_3d<rgba16float, write>;
@group(0) @binding(20) var velocityMin: texture_3d<f32>;
@group(0) @binding(21) var velocityMax: texture_3d<f32>;
@group(0) @binding(22) var<uniform> flameParams: FlameParams;
@group(0) @binding(23) var<storage, read> flameKnots: array<vec2f>;
// Mesh colliders' occupancy, one bit per lattice cell, with each mask word-aligned.
@group(0) @binding(25) var<storage, read> colliderMasks: array<u32>;
// Mesh emission by brick: each brick id's table, or -1, then the tables, which hold each
// field cell's record in `meshSources` (0 for none).
@group(0) @binding(26) var<storage, read> meshTables: array<i32>;
@group(0) @binding(27) var<storage, read> meshSources: array<MeshSource>;
// What reaches each tile slot: 8 words a tile, the first index and count of its emitters,
// forces and colliders in the rest of the array, which lists their indices.
@group(0) @binding(70) var<storage, read> tileLists: array<u32>;
const EMITTERS: u32 = 0u;
const FORCES: u32 = 1u;
const COLLIDERS: u32 = 2u;
// The home tile's list of `kind`: its first index and count.
fn tileList(kind: u32) -> vec2u {
  let header = u32(home.tile) * 8u + kind * 2u;
  return vec2u(tileLists[header], tileLists[header + 1u]);
}

@group(0) @binding(29) var outputVelocityMin: texture_storage_3d<rgba16float, write>;
// Where each cell's backward trace landed (packDonor). Velocity correction finishes
// before scalar advection starts, so both reuse one compact pool with no aprons.
// Scalars store four words of packed half bounds. Velocity
// packs its three u16 donors into two texels' X words in either pool format.
@group(0) @binding(5) var donorCells: texture_3d<u32>;
@group(0) @binding(13) var outputScalarDonors: texture_storage_3d<rgba32uint, write>;
// The active bricks this dispatch covers. Every 3D field and velocity texture is a brick
// pool (pool.wgsl), whose slots the tile directory finds (tiles.wgsl).
@group(0) @binding(37) var<storage, read> brickList: array<u32>;
// Rounds to half precision toward zero, as storing to a half-float texture does on Apple
// GPUs: values that skip a texture between kernels keep the same precision.
fn truncateHalf(v: vec4f) -> vec4f {
  // Below 2^-14 halves are subnormal, multiples of 2^-24; above, ten mantissa bits remain.
  let subnormal = trunc(v * 16777216.0) / 16777216.0;
  let normal = bitcast<vec4f>(bitcast<vec4u>(v) & vec4u(0xffffe000u));
  return select(normal, subnormal, abs(v) < vec4f(6.103515625e-5));
}
