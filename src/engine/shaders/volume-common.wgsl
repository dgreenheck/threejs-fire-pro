// The renderer's compute passes: the lighting and occupancy pools (lighting.wgsl) and the
// scene lights (scene-lights.wgsl). They read the field pool through the boxes'
// page tables (volume-bricks.wgsl), and the simulation's bricks directly.
struct RenderParams {
  grid: vec4f, // voxel size, log2 of the field brick size, shadow samples, flame opacity
  optical: vec4f, // smoke extinction, fire brightness, shadow extinction, anisotropy
  light: vec4f, // directional light direction, intensity
  smoke: vec4f, // smoke albedo RGB, hot-soot contribution
  fire: vec4f, // fire tint RGB, artistic temperature mapping
};

@group(0) @binding(0) var<uniform> u: RenderParams;
@group(0) @binding(1) var fields: texture_3d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(8) var smokeDensity: texture_3d<f32>;
@group(0) @binding(9) var pages: texture_2d<f32>;
@group(0) @binding(10) var boxes: texture_2d<f32>;
@group(0) @binding(11) var blackbodyLut: texture_2d<f32>;
// The simulation's bricks (FluidSimulation.ts RenderBricks): each brick id's page, every
// tile slot's coordinates and box, and the bricks that computed in the latest step, a
// workgroup each (bricks.wgsl).
@group(0) @binding(12) var<storage, read> brickPages: array<i32>;
@group(0) @binding(13) var<storage, read> tileBoxes: array<vec4i>;
@group(0) @binding(14) var<storage, read> activeBricks: array<u32>;
fn fieldShift() -> u32 {
  return u32(u.grid.y);
}

// A brick's edge in meters.
fn brickSize() -> f32 {
  return u.grid.x * exp2(u.grid.y);
}

// The world brick of brick id `id`, in a tile slot whose coordinates are `tile`.
fn brickOfId(id: u32, tile: vec3i) -> vec3i {
  let l = id & 63u;
  return tile * 4 + vec3i(vec3u(l & 3u, (l >> 2u) & 3u, l >> 4u));
}

// A filtered field sample at world point `p`, or zero outside `box`.
fn sampleWorld(box: FireProBox, p: vec3f) -> vec4f {
  let size = brickSize();
  if any(p < vec3f(box.origin) * size) || any(p > vec3f(box.origin + box.size) * size) {
    return vec4f(0);
  }
  var f = fireProField(fields, linearSampler, pages, box, fieldShift(), p / u.grid.x);
  if renderSmokeShift() < fieldShift() {
    f.x = renderSmokeAt(box, p / u.grid.x);
  }
  return f;
}

// Integral of Beer transmission over a homogeneous segment. The series handles
// vacuum/emission-only passes and avoids cancellation for optically thin wisps.
fn integratedTransmission(extinction: f32, distance: f32, attenuation: f32) -> f32 {
  let opticalDepth = extinction * distance;
  if opticalDepth < .001 {
    return distance * (1.0 - .5 * opticalDepth + opticalDepth * opticalDepth / 6.0);
  }
  return (1.0 - attenuation) / extinction;
}

fn emissionAt(f: vec4f) -> vec3f {
  return fireProEmission(f, u.smoke, u.fire, u.optical, u.grid.w, blackbodyLut, linearSampler);
}

// Field and smoke pools share slot coordinates, with different cell counts per slot.
// Infer the smoke lattice from their exact integer dimensions; no extra uniform is
// needed by the scene-light and volume-light caches.
fn renderSmokeShift() -> u32 {
  return fireProSmokeShift(fields, smokeDensity, fieldShift());
}

fn renderSmokeAt(box: FireProBox, fieldPosition: vec3f) -> f32 {
  let shift = renderSmokeShift();
  let ratio = f32(1u << (fieldShift() - shift));
  return fireProField(smokeDensity, linearSampler, pages, box, shift, fieldPosition / ratio).x;
}
