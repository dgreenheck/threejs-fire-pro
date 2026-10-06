// Separable field reconstruction. Modes match RenderingOptions.filter order.
// B-splines use paired linear samples (Sigg/Hadwiger, GPU Gems 2 ch.20).
// `position` is in field cells; `fields` is a pool of `1 << shift`-cell bricks, sampled
// within `box` (volume-bricks.wgsl).
fn fireProReconstruct(
  fields: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  shift: u32,
  position: vec3f,
  mode: f32
) -> vec4f {
  if mode < .5 {
    return fireProField(fields, linearSampler, pages, box, shift, position);
  }
  let grid = position - .5;
  let base = floor(grid);
  let q = fract(grid);
  var positions: array<vec3f, 2>;
  var weights: array<vec3f, 2>;
  if mode < 1.5 {
    let center = floor(grid + .5);
    let delta = grid - center;
    let w0 = .5 * (.5 - delta) * (.5 - delta);
    let w1 = .75 - delta * delta;
    let w2 = .5 * (.5 + delta) * (.5 + delta);
    weights[0] = w0 + w1;
    weights[1] = w2;
    positions[0] = center - .5 + w1 / weights[0];
    positions[1] = center + 1.5;
  } else {
    let inv = 1 - q;
    let w0 = inv * inv * inv / 6;
    let w1 = (3 * q * q * q - 6 * q * q + 4) / 6;
    let w2 = (-3 * q * q * q + 3 * q * q + 3 * q + 1) / 6;
    let w3 = q * q * q / 6;
    weights[0] = w0 + w1;
    weights[1] = w2 + w3;
    positions[0] = base - .5 + w1 / weights[0];
    positions[1] = base + 1.5 + w3 / weights[1];
  }
  var value = vec4f(0);
  // Constant indices let the compiler keep positions/weights in registers.
  // Preserve the original accumulation order and all eight filtered samples.
  value += weights[0].x * weights[0].y * weights[0].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[0].x, positions[0].y, positions[0].z));
  value += weights[0].x * weights[0].y * weights[1].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[0].x, positions[0].y, positions[1].z));
  value += weights[0].x * weights[1].y * weights[0].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[0].x, positions[1].y, positions[0].z));
  value += weights[0].x * weights[1].y * weights[1].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[0].x, positions[1].y, positions[1].z));
  value += weights[1].x * weights[0].y * weights[0].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[1].x, positions[0].y, positions[0].z));
  value += weights[1].x * weights[0].y * weights[1].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[1].x, positions[0].y, positions[1].z));
  value += weights[1].x * weights[1].y * weights[0].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[1].x, positions[1].y, positions[0].z));
  value += weights[1].x * weights[1].y * weights[1].z * fireProField(fields,
    linearSampler,
    pages,
    box,
    shift,
    vec3f(positions[1].x, positions[1].y, positions[1].z));
  return max(value, vec4f(0));
}

// Smoke has already been smoothed on its own lattice. Reconstruct fine heat/flame/fuel
// with the authored filter, then sample coarse density once at the camera position.
// Full-resolution smoke is packed into the fine render field and needs no extra fetch.
fn fireProRenderField(
  fields: texture_3d<f32>,
  smoke: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  shift: u32,
  position: vec3f,
  mode: f32
) -> vec4f {
  var f = fireProReconstruct(fields, linearSampler, pages, box, shift, position, mode);
  let smokeShift = fireProSmokeShift(fields, smoke, shift);
  if smokeShift < shift {
    let ratio = f32(1u << (shift - smokeShift));
    f.x = fireProField(smoke, linearSampler, pages, box, smokeShift, position / ratio).x;
  }
  return f;
}
