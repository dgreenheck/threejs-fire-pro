// Camera raymarch shared by the full-resolution volume draw and the half-resolution depth
// slices. Rays are in world meters and march the renderer's boxes (volume-bricks.wgsl) in
// order along the ray. `grid` holds the voxel size in meters, log2 of the field cells per
// brick and the number of boxes.
// Samples per voxel along a ray, unless the ray's length in the boxes needs more than
// `raySteps` of them.
const FIRE_PRO_STEP_VOXELS: f32 = .5;
fn min3(v: vec3f) -> f32 {
  return min(v.x, min(v.y, v.z));
}

// A pixel's view ray from the near plane.
struct FireProRay {
  origin: vec3f,
  direction: vec3f
};

fn fireProRay(inverseViewProjection: mat4x4f, uv: vec2f) -> FireProRay {
  let clip = uv * vec2f(2, -2) + vec2f(-1, 1);
  let nearH = inverseViewProjection * vec4f(clip, 0, 1);
  let farH = inverseViewProjection * vec4f(clip, .99999, 1);
  let ro = nearH.xyz / nearH.w;
  return FireProRay(ro, normalize(farH.xyz / farH.w - ro));
}

// Where the ray enters and leaves a box of bricks `brickSize` meters wide, from t = 0. The
// exit comes before the entry when the ray misses.
fn fireProHit(ray: FireProRay, box: FireProBox, brickSize: f32) -> vec2f {
  let safeRay = select(vec3f(.000001), ray.direction, abs(ray.direction) > vec3f(.000001));
  let a = (vec3f(box.origin) * brickSize - ray.origin) / safeRay;
  let b = (vec3f(box.origin + box.size) * brickSize - ray.origin) / safeRay;
  let lo = min(a, b);
  let hi = max(a, b);
  return vec2f(max(0.0, max(lo.x, max(lo.y, lo.z))), min3(hi));
}

// The ray's first entry into a box, its last exit, and its length inside every box. The
// entry follows the exit when it misses them all.
fn fireProSpan(ray: FireProRay, boxes: texture_2d<f32>, count: u32, brickSize: f32) -> vec3f {
  var span = vec3f(1e30, 0, 0);
  for (var i = 0u; i < count; i++) {
    let hit = fireProHit(ray, fireProBox(boxes, i), brickSize);
    if hit.y > hit.x {
      span = vec3f(min(span.x, hit.x), max(span.y, hit.y), span.z + hit.y - hit.x);
    }
  }
  return span;
}

// The box the ray is in at `t`, or the next one it enters: its entry, exit and index, or
// index -1. Boxes never overlap, so of the boxes the ray has not left yet, the one it
// enters first is the one.
fn fireProNext(
  ray: FireProRay,
  boxes: texture_2d<f32>,
  count: u32,
  brickSize: f32,
  t: f32
) -> vec3f {
  var next = vec3f(1e30, 0, -1);
  for (var i = 0u; i < count; i++) {
    let hit = fireProHit(ray, fireProBox(boxes, i), brickSize);
    if hit.y > hit.x && hit.y > t && hit.x < next.x {
      next = vec3f(hit, f32(i));
    }
  }
  return next;
}

// The step length of a ray `length` meters long inside the boxes.
fn fireProStep(grid: vec4f, length: f32, raySteps: f32) -> f32 {
  return max(grid.x * FIRE_PRO_STEP_VOXELS, length / raySteps);
}

// Distance to the opaque scene along the ray, or `exit` when the depth is empty.
fn fireProSceneDistance(
  inverseViewProjection: mat4x4f,
  uv: vec2f,
  ray: FireProRay,
  exit: f32,
  sceneDepth: f32
) -> f32 {
  if sceneDepth >= .999999 {
    return exit;
  }
  let clip = uv * vec2f(2, -2) + vec2f(-1, 1);
  let hit = inverseViewProjection * vec4f(clip, sceneDepth, 1);
  return min(exit, dot(hit.xyz / hit.w - ray.origin, ray.direction));
}

// Per-pixel noise in [0, 1) that offsets each sample inside its interval.
fn fireProJitter(pixel: vec2u, time: f32) -> f32 {
  var seed = vec3u(pixel, u32(time * 60.0)) * 1664525u + 1013904223u;
  seed.x += seed.y * seed.z;
  seed.y += seed.z * seed.x;
  seed.z += seed.x * seed.y;
  seed ^= seed >> vec3u(16u);
  seed.x += seed.y * seed.z;
  return f32(seed.x >> 8u) / 16777216.0;
}

// Source radiance (scattering + emission) and extinction of one visible sample at `cell`,
// in field cells of `1 << shift`-cell bricks.
fn fireProShade(
  f: vec4f,
  cell: vec3f,
  direct: vec3f,
  lighting: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  lightShift: u32,
  lightRatio: f32,
  lut: texture_2d<f32>,
  optical: vec4f,
  smoke: vec4f,
  fire: vec4f,
  sootGlow: f32,
  ambientLight: vec4f
) -> vec4f {
  let glow = fireProEmission(f,
    vec4f(smoke.xyz, sootGlow),
    fire,
    optical,
    smoke.w,
    lut,
    linearSampler);
  let density = max(0.0, f.x) * optical.x;
  var scattering = vec3f(0);
  if density > 0.0 {
    let incident = fireProLight(lighting, linearSampler, pages, box, lightShift, cell / lightRatio);
    scattering = smoke.xyz * (ambientLight.xyz + direct * incident.w + incident.rgb * .65) * density;
  }
  return vec4f(scattering + glow, fireProExtinction(f, optical, smoke.w));
}

// The sun term's sample-independent factors, applied once per ray.
fn fireProDirect(direction: vec3f, optical: vec4f, light: vec4f, sceneLight: vec4f) -> vec3f {
  let g = optical.w;
  let mu = dot(direction, normalize(light.xyz));
  let phase = (1 - g * g) / pow(max(.05, 1 + g * g - 2 * g * mu), 1.5);
  return sceneLight.xyz * light.w * (.35 + phase * .3);
}

// Radiance and transmittance accumulated from the ray start up to `t`.
struct FireProMarch {
  t: f32,
  transmittance: f32,
  radiance: vec3f
};

// Where a march stands: the box it is in, and that box's exit.
struct FireProCursor {
  box: FireProBox,
  exit: f32
};

// The cursor at `t`: the box holding it, or else the next box and its entry in `t`.
// False past the last box.
fn fireProEnter(
  cursor: ptr<function, FireProCursor>,
  t: ptr<function, f32>,
  ray: FireProRay,
  boxes: texture_2d<f32>,
  grid: vec4f
) -> bool {
  let next = fireProNext(ray, boxes, u32(grid.z), grid.x * exp2(grid.y), *t);
  if next.z < 0.0 {
    return false;
  }
  *cursor = FireProCursor(fireProBox(boxes, u32(next.z)), next.y);
  *t = max(*t, next.x);
  return true;
}

// Integrate [state.t, end) through the boxes. Samples sit on a lattice of `stepSize` from
// where the march enters each box, jittered inside each interval; steps end at the box's
// exit. Jitter never moves the integration bounds: moving the start drops energy and can
// erase thin depth-clipped slices.
fn fireProMarch(
  state: ptr<function, FireProMarch>,
  end: f32,
  ray: FireProRay,
  stepSize: f32,
  jitter: f32,
  fields: texture_3d<f32>,
  smokeField: texture_3d<f32>,
  pages: texture_2d<f32>,
  boxes: texture_2d<f32>,
  grid: vec4f,
  reconstructionFilter: f32,
  linearSampler: sampler,
  lighting: texture_3d<f32>,
  occupancy: texture_3d<f32>,
  lut: texture_2d<f32>,
  optical: vec4f,
  light: vec4f,
  smoke: vec4f,
  fire: vec4f,
  sootGlow: f32,
  sceneLight: vec4f,
  ambientLight: vec4f
) {
  let rd = ray.direction;
  let safeRay = select(vec3f(.000001), rd, abs(rd) > vec3f(.000001));
  let exitSide = select(vec3f(0), vec3f(1), rd > vec3f(0));
  let voxel = grid.x;
  let shift = u32(grid.y);
  let lightShift = fireProPoolShift(textureDimensions(fields).x, textureDimensions(lighting).x, shift);
  let lightRatio = f32(1u << (shift - lightShift));
  let brickSize = voxel * exp2(grid.y);
  let direct = fireProDirect(rd, optical, light, sceneLight);
  var t = (*state).t;
  var transmittance = (*state).transmittance;
  var radiance = (*state).radiance;
  var cursor = FireProCursor(FireProBox(vec3i(0), 0, vec3i(0)), t);
  loop {
    if t >= end || transmittance < .008 {
      break;
    }
    if t >= cursor.exit {
      if !fireProEnter(&cursor, &t, ray, boxes, grid) {
        break;
      }
      continue;
    }
    let ds = min(stepSize, min(end, cursor.exit) - t);
    let p = ray.origin + rd * (t + jitter * ds);
    let cell = p / voxel;
    // A brick without a slot holds nothing. In one with a slot, the sample's 8-cell
    // block says whether it can find anything, and flame.
    let c = vec3i(floor(cell));
    let page = fireProPage(pages, cursor.box, c >> vec3u(shift));
    var size = brickSize;
    var flags = vec2f(0);
    if page >= 0 {
      flags = fireProOccupancy(occupancy, page, shift, c);
      size = voxel * 8.0;
    }
    if flags.y == 0.0 {
      // Skip whole steps past the empty block, keeping the lattice.
      let edge = (floor(p / size) + exitSide) * size;
      let distance = min3((edge - p) / safeRay);
      t = min(t + max(1.0, floor(distance / stepSize) + 1.0) * stepSize, cursor.exit);
      continue;
    }
    // Smoke was smoothed once on its own lattice. Flame keeps the authored filter.
    let f = fireProRenderField(
      fields, smokeField, linearSampler, pages, cursor.box, shift, cell,
      select(0.0, reconstructionFilter, flags.x > 0.0),
    );
    if f.x + fireProGlow(f.z) > .002 {
      let sample = fireProShade(
        f, cell, direct, lighting, linearSampler, pages, cursor.box, lightShift, lightRatio, lut, optical, smoke,
        fire, sootGlow, ambientLight,
      );
      let tau = sample.w * ds;
      var integral = ds * (1 - .5 * tau + tau * tau / 6);
      if tau >= .001 {
        integral = (1 - exp(-tau)) / sample.w;
      }
      radiance += transmittance * sample.rgb * integral;
      transmittance *= 1 - min(1.0, sample.w * integral);
    }
    // A step cut short by `end` stops exactly there, so a later call resumes
    // without a gap.
    t += ds;
  }
  *state = FireProMarch(t, transmittance, radiance);
}

// Debug views' colors, as linear radiance: a heat map of 0 to 1 (Google's Turbo, fitted by
// polynomials), and the sign of a signed value: blue below zero, red above. Opacity shows
// a signed value's size, so values near zero stay clear.
fn fireProHeatMap(x: f32) -> vec3f {
  let a = vec4f(1, x, x * x, x * x * x);
  let b = a.zw * a.z;
  let srgb = vec3f(
    dot(a,
      vec4f(.13572138, 4.6153926, -42.66032258, 132.13108234)) + dot(b,
      vec2f(-152.94239396, 59.28637943)),
    dot(a,
      vec4f(.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(b,
      vec2f(4.27729857, 2.82956604)),
    dot(a,
      vec4f(.1066733, 12.64194608, -60.58204836, 110.36276771)) + dot(b,
      vec2f(-89.90310912, 27.34824973)),
  );
  return pow(clamp(srgb, vec3f(0), vec3f(1)), vec3f(2.2));
}

fn fireProSignMap(x: f32) -> vec3f {
  return pow(select(vec3f(.85, .1, .16), vec3f(.23, .35, .9), x < 0.0), vec3f(2.2));
}
