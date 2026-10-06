// Half-resolution camera march. Each texel stores the radiance and opacity accumulated
// from the first box entry to (k + 1) / count of the way to the last box exit, so the
// full-resolution volume draw can resume from the last depth in front of the scene.
struct SliceParams {
  inverseViewProjection: mat4x4f,
  optical: vec4f,
  light: vec4f,
  smoke: vec4f,
  fire: vec4f,
  grid: vec4f, // voxel size, log2 of the field brick size, box count
  sceneLight: vec4f,
  ambientLight: vec4f,
  sampling: vec4f, // ray steps, simulation time, reconstruction filter, soot glow
};

@group(0) @binding(0) var<uniform> params: SliceParams;
@group(0) @binding(1) var fields: texture_3d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var lighting: texture_3d<f32>;
@group(0) @binding(4) var occupancy: texture_3d<f32>;
@group(0) @binding(5) var lut: texture_2d<f32>;
@group(0) @binding(6) var slices: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var pages: texture_2d<f32>;
@group(0) @binding(8) var boxes: texture_2d<f32>;
@group(0) @binding(9) var smokeField: texture_3d<f32>;
// One loop per ray records each depth as the march passes it. Separate marches per
// depth would make every lane wait for the slowest lane at each depth.
@compute @workgroup_size(8, 8)
fn buildSlices(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(slices);
  if any(id.xy >= size.xy) {
    return;
  }
  let uv = (vec2f(id.xy) + .5) / vec2f(size.xy);
  let ray = fireProRay(params.inverseViewProjection, uv);
  let grid = params.grid;
  let span = fireProSpan(ray, boxes, u32(grid.z), grid.x * exp2(grid.y));
  let count = size.z;
  let spacing = max(span.y - span.x, 0.0) / f32(count);
  let stepSize = fireProStep(grid, span.z, params.sampling.x);
  let jitter = fireProJitter(id.xy, params.sampling.y);
  let rd = ray.direction;
  let safeRay = select(vec3f(.000001), rd, abs(rd) > vec3f(.000001));
  let exitSide = select(vec3f(0), vec3f(1), rd > vec3f(0));
  let voxel = grid.x;
  let shift = u32(grid.y);
  let lightShift = fireProPoolShift(textureDimensions(fields).x, textureDimensions(lighting).x, shift);
  let lightRatio = f32(1u << (shift - lightShift));
  let brickSize = voxel * exp2(grid.y);
  let direct = fireProDirect(rd, params.optical, params.light, params.sceneLight);
  var t = span.x;
  var transmittance = 1.0;
  var radiance = vec3f(0);
  var recorded = 0u;
  var depth = span.x + spacing;
  var cursor = FireProCursor(FireProBox(vec3i(0), 0, vec3i(0)), t);
  loop {
    if recorded >= count || t >= span.y || transmittance < .008 {
      break;
    }
    if t >= cursor.exit {
      if !fireProEnter(&cursor, &t, ray, boxes, grid) {
        break;
      }
    } else {
      // Steps stop at the next recorded depth so each slice is exact, and at the box's
      // exit.
      let ds = min(stepSize, min(depth, cursor.exit) - t);
      let p = ray.origin + rd * (t + jitter * ds);
      let cell = p / voxel;
      let c = vec3i(floor(cell));
      let page = fireProPage(pages, cursor.box, c >> vec3u(shift));
      var size = brickSize;
      var flags = vec2f(0);
      var skip = 0.0;
      if page >= 0 {
        flags = fireProOccupancy(occupancy, page, shift, c);
        size = voxel * 8.0;
      }
      if flags.y == 0.0 {
        let edge = (floor(p / size) + exitSide) * size;
        skip = max(1.0, floor(min3((edge - p) / safeRay) / stepSize) + 1.0) * stepSize;
      }
      if skip > 0.0 {
        t = min(t + skip, cursor.exit);
      } else {
        let f = fireProRenderField(
          fields, smokeField, linearSampler, pages, cursor.box, shift, cell,
          select(0.0, params.sampling.z, flags.x > 0.0),
        );
        if f.x + fireProGlow(f.z) > .002 {
          let sample = fireProShade(
            f, cell, direct, lighting, linearSampler, pages, cursor.box, lightShift, lightRatio, lut,
            params.optical, params.smoke, params.fire, params.sampling.w, params.ambientLight,
          );
          let tau = sample.w * ds;
          var integral = ds * (1 - .5 * tau + tau * tau / 6);
          if tau >= .001 {
            integral = (1 - exp(-tau)) / sample.w;
          }
          radiance += transmittance * sample.rgb * integral;
          transmittance *= 1 - min(1.0, sample.w * integral);
        }
        t += ds;
      }
    }
    // A skip or a jump between boxes can pass several depths; each keeps the current
    // values.
    loop {
      if recorded >= count || t < depth {
        break;
      }
      textureStore(slices, vec3u(id.xy, recorded), vec4f(radiance, 1 - transmittance));
      recorded++;
      depth = span.x + spacing * f32(recorded + 1u);
    }
  }
  // Depths past the exit or an opaque ray hold the final values.
  for (var slice = recorded; slice < count; slice++) {
    textureStore(slices, vec3u(id.xy, slice), vec4f(radiance, 1 - transmittance));
  }
}
