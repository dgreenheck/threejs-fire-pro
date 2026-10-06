// Native function used by a Three.js NodeMaterial. TSL supplies resource bindings
// and fragment IO; the matrix maps viewport clip coordinates into world space.
// TSL reads the parameters up to the first closing parenthesis: keep them out of the
// parameter comments.
// volume-march.wgsl supplies the ray setup and the camera march.
fn fireProSceneVolume(
  fields: texture_3d<f32>,
  smokeField: texture_3d<f32>,
  smokeDivisor: f32,
  velocityDivisor: f32,
  pages: texture_2d<f32>,
  boxes: texture_2d<f32>,
  grid: vec4f, // voxel size, log2 of the field brick size, box count
  reconstructionFilter: f32,
  velocityField: texture_3d<f32>,
  solverField: texture_3d<f32>,
  linearSampler: sampler,
  lighting: texture_3d<f32>,
  occupancy: texture_3d<f32>,
  lut: texture_2d<f32>,
  slices: texture_3d<f32>,
  halfResolution: f32,
  inverseViewProjection: mat4x4f,
  optical: vec4f,
  light: vec4f,
  smoke: vec4f,
  fire: vec4f,
  sootGlow: f32,
  sampling: vec2f,
  viewport: vec2f,
  sceneDepth: f32,
  uv: vec2f,
  sceneLight: vec4f,
  ambientLight: vec4f,
  displayField: f32,
  debugScale: vec2f
) -> vec4f {
  let ray = fireProRay(inverseViewProjection, uv);
  let span = fireProSpan(ray, boxes, u32(grid.z), grid.x * exp2(grid.y));
  if span.x >= span.y {
    return vec4f(0);
  }
  let end = fireProSceneDistance(inverseViewProjection, uv, ray, span.y, sceneDepth);
  if end <= span.x {
    return vec4f(0);
  }
  let stepSize = fireProStep(grid, span.z, sampling.x);
  let jitter = fireProJitter(vec2u(uv * viewport), sampling.y);
  if displayField > .5 {
    let shift = u32(grid.y);
    var t = span.x;
    var transmittance = 1.0;
    var radiance = vec3f(0);
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
      let cell = (ray.origin + ray.direction * (t + jitter * ds)) / grid.x;
      // Velocity spacing and brick footprint are independent.
      let velocityShift = shift - u32(log2(velocityDivisor));
      let velocityCell = cell / velocityDivisor;
      let spacing = grid.x * velocityDivisor;
      var scalar = 0.0;
      if displayField > 3.5 && displayField < 4.5 {
        scalar = length(
          fireProField(velocityField, linearSampler, pages, cursor.box, velocityShift, velocityCell).xyz,
        );
      } else if displayField > 6.5 && displayField < 7.5 {
        // Vorticity: the magnitude of the velocity's curl, by central differences over a cell.
        var d: array<vec3f, 3>;
        for (var axis = 0; axis < 3; axis++) {
          var offset = vec3f(0);
          offset[axis] = 1.0;
          let ahead = fireProField(
            velocityField, linearSampler, pages, cursor.box, velocityShift, velocityCell + offset,
          ).xyz;
          let behind = fireProField(
            velocityField, linearSampler, pages, cursor.box, velocityShift, velocityCell - offset,
          ).xyz;
          d[axis] = (ahead - behind) / (2.0 * spacing);
        }
        scalar = length(vec3f(d[1].z - d[2].y, d[2].x - d[0].z, d[0].y - d[1].x));
      } else if displayField > 7.5 && displayField < 8.5 {
        // The flames' target divergence, on the field grid.
        scalar = fireProField(solverField, linearSampler, pages, cursor.box, shift, cell).x;
      } else if displayField > 8.5 {
        // The solver's pressure, a velocity times a cell, and the divergence it solved for,
        // 1/s times a cell squared, on the velocity grid.
        let raw = fireProSolverField(solverField,
          linearSampler,
          pages,
          cursor.box,
          shift,
          velocityShift,
          velocityCell);
        scalar = select(raw / (spacing * spacing), raw / spacing, displayField < 9.5);
      } else {
        let f = fireProReconstruct(
          fields, linearSampler, pages, cursor.box, shift, cell, reconstructionFilter,
        );
        if displayField < 1.5 {
          scalar = f.z;
        } else if displayField < 2.5 {
          scalar = f.y;
        } else if displayField < 3.5 {
          scalar = fireProField(smokeField,
            linearSampler,
            pages,
            cursor.box,
            shift - u32(log2(smokeDivisor)),
            cell / smokeDivisor).x;
        } else if displayField < 5.5 {
          scalar = fireProGlow(f.z);
        } else {
          scalar = f.w;
        }
      }
      // A heat map of 0 to debugScale.x, or when debugScale.y is 1 the sign of a value from
      // -debugScale.x to debugScale.x. Opacity grows with the magnitude: a full-scale voxel
      // blocks a fifth of the light.
      let level = scalar / debugScale.x;
      var color: vec3f;
      var amount: f32;
      if debugScale.y > .5 {
        amount = min(abs(level), 1.0);
        color = fireProSignMap(level);
      } else {
        amount = clamp(level, 0.0, 1.0);
        color = fireProHeatMap(amount);
      }
      let alpha = 1 - exp(-amount * ds / grid.x * .2);
      radiance += transmittance * color * alpha;
      transmittance *= 1 - alpha;
      t += ds;
    }
    return vec4f(radiance, 1 - transmittance);
  }
  var state = FireProMarch(span.x, 1.0, vec3f(0));
  if halfResolution > .5 {
    // The half-resolution pass stored the march up to evenly spaced depths from the first
    // box entry to the last exit. Resume from the last one in front of the scene and
    // march the rest at full resolution, so opaque geometry inside the volume still clips
    // exactly.
    let count = f32(textureDimensions(slices).z);
    let slice = floor((end - span.x) / (span.y - span.x) * count);
    if slice >= 1.0 {
      let stored = textureSampleLevel(
        slices, linearSampler, vec3f(uv, (min(slice, count) - .5) / count), 0,
      );
      let t = span.x + (span.y - span.x) * min(slice, count) / count;
      state = FireProMarch(t, 1.0 - stored.a, stored.rgb);
    }
  }
  fireProMarch(
    &state, end, ray, stepSize, jitter, fields, smokeField, pages, boxes, grid, reconstructionFilter,
    linearSampler, lighting, occupancy, lut, optical, light, smoke, fire, sootGlow, sceneLight,
    ambientLight,
  );
  return vec4f(state.radiance, 1 - state.transmittance);
}
