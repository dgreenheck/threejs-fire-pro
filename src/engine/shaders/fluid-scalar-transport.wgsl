// Scalar predictor and bounded transport correction for heat, smoke, lifetime, and fuel.
// Assembled with the other fluid modules by ShaderSources.ts.
// Scalar predictor, bounded correction, and combustion
// Scalars flow in as zero from bricks without a slot. The closed floor and collider cells
// are cleared by scalar evolution. No per-frame boundary fade changes lifetime.
fn sampleScalarsExplicit(t: texture_3d<f32>, grid: vec3f) -> vec4f {
  // Lifetime needs f32 interpolation. Reuse its eight RGBA reads for every channel.
  let base = vec3i(floor(grid));
  let fraction = fract(grid);
  let block = blockAt(base, vec3i(1));
  var value = vec4f(0);
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let weights = mix(1.0 - fraction, fraction, vec3f(f32(x), f32(y), f32(z)));
        value += blockLoad(t, block, base, vec3i(x, y, z)) * weights.x * weights.y * weights.z;
      }
    }
  }
  return value;
}

// The eight donors of a trilinear sample from `base`: the range of their smoke, heat and
// fuel (x, y, w), which share the bounds test, and of their lifetime, which starts as the
// empty [1, 0].
struct ScalarDonorRange {
  minimum: vec3h,
  maximum: vec3h,
  lifetimeMinimum: f16,
  lifetimeMaximum: f16
};

struct ScalarPrediction {
  value: vec4f,
  donors: ScalarDonorRange
};

fn predictScalarsWithBounds(base: vec3i, fraction: vec3f) -> ScalarPrediction {
  let block = blockAt(base, vec3i(1));
  if block.mode == BLOCK_EMPTY {
    return ScalarPrediction(vec4f(0), ScalarDonorRange(vec3h(0), vec3h(0), 0.0h, 0.0h));
  }
  // Donors come from rgba16float: half bounds retain their exact stored values.
  // Interpolate all channels from the same reads; only the extrema use half precision.
  var donors = ScalarDonorRange(vec3h(65504.0), vec3h(-65504.0), 1.0h, 0.0h);
  var value = vec4f(0);
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let sample = blockLoad(fields, block, base, vec3i(x, y, z));
        let boundsSample = vec4h(sample);
        donors.minimum = min(donors.minimum, boundsSample.xyw);
        donors.maximum = max(donors.maximum, boundsSample.xyw);
        donors.lifetimeMinimum = min(donors.lifetimeMinimum, boundsSample.z);
        donors.lifetimeMaximum = max(donors.lifetimeMaximum, boundsSample.z);
        let interpolationWeights = mix(1.0 - fraction, fraction, vec3f(f32(x), f32(y), f32(z)));
        value += sample * interpolationWeights.x * interpolationWeights.y * interpolationWeights.z;
      }
    }
  }
  return ScalarPrediction(value, donors);
}

// A constant donor stencil leaves only its own value.
fn sameDonors(donors: ScalarDonorRange) -> bool {
  return all(donors.minimum == donors.maximum) && donors.lifetimeMinimum == donors.lifetimeMaximum;
}

// Eight exact half bounds in four words: min smoke/heat, min fuel/lifetime,
// max smoke/heat, max fuel/lifetime. No donor address or trace flags are needed.
fn packScalarBounds(donors: ScalarDonorRange) -> vec4u {
  return vec4u(pack2x16float(vec2f(donors.minimum.xy)),
    pack2x16float(vec2f(vec2h(donors.minimum.z, donors.lifetimeMinimum))),
    pack2x16float(vec2f(donors.maximum.xy)),
    pack2x16float(vec2f(vec2h(donors.maximum.z, donors.lifetimeMaximum))));
}

fn unpackScalarBounds(words: vec4u) -> ScalarDonorRange {
  let minimum = vec2h(unpack2x16float(words.x));
  let minimumFuelLifetime = vec2h(unpack2x16float(words.y));
  let maximum = vec2h(unpack2x16float(words.z));
  let maximumFuelLifetime = vec2h(unpack2x16float(words.w));
  return ScalarDonorRange(vec3h(minimum, minimumFuelLifetime.x),
    vec3h(maximum, maximumFuelLifetime.x), minimumFuelLifetime.y,
    maximumFuelLifetime.y);
}

// Rendering borrows the scalar predictor between steps and smooths density into
// its apron, including past the sparse domain. Existing neighbor cells replace
// their apron copies during advection; absent neighbors have no invocation, so
// the last interior cell explicitly restores those apron texels to empty air.
fn clearMissingPredictorApron(cell: vec3i) {
  if u.fuel.w < .5 {
    return;
  }
  let last = (cell & cellMask()) == cellMask();
  if !any(last) {
    return;
  }
  let brick = cell >> vec3u(gridShift());
  let texel = slotTexel(cell, cellPage(cell));
  for (var m = 1u; m < 8u; m++) {
    let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
    if any(axes & !last) {
      continue;
    }
    let step = select(vec3i(0), vec3i(1), axes);
    if brickPage(brick + step) < 0 {
      textureStore(outputVector, texel + step, vec4f(0));
    }
  }
}

// Corrected transport computes its predictor and cached bounds from one donor scan.
// First-order transport uses hardware interpolation and writes no correction metadata.
@compute @workgroup_size(8, 4, 4)
fn advectScalars(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  if !cell.listed {
    return;
  }
  var trace = cellTrace(cell.id);
  let departure = traceBack(&trace);
  if !SCALAR_MACCORMACK {
    let sampled = sampleAt(fields, departure, u.grid.xyz);
    let predicted = vec4f(select(sampled.x, 0.0, coarseSmoke()), sampled.yzw);
    storeVector(cell.id, vec4f(predicted.xy, clamp(predicted.z, 0.0, 1.0), predicted.w));
    clearMissingPredictorApron(cell.id);
    return;
  }
  // A still trace must stay exactly at its voxel center. Dividing its world position
  // back into grid coordinates can round an edge sample into the empty neighbor.
  let departureCell = vec3f(cell.id) + (departure - trace.position) / u.grid.xyz;
  let donorBase = vec3i(floor(departureCell));
  let prediction = predictScalarsWithBounds(donorBase, fract(departureCell));
  let predicted = vec4f(select(prediction.value.x, 0.0, coarseSmoke()), prediction.value.yzw);
  let lifetime = select(predicted.z, 0.0, u.dynamics.w > .5);
  storeVector(cell.id, vec4f(predicted.xy, lifetime, predicted.w));
  textureStore(outputScalarDonors, donorTexel(cell.id, 0u), packScalarBounds(prediction.donors));
  clearMissingPredictorApron(cell.id);
}

fn correctedFineScalars(id: vec3i, trace: ptr<function, CellTrace>) -> vec4f {
  // Advection stored the exact half bounds; no source donor addressing or scan remains.
  let texel = slotTexel(id, cellPage(id));
  let donors = unpackScalarBounds(textureLoad(donorCells, donorTexel(id, 0u), 0));
  if sameDonors(donors) {
    return vec4f(vec2f(donors.minimum.xy),
      clamp(f32(donors.lifetimeMinimum), 0.0, 1.0),
      f32(donors.minimum.z));
  }
  let advected = textureLoad(auxiliary, texel, 0);
  // Coarse smoke discards the fine X channel after this function. Variation in
  // that temporary source channel must not force reverse traces when the retained
  // heat/lifetime/fuel are constant. Require the stored predictor to match too:
  // f32 lifetime interpolation followed by a half store can round below the donors.
  let retainedDonors = vec3f(vec3h(donors.minimum.y, donors.lifetimeMinimum, donors.minimum.z));
  if coarseSmoke() && all(donors.minimum.yz == donors.maximum.yz) && donors.lifetimeMinimum == donors.lifetimeMaximum && all(advected.yzw == retainedDonors) {
    return vec4f(advected.xy, clamp(advected.z, 0.0, 1.0), advected.w);
  }
  let forward = traceForward(trace);
  let original = textureLoad(fields, texel, 0);
  // Where the donors' lifetime is constant and the advected lifetime matches it, that is
  // the result: a correction either equals it or reverts to it. Smoke without flame
  // skips the reverse lifetime trace.
  var lifetime = advected.z;
  var reversed: vec3f;
  if donors.lifetimeMinimum != donors.lifetimeMaximum || advected.z != f32(donors.lifetimeMinimum) {
    let forwardCell = vec3f(id) + (forward - (*trace).position) / u.grid.xyz;
    let reversedScalars = sampleScalarsExplicit(auxiliary, forwardCell);
    reversed = reversedScalars.xyw;
    let correctedLifetime = advected.z + .5 * (original.z - reversedScalars.z);
    lifetime = select(correctedLifetime,
      advected.z,
      correctedLifetime < f32(donors.lifetimeMinimum) || correctedLifetime > f32(donors.lifetimeMaximum));
  } else {
    // Without a lifetime correction, hardware filtering remains the cheaper sample.
    reversed = sampleAt(auxiliary, forward, u.grid.xyz).xyw;
  }
  // Use first-order transport where the correction leaves the donor range, like
  // velocity. Clamping there left one-voxel ridges and hard zero edges in smoke.
  let corrected = advected.xyw + .5 * (original.xyw - reversed);
  let bounded = select(corrected,
    advected.xyw,
    (corrected < vec3f(donors.minimum)) | (corrected > vec3f(donors.maximum)));
  return vec4f(bounded.xy, clamp(lifetime, 0.0, 1.0), bounded.z);
}

// Smoke transport uses its own coarse grid; the fine X channel only stages this
// step's integrated source output for volume averaging, never persistent density.
fn correctedScalars(id: vec3i, trace: ptr<function, CellTrace>) -> vec4f {
  if !SCALAR_MACCORMACK {
    let transported = load(auxiliary, id);
    return vec4f(select(transported.x, 0.0, coarseSmoke()), transported.yzw);
  }
  let corrected = correctedFineScalars(id, trace);
  return vec4f(select(corrected.x, 0.0, coarseSmoke()), corrected.yzw);
}
