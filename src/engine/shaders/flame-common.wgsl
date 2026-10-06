// FLAME-LIFETIME-001 equations used by the fluid solver.
struct FlameParams {
  rates: vec4f, // dt, lifespan, smoke/sec, heat/sec
  extra: vec4f, // authored target divergence (1/sec), padding
  counts: vec4u, // smoke, heat, expansion knot counts, reserved
};

// Products: smoke, heat and expansion.
struct FlameResult {
  lifetime: f32,
  products: vec3f
};

fn rampIntegral(channel: u32, lo: f32, hi: f32) -> f32 {
  var area = 0.0;
  for (var i = 1u; i < flameParams.counts[channel]; i++) {
    let a = flameKnots[channel * 8u + i - 1u];
    let b = flameKnots[channel * 8u + i];
    let left = max(lo, a.x);
    let right = min(hi, b.x);
    if right > left {
      let y0 = mix(a.y, b.y, (left - a.x) / (b.x - a.x));
      let y1 = mix(a.y, b.y, (right - a.x) / (b.x - a.x));
      area += (right - left) * (y0 + y1) * 0.5;
    }
  }
  return area;
}

fn evaluateFlame(before: f32) -> FlameResult {
  if before <= 0.0 {
    return FlameResult(0.0, vec3f(0));
  }
  let after = max(0.0, before - flameParams.rates.x / flameParams.rates.y);
  let smoke = flameParams.rates.z * flameParams.rates.y * rampIntegral(0u, after, before);
  let heat = flameParams.rates.w * flameParams.rates.y * rampIntegral(1u, after, before);
  // Average across the full step, including any inactive time after expiry.
  let expansion = flameParams.extra.x * (flameParams.rates.y * rampIntegral(2u,
      after,
      before) / flameParams.rates.x);
  return FlameResult(after, vec3f(smoke, heat, expansion));
}
