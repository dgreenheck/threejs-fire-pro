// Emitter masks, mesh emission lookup, and seeded burst variation.
// Assembled with the other fluid modules by ShaderSources.ts.
// The table of mesh emission of brick id `id`, or -1.
fn meshTable(id: i32) -> i32 {
  return select(-1, meshTables[max(id, 0)], id >= 0);
}

// Mesh emission is indexed on the field grid, like the solids (solids.wgsl).
fn meshSourceAt(w: vec3f) -> MeshSource {
  if u.counts.z == 0.0 {
    return MeshSource(vec4i(0), vec4f(0), vec4f(0));
  }
  let cell = fieldCell(w);
  let shift = u32(u.pool.w);
  let table = meshTable(brickId(cell >> vec3u(shift)));
  if table < 0 {
    return MeshSource(vec4i(0), vec4f(0), vec4f(0));
  }
  let local = vec3u(cell & vec3i((1 << shift) - 1));
  let index = (((((u32(table) << shift) | local.z) << shift) | local.y) << shift) | local.x;
  return meshSources[meshTables[brickCount() + index]];
}

fn emitterWeight(e: Emitter, p: vec3f) -> f32 {
  let d = (p - e.position.xyz) / max(e.size.xyz, vec3f(.03));
  var r: f32;
  if e.size.w > .5 {
    r = max(abs(d.x), max(abs(d.y), abs(d.z)));
  } else {
    r = length(d);
  }
  if r >= 1.0 || e.position.w < .5 {
    return 0;
  }
  // Authored compact source mask. Active intervals are resolved by the host.
  return 1.0 - smoothstep(.5, 1.0, r);
}

// Seeded 3D gradient Perlin noise: corner dot products with quintic interpolation.
// Unlike the velocity curl texture, this is sampled only during burst injection.
fn burstGradient(cell: vec3i, offset: vec3f) -> f32 {
  let c = bitcast<vec3u>(cell);
  var h = c.x * 374761393u + c.y * 668265263u + c.z * 2246822519u + bitcast<u32>(u.noise.y) * 3266489917u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = (h ^ (h >> 16u)) & 15u;
  let a = select(offset.y, offset.x, h < 8u);
  let b = select(select(offset.z, offset.x, h == 12u || h == 14u), offset.y, h < 4u);
  return select(-a, a, (h & 1u) == 0u) + select(-b, b, (h & 2u) == 0u);
}

fn burstPerlin(p: vec3f) -> f32 {
  let cell = vec3i(floor(p));
  let f = fract(p);
  let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = burstGradient(cell, f);
  let b = burstGradient(cell + vec3i(1, 0, 0), f - vec3f(1, 0, 0));
  let c = burstGradient(cell + vec3i(0, 1, 0), f - vec3f(0, 1, 0));
  let d = burstGradient(cell + vec3i(1, 1, 0), f - vec3f(1, 1, 0));
  let e = burstGradient(cell + vec3i(0, 0, 1), f - vec3f(0, 0, 1));
  let g = burstGradient(cell + vec3i(1, 0, 1), f - vec3f(1, 0, 1));
  let h = burstGradient(cell + vec3i(0, 1, 1), f - vec3f(0, 1, 1));
  let j = burstGradient(cell + vec3i(1, 1, 1), f - vec3f(1, 1, 1));
  return mix(mix(mix(a, b, s.x), mix(c, d, s.x), s.y),
    mix(mix(e, g, s.x), mix(h, j, s.x), s.y),
    s.z);
}

fn burstPattern(p: vec3f) -> f32 {
  let n = (burstPerlin(p) + .35 * burstPerlin(p * 2.0 + vec3f(5.2, 9.7, 3.1))) / 1.35;
  return clamp(.5 + .5 * n, 0.0, 1.0);
}

fn burstVariation(e: Emitter, p: vec3f) -> vec2f {
  if e.variation.y == 0.0 {
    return vec2f(1);
  }
  // Source-relative coordinates make translated bursts share the same material
  // pattern. Fixed offsets decorrelate heat and lifetime; neither animates in time.
  let q = (p - e.position.xyz) / e.variation.x + vec3f(3.17, 7.53, 11.29);
  return vec2f(1) - e.variation.y * vec2f(burstPattern(q),
    burstPattern(q + vec3f(17.3, 2.8, 8.6)));
}
