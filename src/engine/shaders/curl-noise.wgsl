fn hash(p: vec3f) -> f32 {
  var v = (bitcast<vec3u>(vec3i(p)) & vec3u(15u)) * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return f32(v.x >> 8u) / 16777216.0;
}

fn noiseGradient(p: vec3f) -> vec3f {
  let i = floor(p);
  let f = fract(p);
  let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let ds = 30.0 * f * f * (f - 1.0) * (f - 1.0);
  let a = hash(i);
  let b = hash(i + vec3f(1, 0, 0));
  let c = hash(i + vec3f(0, 1, 0));
  let d = hash(i + vec3f(1, 1, 0));
  let e = hash(i + vec3f(0, 0, 1));
  let f1 = hash(i + vec3f(1, 0, 1));
  let g = hash(i + vec3f(0, 1, 1));
  let h = hash(i + vec3f(1, 1, 1));
  return vec3f(
    mix(mix(b - a, d - c, s.y), mix(f1 - e, h - g, s.y), s.z),
    mix(mix(c - a, d - b, s.x), mix(g - e, h - f1, s.x), s.z),
    mix(mix(e - a, f1 - b, s.x), mix(g - c, h - d, s.x), s.y)
  ) * ds;
}

@group(0) @binding(0) var outputNoise: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(8, 4, 4)
fn buildNoise(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(outputNoise);
  if any(id >= size) {
    return;
  }
  let p = (vec3f(id) + .5) / vec3f(size) * 16.0;
  let dx = noiseGradient(p);
  let dy = noiseGradient(p + vec3f(7.1, 3.7, 11.3));
  let dz = noiseGradient(p + vec3f(5.2, 7.9, 9.5));
  textureStore(outputNoise, id, vec4f(dz.y - dy.z, dx.z - dz.x, dy.x - dx.y, 0));
}
