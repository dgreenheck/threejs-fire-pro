// Curl, source/external forces, divergence, and pressure projection.
// Assembled with the other fluid modules by ShaderSources.ts.
fn curlNoise(p: vec3f) -> vec3f {
  let rotation = mat3x3f(vec3f(.36, -.8, .48), vec3f(.48, .6, .64), vec3f(-.8, 0, .6));
  return transpose(rotation) * textureSampleLevel(turbulenceNoise,
    repeatSampler,
    rotation * p / 16.0,
    0).xyz;
}

fn curlMagnitude(p: vec3i) -> f32 {
  return load(auxiliary, p).w;
}

fn enforceFaces(input: vec3f, w: vec3f) -> vec3f {
  let cell = u.grid.xyz;
  if solidAt(w) {
    return vec3f(0);
  }
  var v = input;
  if solidAt(w + vec3f(cell.x, 0, 0)) {
    v.x = 0;
  }
  if solidAt(w + vec3f(0, cell.y, 0)) {
    v.y = 0;
  }
  if solidAt(w + vec3f(0, 0, cell.z)) {
    v.z = 0;
  }
  return v;
}

// Curl, external forces, divergence, and pressure projection
// Curl uses only off-axis derivatives of centered MAC components. Their source
// cells fit in a one-cell halo: share 360 velocity loads across 128 cells instead
// of reconstructing six centered vectors independently in every invocation.
// Keep f32 values and the original arithmetic order, including the .5 average.
var<workgroup> curlTile: array<vec4f, 360>;
fn curlCentered(p: vec3u, component: u32) -> f32 {
  var before = p;
  before[component] -= 1u;
  let a = curlTile[(p.z * 6u + p.y) * 10u + p.x][component];
  let b = curlTile[(before.z * 6u + before.y) * 10u + before.x][component];
  return .5 * (a + b);
}

@compute @workgroup_size(8, 4, 4)
fn computeCurl(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  if cell.listed {
    let first = cell.id - vec3i(local) - 1;
    for (var i = lane; i < 360u; i += 128u) {
      let offset = vec3i(vec3u(i % 10u, (i / 10u) % 6u, i / 60u));
      curlTile[i] = load(velocity, first + offset);
    }
  }
  workgroupBarrier();
  if !cell.listed || !cell.interior {
    return;
  }
  let p = local + 1u;
  let h = .5 / u.grid.xyz;
  let x = vec3u(1, 0, 0);
  let y = vec3u(0, 1, 0);
  let z = vec3u(0, 0, 1);
  let dx = vec2f(curlCentered(p + x, 1u) - curlCentered(p - x, 1u),
    curlCentered(p + x, 2u) - curlCentered(p - x, 2u)) * h.x;
  let dy = vec2f(curlCentered(p + y, 0u) - curlCentered(p - y, 0u),
    curlCentered(p + y, 2u) - curlCentered(p - y, 2u)) * h.y;
  let dz = vec2f(curlCentered(p + z, 0u) - curlCentered(p - z, 0u),
    curlCentered(p + z, 1u) - curlCentered(p - z, 1u)) * h.z;
  let c = vec3f(dy.y - dz.y, dz.x - dx.y, dx.x - dy.x);
  storeLoaded(cell.id, vec4f(c, length(c)));
}

// Forces live at cell centers. Transfer to staggered faces by averaging,
// as in Fedkiw/Stam/Jensen 2001 Appendix A; never shift all components alike.
fn centerForce(p: vec3i) -> vec3f {
  let w = center(p);
  if solidAt(w) {
    return vec3f(0);
  }
  let f = velocityCellAverage(fields, p);
  var density = f.x;
  if coarseSmoke() {
    density = smokeVelocityAverage(p);
  }
  var confinement = vec3f(0);
  if u.dynamics.y != 0.0 {
    let c = load(auxiliary, p).xyz;
    let eta = vec3f(curlMagnitude(p + vec3i(1, 0, 0)) - curlMagnitude(p - vec3i(1, 0, 0)),
      curlMagnitude(p + vec3i(0, 1, 0)) - curlMagnitude(p - vec3i(0, 1, 0)),
      curlMagnitude(p + vec3i(0, 0, 1)) - curlMagnitude(p - vec3i(0, 0, 1)));
    let spacing = u.grid.xyz;
    confinement = cross(eta / (length(eta) + .0001),
      c) * u.dynamics.y * min(spacing.x,
      min(spacing.y, spacing.z));
  }
  return confinement + externalForces(p,
    w) + vec3f(0,
    u.forces.x * f.y - u.forces.z * density,
    0);
}

// Whether the brick of velocity cell `p` computes.
fn computesAt(p: vec3i) -> bool {
  return entryComputes(brickEntry(p >> vec3u(u32(u.velocityGrid.w))));
}

// The force at cell `p`. Bricks that compute have forces, the rest none; rounded toward
// zero to half precision, as the half-float force pool held them.
fn cellForce(p: vec3i) -> vec3f {
  if !computesAt(p) {
    return vec3f(0);
  }
  return truncateHalf(vec4f(centerForce(p), 0)).xyz;
}

// The tile's forces and those of the cells after it on each axis, which its faces average.
var<workgroup> forceTile: array<vec3f, 225>;
fn sourceVelocity(value: f32, face: vec3f, component: u32) -> f32 {
  var result = value;
  let list = tileList(EMITTERS);
  for (var i = 0u; i < list.y; i++) {
    let e = emitters[tileLists[list.x + i]];
    if e.rates.w == 0.0 {
      continue;
    }
    let weight = emitterWeight(e, face);
    if weight == 0.0 {
      continue;
    }
    var radial = vec3f(0);
    if e.velocity.w != 0.0 {
      let d = face - e.position.xyz;
      radial = d / max(length(d), 1e-6) * e.velocity.w;
    }
    result = mix(result,
      e.velocity[component] + radial[component],
      1.0 - exp(-e.rates.w * weight * u.grid.w));
  }
  let mesh = meshSourceAt(face);
  if mesh.motion.w > 0.0 {
    result = mix(result, mesh.motion[component] / mesh.motion.w,
      1.0 - exp(-mesh.motion.w * u.grid.w));
  }
  return result;
}

@compute @workgroup_size(8, 4, 4)
fn applyForces(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  if cell.listed {
    let origin = cell.id - vec3i(local);
    for (var i = lane; i < 225u; i += 128u) {
      let q = vec3u(i % 9u, (i / 9u) % 5u, i / 45u);
      // Only cells past the tile on one axis are read.
      if dot(vec3u(q >= vec3u(8u, 4u, 4u)), vec3u(1u)) < 2u {
        forceTile[i] = cellForce(origin + vec3i(q));
      }
    }
  }
  workgroupBarrier();
  if !cell.listed || !cell.interior {
    return;
  }
  let p = cell.id;
  let pos = center(p);
  let halfCell = .5 * u.grid.xyz;
  let t = local.x + 9u * (local.y + 5u * local.z);
  let centered = forceTile[t];
  let force = .5 * (centered + vec3f(forceTile[t + 1u].x,
      forceTile[t + 9u].y,
      forceTile[t + 45u].z));
  let v = cellVelocity(p) * exp(-u.forces.w * u.grid.w) + u.grid.w * force;
  let sourced = vec3f(sourceVelocity(v.x, pos + vec3f(halfCell.x, 0, 0), 0u),
    sourceVelocity(v.y, pos + vec3f(0, halfCell.y, 0), 1u),
    sourceVelocity(v.z, pos + vec3f(0, 0, halfCell.z), 2u));
  storeLoaded(p, vec4f(enforceFaces(sourced, pos), 0));
}

@compute @workgroup_size(8, 4, 4)
fn computeDivergence(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  if !cell.listed || !cell.interior {
    return;
  }
  let p = cell.id;
  let h = u.grid.xyz;
  let reference = min(h.x, min(h.y, h.z));
  let centered = cellVelocity(p);
  let divergence = (centered.x - cellVelocity(p - vec3i(1,
        0,
        0)).x) / h.x + (centered.y - cellVelocity(p - vec3i(0,
        1,
        0)).y) / h.y + (centered.z - cellVelocity(p - vec3i(0,
        0,
        1)).z) / h.z;
  writeScalar(p, reference * reference * (divergence - velocityCellAverage(expansionRate, p).x));
}

// Pressure unknowns, as the pressure solver's finest level defines them: cells whose
// own brick and the bricks of their negative neighbors compute. Around them lies a
// one-cell ring of still, ambient air at zero pressure. Below the ground is solid, so no
// brick there needs to compute.
@group(0) @binding(39) var<storage, read_write> brickState: array<u32>;
fn pressureUnknown(p: vec3i) -> bool {
  var lo = (p - 1) >> vec3u(u32(u.velocityGrid.w));
  if u.counts.w > .5 {
    lo.y = max(lo.y, 0);
  }
  let hi = p >> vec3u(u32(u.velocityGrid.w));
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        if !entryComputes(brickEntry(vec3i(x, y, z))) {
          return false;
        }
      }
    }
  }
  return true;
}

fn projectComponent(p: vec3i,
  offset: vec3i,
  axis: u32,
  centerPressure: f32,
  value: f32) -> f32 {
  let neighbor = p + offset;
  if solidAt(center(neighbor)) {
    return 0;
  }
  // Faces with no pressure unknown on either side belong to the ambient ring.
  if !pressureUnknown(p) && !pressureUnknown(neighbor) {
    return 0;
  }
  return value - (cellPressure(neighbor) - centerPressure) / u.grid[axis];
}

@compute @workgroup_size(8, 4, 4)
fn project(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  if !cell.listed || !cell.interior {
    return;
  }
  let p = cell.id;
  if solidAt(center(p)) {
    storeVector(p, vec4f(0));
    return;
  }
  let centerPressure = cellPressure(p);
  let v = cellVelocity(p);
  // Each face's solid test also enforces zero flux: no second geometry traversal.
  let projected = vec3f(projectComponent(p, vec3i(1, 0, 0), 0u, centerPressure, v.x),
    projectComponent(p, vec3i(0, 1, 0), 1u, centerPressure, v.y),
    projectComponent(p, vec3i(0, 0, 1), 2u, centerPressure, v.z));
  storeVector(p, vec4f(projected, 0));
}
