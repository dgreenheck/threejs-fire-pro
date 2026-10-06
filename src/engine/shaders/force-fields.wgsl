// Forces as rows of vec4s (ForceFields.ts): five per force, then the regions of the
// emitters that targeted forces follow, two each. Directions and centers are in world
// space. A tile lists the forces that reach it (fluid-common.wgsl tileList).
struct ForceField {
  axis: vec4f, // direction/axis xyz, acceleration strength
  center: vec4f, // center xyz, kind: wind=0, turbulence=1, vortex=2, radial=3
  settings: vec4f, // scale/radius, lift, inward pull, reserved
  source: vec4f, // first region row, region count, has mesh targets, targeted
  size: vec4f, // reserved
};

@group(0) @binding(30) var<storage, read> forceData: array<vec4f>;
// Velocity cells that mesh targets cover, by brick: each brick id's first entry, or
// NO_COVERAGE, then for each such brick a count and its entries: a force's index and one
// bit per velocity cell of the brick.
@group(0) @binding(31) var<storage, read> forceCoverage: array<u32>;
const NO_COVERAGE: u32 = 0xffffffffu;
fn forceRecord(index: u32) -> ForceField {
  let row = index * 5u;
  return ForceField(forceData[row],
    forceData[row + 1u],
    forceData[row + 2u],
    forceData[row + 3u],
    forceData[row + 4u]);
}

// Whether mesh targets of force `index` cover velocity cell `cell`.
fn meshCovers(index: u32, cell: vec3i) -> bool {
  let shift = u32(u.velocityGrid.w);
  let side = 1u << shift;
  let id = brickId(cell >> vec3u(shift));
  if id < 0 {
    return false;
  }
  let first = forceCoverage[id];
  if first == NO_COVERAGE {
    return false;
  }
  let local = vec3u(cell & vec3i(i32(side) - 1));
  let bit = (local.z * side + local.y) * side + local.x;
  for (var entry = 0u; entry < forceCoverage[first]; entry++) {
    let row = first + 1u + entry * (1u + (side * side * side + 31u) / 32u);
    if forceCoverage[row] == index {
      return ((forceCoverage[row + 1u + (bit >> 5u)] >> (bit & 31u)) & 1u) != 0u;
    }
  }
  return false;
}

fn coverage(f: ForceField, index: u32, cell: vec3i, p: vec3f) -> f32 {
  if f.source.w == 0.0 {
    return 1.0;
  }
  if f.source.z == 1.0 && meshCovers(index, cell) {
    return 1.0;
  }
  var covered = 0.0;
  for (var i = 0u; i < u32(f.source.y); i++) {
    let row = u32(f.source.x) + 2u * i;
    let center = forceData[row];
    let delta = abs((p - center.xyz) / max(forceData[row + 1u].xyz, vec3f(.03)));
    let distance = select(length(delta), max(delta.x, max(delta.y, delta.z)), center.w == 3.0);
    covered = max(covered, 1.0 - smoothstep(.5, 1.0, distance));
  }
  return covered;
}

fn externalForces(cell: vec3i, p: vec3f) -> vec3f {
  var result = vec3f(0);
  let list = tileList(FORCES);
  for (var i = 0u; i < list.y; i++) {
    let index = tileLists[list.x + i];
    let f = forceRecord(index);
    let weight = coverage(f, index, cell, p);
    if weight == 0.0 {
      continue;
    }
    var acceleration = f.axis.xyz * f.axis.w;
    if f.center.w == 1.0 {
      let q = p * f.settings.x * .65 + vec3f(u.noise.y, -u.noise.x * .7, u.noise.y * .25);
      acceleration = (curlNoise(q) + .45 * curlNoise(q * 2.03 + vec3f(13.7, 5.3, 19.1))) * f.axis.w;
    } else if f.center.w >= 2.0 {
      let delta = p - f.center.xyz;
      var radial = delta;
      if f.center.w == 2.0 {
        radial -= f.axis.xyz * dot(delta, f.axis.xyz);
      }
      let distance = length(radial);
      let radius = max(f.settings.x, .01);
      // Finite at the axis/center and smoothly localized around the radius.
      let direction = radial / max(distance, radius * .25);
      let falloff = 1.0 / (1.0 + dot(radial, radial) / (radius * radius));
      if f.center.w == 2.0 {
        acceleration = (cross(f.axis.xyz,
            direction) * f.axis.w - direction * f.settings.z + f.axis.xyz * f.settings.y) * falloff;
      } else {
        acceleration = direction * f.axis.w * falloff;
      }
    }
    result += acceleration * weight;
  }
  return result;
}
