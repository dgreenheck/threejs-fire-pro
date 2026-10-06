// Shared MAC velocity sampling, RK2 traces, and packed donor addresses.
// Assembled with the other fluid modules by ShaderSources.ts.
// MAC velocity sampling and cached RK2 traces
// Bricks without a slot hold still air, and nothing lies below the ground.
fn cellVelocity(p: vec3i) -> vec3f {
  return loadCell(velocity, p, u.velocityGrid.xyz).xyz;
}

// Velocity components live on each cell's positive faces (a staggered MAC grid).
fn velocityComponentAt(position: vec3f, component: u32) -> f32 {
  let spacing = u.velocityGrid.xyz;
  var faceOffset = vec3f(0);
  faceOffset[component] = .5 * spacing[component];
  return sampleAt(velocity, position - faceOffset, spacing)[component];
}

fn velocityAt(position: vec3f) -> vec3f {
  return vec3f(velocityComponentAt(position, 0u),
    velocityComponentAt(position, 1u),
    velocityComponentAt(position, 2u));
}

// At a MAC face its normal component is already stored in this cell. Reuse it;
// only the two tangential components need interpolation. Work in cell coordinates
// so constructing these exact half-cell offsets does not round through world space.
fn velocityOnFace(id: vec3i, component: u32, originalVelocity: vec3f) -> vec3f {
  var facePosition = vec3f(id) + .5;
  facePosition[component] += .5;
  if component == 0u {
    return vec3f(originalVelocity.x,
      sampleCells(velocity, facePosition - vec3f(0, .5, 0), u.velocityGrid.xyz).y,
      sampleCells(velocity, facePosition - vec3f(0, 0, .5), u.velocityGrid.xyz).z);
  }
  if component == 1u {
    return vec3f(sampleCells(velocity, facePosition - vec3f(.5, 0, 0), u.velocityGrid.xyz).x,
      originalVelocity.y, sampleCells(velocity,
        facePosition - vec3f(0, 0, .5),
        u.velocityGrid.xyz).z);
  }
  return vec3f(sampleCells(velocity, facePosition - vec3f(.5, 0, 0), u.velocityGrid.xyz).x,
    sampleCells(velocity,
      facePosition - vec3f(0, .5, 0),
      u.velocityGrid.xyz).y, originalVelocity.z);
}

// A cell's RK2 traces over one step from its center: back to its donors, and forward for
// the MacCormack reverse sample. Both start from the velocity at the center, sampled when
// first needed.
struct CellTrace {
  position: vec3f,
  velocity: vec3f,
  velocitySampled: bool
};

fn cellTrace(id: vec3i) -> CellTrace {
  return CellTrace(center(id), vec3f(0), false);
}

fn traceVelocity(trace: ptr<function, CellTrace>) -> vec3f {
  if !(*trace).velocitySampled {
    (*trace).velocity = velocityAt((*trace).position);
    (*trace).velocitySampled = true;
  }
  return (*trace).velocity;
}

fn traceBack(trace: ptr<function, CellTrace>) -> vec3f {
  let centerVelocity = traceVelocity(trace);
  return (*trace).position - u.grid.w * velocityAt((*trace).position - .5 * u.grid.w * centerVelocity);
}

fn traceForward(trace: ptr<function, CellTrace>) -> vec3f {
  let centerVelocity = traceVelocity(trace);
  return (*trace).position + u.grid.w * velocityAt((*trace).position + .5 * u.grid.w * centerVelocity);
}

// Where a backward trace landed: the base cell of its trilinear sample, whose donors bound
// the correction. The advection kernels store it as an offset from the cell that traced,
// `bits` bits an axis, so the corrections need not trace back again. Past that range they
// store noDonor(bits), and the correction traces again.
// Velocity u16: [15: retrace][14..10: Z][9..5: Y][4..0: X]. X/Y faces share
// one u32; the Z face uses the next u32. Offsets are biased signed integers.
const VELOCITY_DONOR_BITS: u32 = 5u;
fn noDonor(bits: u32) -> u32 {
  return 1u << (3u * bits);
}

fn packDonor(offset: vec3i, bits: u32) -> u32 {
  let biased = offset + (1 << (bits - 1u));
  if any(biased < vec3i(0)) || any(biased >= vec3i(1 << bits)) {
    return noDonor(bits);
  }
  let b = vec3u(biased);
  return b.x | (b.y << bits) | (b.z << (2u * bits));
}

fn unpackDonor(word: u32, bits: u32) -> vec3i {
  let axes = vec3u(word, word >> bits, word >> (2u * bits)) & vec3u((1u << bits) - 1u);
  return vec3i(axes) - (1 << (bits - 1u));
}
