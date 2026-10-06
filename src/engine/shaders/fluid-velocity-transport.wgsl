// Velocity predictor, donor bounds, and bounded MacCormack correction.
// Assembled with the other fluid modules by ShaderSources.ts.
// Bounds borrow the fine scalar scratch. Keep the velocity cells inside each
// fine slot, away from its apron and every other slot. They are point-loaded, and
// every active brick rewrites them before correction, so no clearing is needed.
fn velocityScratchTexel(cell: vec3i, page: i32) -> vec3i {
  return poolOrigin(page, u32(u.pool.w)) + (cell & vec3i(i32(1u << u32(u.velocityGrid.w)) - 1));
}

fn storeVelocityBounds(cell: vec3i, lo: vec3f, hi: vec3f) {
  let page = cellPage(cell);
  if page < 0 {
    return;
  }
  textureStore(outputVelocityMin, velocityScratchTexel(cell, page), vec4f(lo, 0));
  textureStore(outputVelocityMax, velocityScratchTexel(cell, page), vec4f(hi, 0));
}

// Velocity predictor and bounded MacCormack correction
// Traces into bricks without a slot carry still, empty air: open-air inflow takes the
// ambient momentum, not a copy of the outgoing flow (Bridson 2007 notes, section 3.1).
// The component of cell `id` at `w`, and where its trace landed for correctVelocity.
struct AdvectedComponent {
  value: f32,
  donor: u32
};

fn advectVelocityComponent(id: vec3i,
  position: vec3f,
  component: u32,
  original: vec3f) -> AdvectedComponent {
  var faceOffset = vec3f(0);
  faceOffset[component] = .5 * u.grid[component];
  let facePosition = position + faceOffset;
  let faceVelocity = velocityOnFace(id, component, original);
  let midpoint = facePosition - .5 * u.grid.w * faceVelocity;
  let departure = facePosition - u.grid.w * velocityAt(midpoint);
  let donorBase = vec3i(floor((departure - faceOffset) / u.grid.xyz - .5));
  return AdvectedComponent(velocityComponentAt(departure, component),
    packDonor(donorBase - id, VELOCITY_DONOR_BITS));
}

fn writeAdvectedVelocity(id: vec3i, original: vec3f) {
  let position = center(id);
  if solidAt(position) {
    storeVector(id, vec4f(0));
    return;
  }
  // Constant component arguments let the compiler specialize MAC offsets without
  // dynamic vector indexing in a component loop.
  let x = advectVelocityComponent(id, position, 0u, original);
  let y = advectVelocityComponent(id, position, 1u, original);
  let z = advectVelocityComponent(id, position, 2u, original);
  storeVector(id, vec4f(x.value, y.value, z.value, 0));
  // Solid cells skip correction: they store no donors.
  let donor = donorTexel(id, 1u);
  textureStore(outputScalarDonors, donor, vec4u(x.donor | (y.donor << 16u), 0u, 0u, 0u));
  textureStore(outputScalarDonors, donor + vec3i(1, 0, 0), vec4u(z.donor, 0u, 0u, 0u));
}

// All three MAC components reuse these donor bounds. Values are selected from
// half-float inputs, so writing the min/max to half-float loses no precision.
// An 8x8x4 group shares its one-cell positive halo: 405 loads for 256 cells,
// instead of eight loads per cell. All invocations must reach the barrier.
var<workgroup> boundsTile: array<vec4f, 405>;
@compute @workgroup_size(8, 8, 4)
fn advectVelocityWithBounds(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  // The list count is not uniform to the compiler: every lane reaches the barrier.
  let cell = listedCell(group, local, vec3u(8, 8, 4));
  let id = cell.id;
  let origin = id - vec3i(local);
  if cell.listed {
    // The tile and its halo lie in one brick and its apron.
    let block = blockAt(origin, vec3i(8, 8, 4));
    for (var i = lane; i < 405u; i += 256u) {
      let p = vec3i(vec3u(i % 9u, (i / 9u) % 9u, i / 81u));
      boundsTile[i] = blockLoad(velocity, block, origin, p);
    }
  }
  workgroupBarrier();
  if !cell.listed || !cell.interior {
    return;
  }
  writeAdvectedVelocity(id, boundsTile[(local.z * 9u + local.y) * 9u + local.x].xyz);
  var lo = vec3f(1e5);
  var hi = vec3f(-1e5);
  for (var z = 0u; z < 2u; z++) {
    for (var y = 0u; y < 2u; y++) {
      for (var x = 0u; x < 2u; x++) {
        let p = local + vec3u(x, y, z);
        let v = boundsTile[(p.z * 9u + p.y) * 9u + p.x].xyz;
        lo = min(lo, v);
        hi = max(hi, v);
      }
    }
  }
  storeVelocityBounds(id, lo, hi);
}

// Both bounds textures share the same atlas layout: resolve their slot once.
// Bricks without a slot contribute the ambient range [0, 0].
fn velocityBoundsAt(base: vec3i, component: u32) -> vec2f {
  let page = cellPage(base);
  if page < 0 {
    return vec2f(0);
  }
  let texel = velocityScratchTexel(base, page);
  return vec2f(textureLoad(velocityMin, texel, 0)[component],
    textureLoad(velocityMax, texel, 0)[component]);
}

fn correctVelocityComponent(
  id: vec3i,
  position: vec3f,
  component: u32,
  original: vec3f,
  predicted: f32,
  packedDonor: u32
) -> f32 {
  var faceOffset = vec3f(0);
  faceOffset[component] = .5 * u.grid[component];
  let facePosition = position + faceOffset;
  // Use the predictor's cached departure cell. Only traces outside the packed
  // offset range need another backward RK2 trace.
  var donorBase = id + unpackDonor(packedDonor, VELOCITY_DONOR_BITS);
  var faceVelocity = vec3f(0);
  let needsRetrace = packedDonor == noDonor(VELOCITY_DONOR_BITS);
  if needsRetrace {
    faceVelocity = velocityOnFace(id, component, original);
    let backwardMidpoint = facePosition - .5 * u.grid.w * faceVelocity;
    let departure = facePosition - u.grid.w * velocityAt(backwardMidpoint) - faceOffset;
    donorBase = vec3i(floor(departure / u.grid.xyz - .5));
  }
  let bounds = velocityBoundsAt(donorBase, component);
  // A singleton donor range with an identical predictor cannot change: every
  // correction either equals that value or falls back to the predictor. Checking
  // predictor equality preserves half-store rounding and fresh-slot interpolation.
  // Quiet/constant regions skip the face sample, forward RK2 trace and reverse fetch.
  if bounds.x == bounds.y && predicted == bounds.x {
    return predicted;
  }
  // Both RK2 traces start at the same face; share its velocity if we retraced.
  if !needsRetrace {
    faceVelocity = velocityOnFace(id, component, original);
  }
  let reverseDt = -u.grid.w;
  let forwardMidpoint = facePosition - .5 * reverseDt * faceVelocity;
  let forward = facePosition - reverseDt * velocityAt(forwardMidpoint) - faceOffset;
  let reversed = sampleAt(auxiliary, forward, u.grid.xyz)[component];
  let corrected = predicted + .5 * (original[component] - reversed);
  // Revert to first-order transport at extrema instead of introducing new peaks.
  return select(corrected, predicted, corrected < bounds.x || corrected > bounds.y);
}

@compute @workgroup_size(8, 8, 4)
fn correctVelocity(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = listedCell(group, local, vec3u(8, 8, 4));
  if !cell.listed || !cell.interior {
    return;
  }
  let id = cell.id;
  let w = center(id);
  if solidAt(w) {
    storeLoaded(id, vec4f(0));
    return;
  }
  let advected = load(auxiliary, id).xyz;
  let original = load(velocity, id).xyz;
  let donor = donorTexel(id, 1u);
  let xy = textureLoad(donorCells, donor, 0).x;
  let z = textureLoad(donorCells, donor + vec3i(1, 0, 0), 0).x;
  let result = vec3f(correctVelocityComponent(id, w, 0u, original, advected.x, xy & 65535u),
    correctVelocityComponent(id, w, 1u, original, advected.y, xy >> 16u),
    correctVelocityComponent(id, w, 2u, original, advected.z, z));
  storeLoaded(id, vec4f(result, 0));
}
