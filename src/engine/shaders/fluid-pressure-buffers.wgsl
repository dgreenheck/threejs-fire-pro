enable f16;
// Pressure and the Poisson RHS live in the independently sized velocity brick slots, in
// half precision (pressure.wgsl). Loads promote to f32 for divergence and projection;
// only stores round to half precision.
@group(0) @binding(4) var<storage, read> pressure: array<f16>;
@group(0) @binding(7) var<storage, read_write> outputScalar: array<f16>;
// The velocity cell `p`'s place in the slot at `page`.
fn scalarIndex(p: vec3i, page: i32) -> u32 {
  let side = 1u << u32(u.velocityGrid.w);
  let local = vec3u(p & vec3i(i32(side) - 1));
  return pageSlot(page) * side * side * side + (local.z * side + local.y) * side + local.x;
}

// Cells of bricks without a slot hold zero pressure.
fn cellPressure(p: vec3i) -> f32 {
  let page = brickPage(p >> vec3u(u32(u.velocityGrid.w)));
  if page < 0 {
    return 0.0;
  }
  return f32(pressure[scalarIndex(p, page)]);
}

fn writeScalar(p: vec3i, value: f32) {
  let page = brickPage(p >> vec3u(u32(u.velocityGrid.w)));
  if page >= 0 {
    outputScalar[scalarIndex(p, page)] = f16(value);
  }
}

// Zero both pressures and the divergence in the cells of bricks that stopped computing
// or got a fresh slot.
@group(0) @binding(60) var<storage, read_write> secondScalar: array<f16>;
@group(0) @binding(61) var<storage, read_write> thirdScalar: array<f16>;
fn zeroScalars(p: vec3i, page: i32) {
  let index = scalarIndex(p, page);
  outputScalar[index] = 0h;
  secondScalar[index] = 0h;
  thirdScalar[index] = 0h;
}

// Debug views: a listed brick's pressure or divergence, whichever binding 4 holds, into its
// fine correction-scratch slot, using the velocity cells plus their one-cell apron.
// At coarse velocity resolutions this fits entirely inside the fine slot interior,
// preserving fine-grid aprons for later scalar transport. Each brick writes its own
// apron from its positive neighbors, or zero where they have no slot.
@compute @workgroup_size(8, 4, 4)
fn exportScalars(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  if !cell.listed || !cell.interior {
    return;
  }
  let texel = velocityScratchTexel(cell.id, cellPage(cell.id));
  textureStore(outputVector, texel, vec4f(cellPressure(cell.id), 0, 0, 0));
  let last = (cell.id & cellMask()) == cellMask();
  for (var m = 1u; m < 8u; m++) {
    let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
    if any(axes & !last) {
      continue;
    }
    let step = select(vec3i(0), vec3i(1), axes);
    textureStore(outputVector, texel + step, vec4f(cellPressure(cell.id + step), 0, 0, 0));
  }
}
