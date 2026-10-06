// Solid cells: one bit per field cell of every brick with a slot, at the slot's place in
// the pool (fluid-allocation.wgsl buildSolids), so a solid test is a single read. Colliders are drawn
// into them; the ground is the half-space below world y = 0, in whole cells. Cells of
// bricks without a slot are open.
@group(0) @binding(71) var<storage, read> solids: array<u32>;
// The field cell holding world position `w`. On a coarser velocity grid a velocity cell's
// center lies exactly on a corner shared by field cells; the quarter-cell bias picks the
// same one in the fluid solver and the pressure solver, whatever the rounding.
fn fieldCell(w: vec3f) -> vec3i {
  return vec3i(floor(w / u.fieldGrid.xyz + .25));
}

// Whether field cell `c` is solid. `u.counts.y` counts the colliders, `u.counts.w` turns
// the ground on.
fn solidCell(c: vec3i) -> bool {
  if u.counts.w > .5 && c.y < 0 {
    return true;
  }
  if u.counts.y == 0.0 {
    return false;
  }
  let shift = u32(u.pool.w);
  let page = brickPage(c >> vec3u(shift));
  if page < 0 {
    return false;
  }
  let local = vec3u(c & vec3i((1 << shift) - 1));
  let bit = (((local.z << shift) | local.y) << shift) | local.x;
  let words = 1u << (3u * shift - 5u);
  return ((solids[poolSlot(page, u.pool.xy) * words + (bit >> 5u)] >> (bit & 31u)) & 1u) != 0u;
}

// Whether world position `w` lies in a solid cell.
fn solidAt(w: vec3f) -> bool {
  return solidCell(fieldCell(w));
}
