// Brick pools: sparse storage for grids that hold content in only a few of their bricks.
// Each brick with a slot owns one slot of a pool texture, a brick of `1 << shift` cells per
// side plus one texel per side: the brick's own cells, then one apron layer holding the
// first cells of the bricks after it on each axis, so a filtered sample based in a brick
// never leaves its slot. A brick's page is where its slot sits in the pool, in slots and
// packed eight bits an axis (x | y << 8 | z << 16), or -1 when the brick has no slot. The
// simulation finds pages through its tile directory (tiles.wgsl), the renderer through the
// page tables of its boxes (volume-bricks.wgsl). Pages and power-of-two bricks keep every
// address to shifts and multiplies: integer division is slow on GPUs.
// The slot at `page`, in a pool of `side.x` slots a row and `side.y` rows a layer: the
// slot's index in the free list and in every per-slot buffer.
fn poolSlot(page: i32, side: vec2f) -> u32 {
  let p = u32(page);
  return ((p >> 16u) * u32(side.y) + ((p >> 8u) & 255u)) * u32(side.x) + (p & 255u);
}

// The first texel of the slot at `page`, in a pool of bricks `1 << shift` cells wide.
fn poolOrigin(page: i32, shift: u32) -> vec3i {
  let p = u32(page);
  return vec3i(vec3u(p & 255u, (p >> 8u) & 255u, p >> 16u) * ((1u << shift) + 1u));
}
