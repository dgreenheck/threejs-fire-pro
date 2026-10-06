// Sparse execution runs kernels over a list of bricks of 8x8x8 velocity cells. A list holds
// its count, then the id of each brick (tiles.wgsl), or for the zeroed bricks an id and a
// page. Dispatch X indexes the workgroup tiles of one brick; Y and Z index the listed brick.
const BRICK_LIST_SPLIT: u32 = 65535u;
fn listedBrick(group: vec3u) -> u32 {
  return group.y + group.z * BRICK_LIST_SPLIT;
}

fn brickTileCell(brick: vec3i, tile: u32, size: u32, workgroup: vec3u, local: vec3u) -> vec3i {
  let tiles = max(vec3u(1u), vec3u(size) / workgroup);
  let t = vec3u(tile % tiles.x, (tile / tiles.x) % tiles.y, tile / (tiles.x * tiles.y));
  return brick * i32(size) + vec3i(t * workgroup + local);
}
