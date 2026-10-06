// Clear new or retired slots and their apron copies before reuse.
// Assembled with the other fluid modules by ShaderSources.ts.
// Zeroing. Every cell of a listed slot, so the slot never holds stale values; a fresh
// slot's apron too. Zeroed cells also reach the aprons that copy them.
@group(0) @binding(49) var<storage, read> zeroList: array<u32>;
struct ZeroCell {
  id: vec3i,
  page: i32,
  fresh: bool,
  listed: bool
};

fn zeroCell(group: vec3u, local: vec3u, workgroup: vec3u) -> ZeroCell {
  let index = listedBrick(group);
  if index >= zeroList[0] {
    return ZeroCell(vec3i(0), -1, false, false);
  }
  let brick = zeroList[1u + 2u * index];
  let payload = zeroList[2u + 2u * index];
  let page = i32(payload & 0xffffffu);
  // A brick that gave up its slot zeroes the aprons its last step's neighbors copy it to.
  enterBrick(brick, true);
  let id = brickTileCell(home.brick, group.x, brickSize(), workgroup, local);
  return ZeroCell(id, page, (payload & FRESH_SLOT) != 0u, all(local < vec3u(brickSize())));
}

// The apron texels a cell on a brick's last layer owns: past it on those axes.
fn apronTexel(cell: ZeroCell, m: u32) -> vec4i {
  let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
  let last = (cell.id & cellMask()) == cellMask();
  if any(axes & !last) {
    return vec4i(0);
  }
  return vec4i(slotTexel(cell.id, cell.page) + select(vec3i(0), vec3i(1), axes), 1);
}

// The three vector pools of a grid, zeroed together.
@group(0) @binding(58) var secondVector: texture_storage_3d<rgba16float, write>;
@group(0) @binding(59) var thirdVector: texture_storage_3d<rgba16float, write>;
fn zeroVectorTexel(texel: vec3i) {
  textureStore(outputVector, texel, vec4f(0));
  textureStore(secondVector, texel, vec4f(0));
  textureStore(thirdVector, texel, vec4f(0));
}

// The cell's texels in every pool, as writeVector stores them, and on a fresh slot the
// apron texels it owns.
fn zeroVectors(cell: ZeroCell) {
  zeroVectorTexel(slotTexel(cell.id, cell.page));
  if apronCopies(cell.id) {
    let brick = cell.id >> vec3u(gridShift());
    for (var m = 1u; m < 8u; m++) {
      let step = apronStep(cell.id, m);
      if all(step == vec3i(0)) {
        continue;
      }
      let neighbor = brickPage(brick - step);
      if neighbor >= 0 {
        zeroVectorTexel(slotTexel(cell.id, neighbor) + step * i32(brickSize()));
      }
    }
  }
  if !cell.fresh {
    return;
  }
  for (var m = 1u; m < 8u; m++) {
    let texel = apronTexel(cell, m);
    if texel.w != 0 {
      zeroVectorTexel(texel.xyz);
    }
  }
}

// The velocity grid's vector pools, and its pressure and divergence.
@compute @workgroup_size(8, 4, 4)
fn zeroVelocityBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = zeroCell(group, local, vec3u(8, 4, 4));
  if !cell.listed {
    return;
  }
  zeroVectors(cell);
  zeroScalars(cell.id, cell.page);
}

// The field grid's vector pools and its expansion rate.
@compute @workgroup_size(8, 4, 4)
fn zeroFieldBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = zeroCell(group, local, vec3u(8, 4, 4));
  if !cell.listed {
    return;
  }
  zeroVectors(cell);
  writeExpansion(cell.id, cell.page, 0.0);
  if !cell.fresh {
    return;
  }
  for (var m = 1u; m < 8u; m++) {
    let texel = apronTexel(cell, m);
    if texel.w != 0 {
      textureStore(outputExpansion, texel.xyz, vec4f(0));
    }
  }
}
