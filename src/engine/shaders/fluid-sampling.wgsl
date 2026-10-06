// Sparse grid addressing, interpolation, and interior/apron writes.
// Assembled with the other fluid modules by ShaderSources.ts.
// Sparse pool addressing and interpolation
// Each grid shares the fine brick footprint, with its own cells per side.
fn brickShift(spacing: vec3f) -> u32 {
  if spacing.x == u.velocityGrid.x {
    return u32(u.velocityGrid.w);
  }
  if spacing.x == u.fieldGrid.x {
    return u32(u.pool.w);
  }
  return u32(u.noise.w);
}

fn coarseSmoke() -> bool {
  return u.noise.z > 1.0;
}

fn smokeSpacing() -> vec3f {
  return u.fieldGrid.xyz * max(1.0, u.noise.z);
}

@group(0) @binding(73) var smokeDensity: texture_3d<f32>;
fn gridShift() -> u32 {
  return brickShift(u.grid.xyz);
}

fn brickSize() -> u32 {
  return 1u << gridShift();
}

fn cellMask() -> vec3i {
  return vec3i(i32(brickSize()) - 1);
}

// The slot at `page` (pool.wgsl).
fn pageSlot(page: i32) -> u32 {
  return poolSlot(page, u.pool.xy);
}

// A listed brick's cell; the brick becomes home (tiles.wgsl), its record shared by the
// workgroup. Every lane must call this, before any return.
struct ListedCell {
  id: vec3i,
  brick: u32,
  listed: bool,
  interior: bool
};

fn listedCell(group: vec3u, local: vec3u, workgroup: vec3u) -> ListedCell {
  let index = listedBrick(group);
  let listed = index < brickList[0];
  var id = 0u;
  if listed {
    id = brickList[index + 1u];
  }
  loadHomeRecord(id, (local.z * workgroup.y + local.y) * workgroup.x + local.x, listed);
  if !listed {
    return ListedCell(vec3i(0), 0u, false, false);
  }
  enterBrick(id, true);
  home.inWorkgroup = true;
  return ListedCell(brickTileCell(home.brick, group.x, brickSize(), workgroup, local),
    id, true, all(local < vec3u(brickSize())));
}

// Cell `p` of a pool on a grid of cells `spacing` meters wide: zero in a brick without a slot.
fn loadCell(t: texture_3d<f32>, p: vec3i, spacing: vec3f) -> vec4f {
  let shift = brickShift(spacing);
  let page = brickPage(p >> vec3u(shift));
  if page < 0 {
    return vec4f(0);
  }
  return textureLoad(t, poolOrigin(page, shift) + (p & vec3i((1 << shift) - 1)), 0);
}

// Cell `p` of a pool on the grid this kernel writes.
fn load(t: texture_3d<f32>, p: vec3i) -> vec4f {
  return loadCell(t, p, u.grid.xyz);
}

// A trilinear sample at `position`, in cells of a grid with cell centers at .5. A sample
// based in a brick with a slot is one filtered fetch inside that slot; its apron supplies
// the neighbors past the brick.
fn sampleCells(field: texture_3d<f32>, position: vec3f, spacing: vec3f) -> vec4f {
  let shift = brickShift(spacing);
  let base = vec3i(floor(position - .5));
  let brick = base >> vec3u(shift);
  let entry = brickEntry(brick);
  let page = entryPage(entry);
  let last = (1 << shift) - 1;
  // Whether the sample reaches the cells after the base brick, which its apron copies.
  let apron = any((base & vec3i(last)) == vec3i(last));
  if page >= 0 && !(apron && entryFresh(entry)) {
    let texel = vec3f(poolOrigin(page, shift)) + position - vec3f(brick << vec3u(shift));
    return textureSampleLevel(field, linearSampler, texel / vec3f(textureDimensions(field)), 0);
  }
  // An empty base brick reads zero unless the sample reaches past it. Past a fresh slot or
  // an empty brick, interpolate cell by cell.
  if page < 0 && !apron {
    return vec4f(0);
  }
  let fraction = position - .5 - vec3f(base);
  var value = vec4f(0);
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let weights = mix(1.0 - fraction, fraction, vec3f(f32(x), f32(y), f32(z)));
        value += loadCell(field, base + vec3i(x, y, z), spacing) * (weights.x * weights.y * weights.z);
      }
    }
  }
  return value;
}

// A trilinear sample at world position `w`.
fn sampleAt(field: texture_3d<f32>, position: vec3f, spacing: vec3f) -> vec4f {
  return sampleCells(field, position / spacing, spacing);
}

// The cells from `base` to `base + extent` of a pool on the grid this kernel writes, read
// with one page lookup. Within one brick and its apron, they lie in one slot. Mode 1 reads
// that slot from `texel`, 2 reads zero (an empty brick the block stays inside), and 0
// reads cell by cell.
const BLOCK_CELL_LOOKUPS: u32 = 0u;
const BLOCK_SLOT_LOOKUP: u32 = 1u;
const BLOCK_EMPTY: u32 = 2u;
struct SampleBlock {
  texel: vec3i,
  mode: u32
};

fn blockAt(base: vec3i, extent: vec3i) -> SampleBlock {
  let size = i32(brickSize());
  let local = base & vec3i(size - 1);
  if any(local + extent > vec3i(size)) {
    return SampleBlock(vec3i(0), BLOCK_CELL_LOOKUPS);
  }
  let entry = brickEntry(base >> vec3u(gridShift()));
  let page = entryPage(entry);
  // Whether the block reaches the brick's apron, which a fresh slot does not copy yet.
  let apron = any(local + extent == vec3i(size));
  if page >= 0 && !(apron && entryFresh(entry)) {
    return SampleBlock(poolOrigin(page, gridShift()) + local, BLOCK_SLOT_LOOKUP);
  }
  if page < 0 && !apron {
    return SampleBlock(vec3i(0), BLOCK_EMPTY);
  }
  return SampleBlock(vec3i(0), BLOCK_CELL_LOOKUPS);
}

fn blockLoad(t: texture_3d<f32>, block: SampleBlock, base: vec3i, offset: vec3i) -> vec4f {
  if block.mode == BLOCK_SLOT_LOOKUP {
    return textureLoad(t, block.texel + offset, 0);
  }
  if block.mode == BLOCK_EMPTY {
    return vec4f(0);
  }
  return load(t, base + offset);
}

// Stores of the grid this kernel writes. A cell goes to its brick's slot, and to the
// apron of every brick before it whose apron copies the cell: cells on a brick's first
// layer on any axis. `page` is the cell's brick's page.
fn apronCopies(cell: vec3i) -> bool {
  return any((cell & cellMask()) == vec3i(0));
}

fn apronStep(cell: vec3i, m: u32) -> vec3i {
  let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
  if any(axes & ((cell & cellMask()) != vec3i(0))) {
    return vec3i(0);
  }
  return select(vec3i(0), vec3i(1), axes);
}

fn slotTexel(cell: vec3i, page: i32) -> vec3i {
  return poolOrigin(page, gridShift()) + (cell & cellMask());
}

fn cellPage(cell: vec3i) -> i32 {
  return brickPage(cell >> vec3u(gridShift()));
}

// A donor slot has fineBrickCells^3 words, doubled when velocity uses the fine grid.
// Flatten the current grid's cells into it;
// velocity uses two adjacent words, with the first always aligned to an even X.
fn donorTexel(cell: vec3i, wordsShift: u32) -> vec3i {
  let shift = u32(u.pool.w);
  let widthShift = shift + select(0u, 1u, u.velocityGrid.x == u.fieldGrid.x);
  let dimensions = vec3u(1u << widthShift, 1u << shift, 1u << shift);
  let local = vec3u(cell & cellMask());
  let size = brickSize();
  let index = ((local.z * size + local.y) * size + local.x) << wordsShift;
  let page = u32(cellPage(cell));
  let origin = vec3u(page & 255u, (page >> 8u) & 255u, page >> 16u) * dimensions;
  let offset = vec3u(index & (dimensions.x - 1u),
    (index >> widthShift) & (dimensions.y - 1u),
    index >> (widthShift + shift));
  return vec3i(origin + offset);
}

fn writeVector(cell: vec3i, page: i32, value: vec4f) {
  textureStore(outputVector, slotTexel(cell, page), value);
  if !apronCopies(cell) {
    return;
  }
  let brick = cell >> vec3u(gridShift());
  for (var m = 1u; m < 8u; m++) {
    let step = apronStep(cell, m);
    if all(step == vec3i(0)) {
      continue;
    }
    let neighbor = brickPage(brick - step);
    if neighbor >= 0 {
      textureStore(outputVector, slotTexel(cell, neighbor) + step * i32(brickSize()), value);
    }
  }
}

fn storeVector(cell: vec3i, value: vec4f) {
  let page = cellPage(cell);
  if page >= 0 {
    writeVector(cell, page, value);
  }
}

// Stores of a pool that is only loaded cell by cell, never filtered, which needs no apron.
fn storeLoaded(cell: vec3i, value: vec4f) {
  let page = cellPage(cell);
  if page >= 0 {
    textureStore(outputVector, slotTexel(cell, page), value);
  }
}

fn writeExpansion(cell: vec3i, page: i32, value: f32) {
  textureStore(outputExpansion, slotTexel(cell, page), vec4f(value, 0, 0, 0));
  if !apronCopies(cell) {
    return;
  }
  let brick = cell >> vec3u(gridShift());
  for (var m = 1u; m < 8u; m++) {
    let step = apronStep(cell, m);
    if all(step == vec3i(0)) {
      continue;
    }
    let neighbor = brickPage(brick - step);
    if neighbor >= 0 {
      textureStore(outputExpansion,
        slotTexel(cell, neighbor) + step * i32(brickSize()),
        vec4f(value, 0, 0, 0));
    }
  }
}

fn storeExpansion(cell: vec3i, value: f32) {
  let page = cellPage(cell);
  if page >= 0 {
    writeExpansion(cell, page, value);
  }
}

// The world position of the center of cell `id` of the grid this kernel writes.
fn center(id: vec3i) -> vec3f {
  return (vec3f(id) + .5) * u.grid.xyz;
}

// Box average of the field cells under one velocity cell. Each linear sample sits
// on the shared corner of a 2x2x2 block of field cells and returns its exact mean.
fn velocityCellAverage(t: texture_3d<f32>, id: vec3i) -> vec4f {
  let ratio = u.velocityGrid.x / u.fieldGrid.x;
  if ratio < 1.5 {
    return loadCell(t, id, u.fieldGrid.xyz);
  }
  let blocks = u32(ratio) / 2u;
  var sum = vec4f(0);
  for (var z = 0u; z < blocks; z++) {
    for (var y = 0u; y < blocks; y++) {
      for (var x = 0u; x < blocks; x++) {
        let corner = vec3f(id) * ratio + 1.0 + 2.0 * vec3f(f32(x), f32(y), f32(z));
        sum += sampleCells(t, corner, u.fieldGrid.xyz);
      }
    }
  }
  return sum / f32(blocks * blocks * blocks);
}
