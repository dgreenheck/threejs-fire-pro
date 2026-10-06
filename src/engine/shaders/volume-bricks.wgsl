// The renderer's view of the bricks (VolumeLighting.ts). It marches boxes of bricks, one
// for each cluster of touching tiles with something to draw, none overlapping another.
// Each box has a page table: the pages of its bricks (pool.wgsl), x fastest, from the
// box's first entry of a 2D texture. Positions are in cells of the world lattice with
// centers at .5: field cell c holds world points from c to c + 1 voxels.
// `size` bricks from brick `origin`, their pages from page table entry `first`.
struct FireProBox {
  origin: vec3i,
  first: i32,
  size: vec3i
};

// Boxes per row of the box texture, two texels each (VolumeLighting.ts BOX_ROW).
const FIRE_PRO_BOX_ROW: u32 = 256u;
// log2 of the entries per row of the page table (VolumeLighting.ts PAGE_ROW).
const FIRE_PRO_PAGE_ROW: u32 = 12u;
fn fireProBox(boxes: texture_2d<f32>, index: u32) -> FireProBox {
  let texel = vec2u((index % FIRE_PRO_BOX_ROW) * 2u, index / FIRE_PRO_BOX_ROW);
  let a = textureLoad(boxes, texel, 0);
  let b = textureLoad(boxes, texel + vec2u(1u, 0u), 0);
  return FireProBox(vec3i(a.xyz), i32(a.w), vec3i(b.xyz));
}

// The page of brick `brick`, or -1 outside the box or without a slot.
fn fireProPage(pages: texture_2d<f32>, box: FireProBox, brick: vec3i) -> i32 {
  let local = brick - box.origin;
  if any(local < vec3i(0)) || any(local >= box.size) {
    return -1;
  }
  let entry = u32(box.first + (local.z * box.size.y + local.y) * box.size.x + local.x);
  let column = entry & ((1u << FIRE_PRO_PAGE_ROW) - 1u);
  return i32(textureLoad(pages, vec2u(column, entry >> FIRE_PRO_PAGE_ROW), 0).x);
}

// The value of cell `cell` of a pool of `1 << shift`-cell bricks: zero in a brick without
// a slot.
fn fireProLoad(
  pool: texture_3d<f32>,
  pages: texture_2d<f32>,
  box: FireProBox,
  shift: u32,
  cell: vec3i
) -> vec4f {
  let page = fireProPage(pages, box, cell >> vec3u(shift));
  if page < 0 {
    return vec4f(0);
  }
  return textureLoad(pool, poolOrigin(page, shift) + (cell & vec3i((1 << shift) - 1)), 0);
}

// A trilinear sample of a pool of `1 << shift`-cell bricks at `position`, in cells. A
// sample based in a brick with a slot is one filtered fetch inside that slot; its apron
// supplies the neighbors past the brick.
fn fireProSample(
  pool: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  shift: u32,
  position: vec3f
) -> vec4f {
  let base = vec3i(floor(position - .5));
  let brick = base >> vec3u(shift);
  let page = fireProPage(pages, box, brick);
  if page >= 0 {
    let texel = vec3f(poolOrigin(page, shift)) + position - vec3f(brick * (1 << shift));
    return textureSampleLevel(pool, linearSampler, texel / vec3f(textureDimensions(pool)), 0);
  }
  // The base brick has no slot. Unless the sample reaches the cells after it, so is the
  // sample empty; otherwise interpolate cell by cell.
  let last = (1 << shift) - 1;
  if all((base & vec3i(last)) < vec3i(last)) {
    return vec4f(0);
  }
  let f = position - .5 - vec3f(base);
  var value = vec4f(0);
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let w = mix(1.0 - f, f, vec3f(f32(x), f32(y), f32(z)));
        value += fireProLoad(pool, pages, box, shift, base + vec3i(x, y, z)) * (w.x * w.y * w.z);
      }
    }
  }
  return value;
}

// A filtered sample at `position`, in cells of `1 << shift`-cell bricks, clamped to the
// box's outer cell centers like a clamp-to-edge sampler.
fn fireProField(
  fields: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  shift: u32,
  position: vec3f
) -> vec4f {
  let size = f32(1 << shift);
  let lo = vec3f(box.origin) * size + .5;
  let hi = vec3f(box.origin + box.size) * size - .5;
  return fireProSample(fields, linearSampler, pages, box, shift, clamp(position, lo, hi));
}

// Field and smoke pools share slots. Their dimensions identify the smoke lattice.
fn fireProPoolShift(fieldWidth: u32, width: u32, shift: u32) -> u32 {
  let slots = fieldWidth / ((1u << shift) + 1u);
  return firstLeadingBit(width / slots - 1u);
}

fn fireProSmokeShift(fields: texture_3d<f32>, smoke: texture_3d<f32>, shift: u32) -> u32 {
  return fireProPoolShift(textureDimensions(fields).x, textureDimensions(smoke).x, shift);
}

// Pressure/divergence use velocity cells plus their apron inside a fine-grid
// scratch slot. Logical brick size is independent of physical slot stride.
// Clamp to the box as fireProField does, and interpolate missing base bricks explicitly.
fn fireProSolverField(
  pool: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  poolShift: u32,
  velocityShift: u32,
  position: vec3f
) -> f32 {
  let side = i32(1u << velocityShift);
  let lo = vec3f(box.origin * side) + .5;
  let hi = vec3f((box.origin + box.size) * side) - .5;
  let p = clamp(position, lo, hi);
  let base = vec3i(floor(p - .5));
  let brick = base >> vec3u(velocityShift);
  let page = fireProPage(pages, box, brick);
  if page >= 0 {
    let texel = vec3f(poolOrigin(page, poolShift)) + p - vec3f(brick * side);
    return textureSampleLevel(pool, linearSampler, texel / vec3f(textureDimensions(pool)), 0).x;
  }
  let f = p - .5 - vec3f(base);
  var value = 0.0;
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let cell = base + vec3i(x, y, z);
        let neighbor = fireProPage(pages, box, cell >> vec3u(velocityShift));
        if neighbor >= 0 {
          let texel = poolOrigin(neighbor, poolShift) + (cell & vec3i(side - 1));
          let w = mix(1.0 - f, f, vec3f(f32(x), f32(y), f32(z)));
          value += textureLoad(pool, texel, 0).x * (w.x * w.y * w.z);
        }
      }
    }
  }
  return value;
}

// Incident light at `position`, in lighting texels at the selected resolution, in a pool of
// `1 << shift`-texel bricks (lighting.wgsl buildLighting). A sample based in a brick
// without a slot is unlit and unshadowed: such bricks hold no smoke.
fn fireProLight(
  lighting: texture_3d<f32>,
  linearSampler: sampler,
  pages: texture_2d<f32>,
  box: FireProBox,
  shift: u32,
  position: vec3f
) -> vec4f {
  let size = f32(1 << shift);
  let lo = vec3f(box.origin) * size + .5;
  let p = clamp(position, lo, vec3f(box.origin + box.size) * size - .5);
  let brick = vec3i(floor(p - .5)) >> vec3u(shift);
  let page = fireProPage(pages, box, brick);
  if page < 0 {
    return vec4f(0, 0, 0, 1);
  }
  let texel = vec3f(poolOrigin(page, shift)) + p - vec3f(brick * (1 << shift));
  return textureSampleLevel(lighting,
    linearSampler,
    texel / vec3f(textureDimensions(lighting)),
    0);
}

// Whether samples in the 8-cell block of field cell `cell`, in a brick of `1 << shift`
// cells at `page`, can find flame (x) and anything visible (y): the block's flags in the
// brick's slot of the occupancy pool (lighting.wgsl buildOccupancy).
fn fireProOccupancy(occupancy: texture_3d<f32>, page: i32, shift: u32, cell: vec3i) -> vec2f {
  let p = u32(page);
  let slot = vec3u(p & 255u, (p >> 8u) & 255u, p >> 16u);
  let block = vec3u(cell & vec3i((1 << shift) - 1)) >> vec3u(3u);
  return textureLoad(occupancy, slot * (1u << (shift - 3u)) + block, 0).xy;
}
