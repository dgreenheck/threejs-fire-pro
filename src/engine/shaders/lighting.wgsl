// volume-common.wgsl supplies the shared uniform, field, sampler, bricks and optical model.
@group(0) @binding(3) var lightOutput: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var occupancyOutput: texture_storage_3d<rg16float, write>;
@group(0) @binding(5) var pageOutput: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var renderFieldOutput: texture_storage_3d<rgba16float, write>;
@group(0) @binding(17) var renderSmokeOutput: texture_storage_3d<r16float, write>;
// Write each brick's page into the page table of its box, every brick id a lane in rows of
// 65535 workgroups. Bricks outside every box keep the -1 their entries start with.
@compute @workgroup_size(64)
fn exportPages(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) groups: vec3u
) {
  let id = gid.x + gid.y * groups.x * 64u;
  if id >= arrayLength(&brickPages) {
    return;
  }
  let tile = tileBoxes[id >> 6u];
  if tile.w < 0 {
    return;
  }
  let box = fireProBox(boxes, u32(tile.w));
  let local = brickOfId(id, tile.xyz) - box.origin;
  if any(local < vec3i(0)) || any(local >= box.size) {
    return;
  }
  let entry = u32(box.first + (local.z * box.size.y + local.y) * box.size.x + local.x);
  let column = entry & ((1u << FIRE_PRO_PAGE_ROW) - 1u);
  textureStore(
    pageOutput, vec2u(column, entry >> FIRE_PRO_PAGE_ROW), vec4f(f32(brickPages[id]), 0, 0, 0),
  );
}

// A brick with a slot inside its box: its id, page, box and world brick.
struct ListedBrick {
  id: u32,
  page: i32,
  box: FireProBox,
  brick: vec3i
};

// Brick `id`'s slot, box and place; false when it has no slot or lies in no box.
fn slotted(id: u32, result: ptr<function, ListedBrick>) -> bool {
  let page = brickPages[id];
  let tile = tileBoxes[id >> 6u];
  if page < 0 || tile.w < 0 {
    return false;
  }
  let box = fireProBox(boxes, u32(tile.w));
  let brick = brickOfId(id, tile.xyz);
  let local = brick - box.origin;
  if any(local < vec3i(0)) || any(local >= box.size) {
    return false;
  }
  *result = ListedBrick(id, page, box, brick);
  return true;
}

// The active brick of workgroup `group`, if it has a slot in a box.
fn listed(group: vec3u, result: ptr<function, ListedBrick>) -> bool {
  let index = listedBrick(group);
  return index < activeBricks[0] && slotted(activeBricks[1u + index], result);
}

// A rendering-only copy: smoke gets the separable [1,2,1]/4 filter in each axis;
// heat, flame lifetime and fuel retain their original values. Eight half-cell-offset
// trilinear fetches evaluate the 27-cell stencil. Read world neighbors through the
// page table, never adjacent pool slots. Write each slot's apron too so subsequent
// camera and shadow sampling needs only the usual single filtered fetch.
@compute @workgroup_size(64)
fn smoothSmoke(
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) groups: vec3u,
  @builtin(local_invocation_index) lane: u32
) {
  var work: ListedBrick;
  if !listed(group, &work) {
    return;
  }
  let shift = fieldShift();
  let side = 1u << shift;
  let edge = side + 1u;
  let origin = poolOrigin(work.page, shift);
  let first = work.brick * i32(side);
  // Fine-grid dispatches distribute independent outputs across groups in X;
  // Y/Z still identify the brick. Each texel keeps its original eight-tap sum.
  for (var index = group.x * 64u + lane; index < edge * edge * edge; index += groups.x * 64u) {
    let local = vec3i(vec3u(index % edge, (index / edge) % edge, index / (edge * edge)));
    let cell = first + local;
    let original = fireProLoad(fields, pages, work.box, shift, cell);
    var density = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
      let offset = vec3f(vec3u(corner, corner >> 1u, corner >> 2u) & vec3u(1u));
      density += fireProField(smokeDensity,
        linearSampler,
        pages,
        work.box,
        shift,
        vec3f(cell) + offset).x;
    }
    textureStore(renderFieldOutput, origin + local, vec4f(max(0.0, density * .125), original.yzw));
  }
}

// The same [1,2,1]/4 filter evaluated at coarse smoke cell centers, including aprons.
// Independent 64-lane chunks avoid smoothing and copying every fine-grid cell.
@compute @workgroup_size(64)
fn smoothCoarseSmoke(
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) groups: vec3u,
  @builtin(local_invocation_index) lane: u32
) {
  var work: ListedBrick;
  if !listed(group, &work) {
    return;
  }
  let shift = renderSmokeShift();
  let side = 1u << shift;
  let edge = side + 1u;
  let origin = poolOrigin(work.page, shift);
  let first = work.brick * i32(side);
  for (var index = group.x * 64u + lane; index < edge * edge * edge; index += groups.x * 64u) {
    let local = vec3i(vec3u(index % edge, (index / edge) % edge, index / (edge * edge)));
    let cell = first + local;
    var density = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
      let offset = vec3f(vec3u(corner, corner >> 1u, corner >> 2u) & vec3u(1u));
      density += fireProField(smokeDensity, linearSampler, pages, work.box, shift,
        vec3f(cell) + offset).x;
    }
    textureStore(renderSmokeOutput, origin + local, vec4f(max(0.0, density * .125), 0, 0, 0));
  }
}

// Smoothing can put density into a populated slot's apron even when the brick it
// represents has no slot. Include those donors in the conservative occupancy bound.
fn cachedSmokeCell(box: FireProBox, shift: u32, cell: vec3i) -> f32 {
  let brick = cell >> vec3u(shift);
  let local = cell & vec3i((1 << shift) - 1);
  let page = fireProPage(pages, box, brick);
  if page >= 0 {
    return textureLoad(smokeDensity, poolOrigin(page, shift) + local, 0).x;
  }
  for (var m = 1u; m < 8u; m++) {
    let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
    if any(axes & (local != vec3i(0))) {
      continue;
    }
    let step = select(vec3i(0), vec3i(1), axes);
    let previous = fireProPage(pages, box, brick - step);
    if previous >= 0 {
      return textureLoad(smokeDensity,
        poolOrigin(previous, shift) + local + step * (1 << shift), 0).x;
    }
  }
  return 0.0;
}

// A workgroup per active brick classifies its 8-cell blocks, each with the cubic
// reconstruction's two-cell halo: x is 1 when any cell has flame glow above the camera's
// .002 cutoff, y conservatively bounds smoke + glow on their respective grids. Positive reconstruction
// weights cannot exceed the largest donor, and glow only rises with lifetime, so the
// camera skips blocks with
// y = 0 and reconstructs flame-free fields (x = 0) with trilinear filtering. The halo
// is whole 2³-cell units, which never straddle bricks. Workgroup memory starts zeroed.
var<workgroup> occupied: atomic<u32>;
var<workgroup> smokeMaximum: atomic<u32>;
var<workgroup> glowMaximum: atomic<u32>;
@compute @workgroup_size(64)
fn buildOccupancy(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32
) {
  var work: ListedBrick;
  // Every lane takes the same branch: the brick is the workgroup's.
  if !listed(group, &work) {
    return;
  }
  let shift = fieldShift();
  let smokeShift = renderSmokeShift();
  let coarse = smokeShift < shift;
  let ratio = f32(1u << (shift - smokeShift));
  let perSide = 1u << (shift - 3u);
  let p = u32(work.page);
  let slot = vec3u(p & 255u, (p >> 8u) & 255u, p >> 16u) * perSide;
  let first = work.brick * (1 << shift);
  for (var block = 0u; block < perSide * perSide * perSide; block++) {
    let b = vec3u(block % perSide, (block / perSide) % perSide, block / (perSide * perSide));
    // Six units of 2³ cells per axis: the block's four and one each side.
    let origin = first + vec3i(b * 8u) - 2;
    var laneGlow = 0.0;
    for (var unit = lane; unit < 216u; unit += 64u) {
      // Both flags set: nothing left to find.
      if atomicLoad(&occupied) == 3u {
        break;
      }
      let cell = origin + vec3i(vec3u(unit % 6u, (unit / 6u) % 6u, unit / 36u) * 2u);
      let page = fireProPage(pages, work.box, cell >> vec3u(shift));
      if page < 0 {
        continue;
      }
      let texel = poolOrigin(page, shift) + (cell & vec3i((1 << shift) - 1));
      var flags = 0u;
      for (var corner = 0u; corner < 8u; corner++) {
        let offset = vec3i(vec3u(corner, corner >> 1u, corner >> 2u) & vec3u(1u));
        let f = textureLoad(fields, texel + offset, 0);
        let glow = fireProGlow(f.z);
        let smoke = select(f.x, 0.0, coarse);
        flags |= select(0u, 1u, glow > .002) | select(0u, 2u, smoke + glow > .002);
        if coarse {
          laneGlow = max(laneGlow, glow);
        }
      }
      if flags != 0u {
        atomicOr(&occupied, flags);
      }
    }
    if coarse {
      if laneGlow > 0.0 {
        atomicMax(&glowMaximum, bitcast<u32>(laneGlow));
      }
      // Every trilinear smoke donor in the block/halo. A sum of independent maxima
      // is conservative even when smoke and flame peak at different positions.
      let lo = vec3i(floor(vec3f(origin) / ratio - .5));
      let hi = vec3i(floor(vec3f(origin + 12) / ratio - .5)) + 1;
      let size = vec3u(hi - lo + 1);
      let smokeSide = 1 << smokeShift;
      let boxLo = work.box.origin * smokeSide;
      let boxHi = (work.box.origin + work.box.size) * smokeSide - 1;
      var laneSmoke = 0.0;
      for (var i = lane; i < size.x * size.y * size.z; i += 64u) {
        let cell = lo + vec3i(vec3u(i % size.x, (i / size.x) % size.y, i / (size.x * size.y)));
        let density = cachedSmokeCell(work.box, smokeShift, clamp(cell, boxLo, boxHi));
        laneSmoke = max(laneSmoke, density);
      }
      if laneSmoke > 0.0 {
        atomicMax(&smokeMaximum, bitcast<u32>(laneSmoke));
      }
    }
    workgroupBarrier();
    if lane == 0u {
      var flags = atomicLoad(&occupied);
      let maxSmoke = bitcast<f32>(atomicLoad(&smokeMaximum));
      let maxGlow = bitcast<f32>(atomicLoad(&glowMaximum));
      if coarse && maxSmoke + maxGlow > .002 {
        flags |= 2u;
      }
      textureStore(occupancyOutput, slot + b, vec4f(f32(flags & 1u), f32(flags >> 1u), 0, 0));
      atomicStore(&occupied, 0u);
      atomicStore(&smokeMaximum, 0u);
      atomicStore(&glowMaximum, 0u);
    }
    workgroupBarrier();
  }
}

// How far `p` lies from the box's faces along `direction`.
fn exitDistance(box: FireProBox, p: vec3f, direction: vec3f) -> f32 {
  let size = brickSize();
  let edge = select(vec3f(box.origin), vec3f(box.origin + box.size), direction > vec3f(0)) * size;
  // A parallel axis cannot limit the exit distance. Substituting a positive
  // epsilon made axis-aligned lights exit immediately through the wrong face.
  let moving = abs(direction) > vec3f(.000001);
  let distances = select(vec3f(1e30), (edge - p) / select(vec3f(1), direction, moving), moving);
  return max(0.0, min(distances.x, min(distances.y, distances.z)));
}

// Incident light at the center of a lighting texel at world point `p`: the sun's
// transmittance through the box to its edge, and a coarse isotropic glow from the fire
// around it.
fn incidentLight(box: FireProBox, p: vec3f) -> vec4f {
  let localField = sampleWorld(box, p);
  if localField.x + fireProGlow(localField.z) < .00001 {
    return vec4f(0, 0, 0, 1);
  }
  let sun = normalize(u.light.xyz);
  let count = u32(u.grid.z);
  let ds = exitDistance(box, p, sun) / f32(count);
  var opticalDepth = 0.0;
  if u.light.w > 0.0 {
    for (var i = 0u; i < count; i++) {
      let f = sampleWorld(box, p + sun * ((f32(i) + .5) * ds));
      opticalDepth += (max(0.0, f.x) * u.optical.z + fireProFlameAbsorption(f, u.grid.w)) * ds;
      if opticalDepth > 9.0 {
        break;
      }
    }
  }
  // Coarse isotropic incident fire radiance. Six rotated directions, growing
  // integration intervals, and Beer attenuation; this is a local approximation.
  let directions = array<vec3f, 6>(
    vec3f(.36, .48, -.8), vec3f(-.36, -.48, .8),
    vec3f(-.8, .6, 0), vec3f(.8, -.6, 0),
    vec3f(.48, .64, .6), vec3f(-.48, -.64, -.6)
  );
  var glow = vec3f(0);
  if u.optical.y > 0.0 && localField.x > .00001 {
    for (var direction = 0u; direction < 6u; direction++) {
      var t = 0.0;
      var interval = .075;
      var transmission = 1.0;
      for (var step = 0u; step < 6u; step++) {
        let f = sampleWorld(box, p + directions[direction] * (t + interval * .5));
        let extinction = fireProExtinction(f, u.optical, u.grid.w);
        let attenuation = exp(-extinction * interval);
        glow += transmission * emissionAt(f) * integratedTransmission(extinction,
          interval,
          attenuation) / 6.0;
        transmission *= attenuation;
        t += interval;
        interval *= 1.65;
      }
    }
  }
  return vec4f(glow, exp(-opticalDepth));
}

// Light a brick's slot of the selected lighting lattice, with an apron
// holding the first texels of the bricks after it, so the camera's filtered samples stay
// in the slot (volume-bricks.wgsl fireProLight). The workgroup's lanes share the texels.
// A build of every brick lights each texel once: a brick's first texels also fill the
// aprons of the bricks before it, and its apron takes only texels of bricks without a slot.
fn lightSlot(work: ListedBrick, lane: u32, every: bool) {
  let shift = fireProPoolShift(textureDimensions(fields).x, textureDimensions(lightOutput).x, fieldShift());
  let side = 1u << shift;
  let edge = side + 1u;
  let origin = poolOrigin(work.page, shift);
  let first = work.brick * i32(side);
  let texel = u.grid.x * f32(1u << (fieldShift() - shift));
  for (var index = lane; index < edge * edge * edge; index += 64u) {
    let local = vec3u(index % edge, (index / edge) % edge, index / (edge * edge));
    let past = vec3i(local == vec3u(side));
    if every && any(past != vec3i(0)) && fireProPage(pages, work.box, work.brick + past) >= 0 {
      continue;
    }
    let light = incidentLight(work.box, (vec3f(first + vec3i(local)) + .5) * texel);
    textureStore(lightOutput, origin + vec3i(local), light);
    if !every || any(past != vec3i(0)) {
      continue;
    }
    for (var m = 1u; m < 8u; m++) {
      let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
      if any(axes & (local != vec3u(0))) {
        continue;
      }
      let step = select(vec3i(0), vec3i(1), axes);
      let page = fireProPage(pages, work.box, work.brick - step);
      if page >= 0 {
        textureStore(lightOutput, poolOrigin(page, shift) + vec3i(local) + step * i32(side), light);
      }
    }
  }
}

// A workgroup per active brick.
@compute @workgroup_size(64)
fn buildLighting(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32
) {
  var work: ListedBrick;
  if listed(group, &work) {
    lightSlot(work, lane, true);
  }
}

// The bricks the latest step zeroed, a workgroup each: a count, then each brick's id and
// its page, the top bit marking a slot given that step (fluid-allocation.wgsl zeroBrick).
@group(0) @binding(6) var<storage, read> zeroedBricks: array<u32>;
// Light the slots the latest step gave out at once: until the next full build they would
// hold the light of the bricks that had them before.
@compute @workgroup_size(64)
fn lightFresh(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32
) {
  let index = listedBrick(group);
  if index >= zeroedBricks[0] || (zeroedBricks[2u + 2u * index] & 0x80000000u) == 0u {
    return;
  }
  var work: ListedBrick;
  if slotted(zeroedBricks[1u + 2u * index], &work) {
    lightSlot(work, lane, false);
  }
}
