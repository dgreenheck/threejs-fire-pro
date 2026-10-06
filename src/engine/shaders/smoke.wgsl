// Optional coarse smoke transport. Smoke is a cell-centered concentration, not a
// per-cell amount: source restriction averages fine-cell production by volume.
// Full-resolution smoke stays in the shared scalar transport path.
fn smokeVelocityAverage(p: vec3i) -> f32 {
  let ratio = max(1u, u32(round(u.velocityGrid.x / smokeSpacing().x)));
  if ratio == 1u {
    return sampleAt(smokeDensity, center(p), smokeSpacing()).x;
  }
  let first = vec3f(p) * f32(ratio);
  var density = 0.0;
  for (var z = 0u; z < ratio; z += 2u) {
    for (var y = 0u; y < ratio; y += 2u) {
      for (var x = 0u; x < ratio; x += 2u) {
        density += sampleCells(smokeDensity,
          first + vec3f(f32(x + 1u), f32(y + 1u), f32(z + 1u)),
          smokeSpacing()).x;
      }
    }
  }
  return density / f32(ratio * ratio * ratio / 8u);
}

// Exact box average using the texture unit to average each 2x2x2 fine-cell block.
fn smokeProduction(cell: vec3i) -> vec4f {
  let ratio = u32(u.noise.z);
  let first = vec3f(cell) * f32(ratio);
  var sum = vec4f(0);
  for (var z = 0u; z < ratio; z += 2u) {
    for (var y = 0u; y < ratio; y += 2u) {
      for (var x = 0u; x < ratio; x += 2u) {
        sum += sampleCells(fields,
          first + vec3f(f32(x + 1u), f32(y + 1u), f32(z + 1u)),
          u.fieldGrid.xyz);
      }
    }
  }
  return sum / f32(ratio * ratio * ratio / 8u);
}

fn smokeWork(group: vec3u, lane: u32) -> ListedCell {
  let index = listedBrick(group);
  let listed = index < brickList[0];
  var id = 0u;
  if listed {
    id = brickList[index + 1u];
  }
  loadHomeRecord(id, lane, listed);
  if !listed {
    return ListedCell(vec3i(0), 0u, false, false);
  }
  enterBrick(id, true);
  home.inWorkgroup = true;
  return ListedCell(home.brick * i32(brickSize()), id, true, true);
}

fn smokeCell(first: vec3i, index: u32) -> vec3i {
  let side = brickSize();
  return first + vec3i(vec3u(index % side, (index / side) % side, index / (side * side)));
}

fn smokeDonorBounds(base: vec3i) -> vec2f {
  let donors = blockAt(base, vec3i(1));
  if donors.mode == BLOCK_EMPTY {
    return vec2f(0);
  }
  var minimumDensity = 1e30;
  var maximumDensity = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let offset = vec3i(vec3u(k, k >> 1u, k >> 2u) & vec3u(1u));
    let value = blockLoad(smokeDensity, donors, base, offset).x;
    minimumDensity = min(minimumDensity, value);
    maximumDensity = max(maximumDensity, value);
  }
  return vec2f(minimumDensity, maximumDensity);
}

// Rendering smooths into the predictor, including its sparse-domain aprons.
// Neighbor invocations restore populated aprons; these cells clear absent ones.
fn clearMissingSmokePredictorApron(cell: vec3i) {
  if u.fuel.w < .5 {
    return;
  }
  let last = (cell & cellMask()) == cellMask();
  if !any(last) {
    return;
  }
  let brick = cell >> vec3u(gridShift());
  let texel = slotTexel(cell, cellPage(cell));
  for (var m = 1u; m < 8u; m++) {
    let axes = (vec3u(m) & vec3u(1u, 2u, 4u)) != vec3u(0);
    if any(axes & !last) {
      continue;
    }
    let step = select(vec3i(0), vec3i(1), axes);
    if brickPage(brick + step) < 0 {
      textureStore(outputExpansion, texel + step, vec4f(0));
    }
  }
}

@compute @workgroup_size(64)
fn advectSmoke(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let work = smokeWork(group, lane);
  if !work.listed {
    return;
  }
  let side = brickSize();
  for (var index = lane; index < side * side * side; index += 64u) {
    let id = smokeCell(work.id, index);
    var trace = cellTrace(id);
    var density = 0.0;
    if !solidAt(center(id)) {
      let departure = traceBack(&trace);
      density = sampleAt(smokeDensity, departure, u.grid.xyz).x;
      if SCALAR_MACCORMACK {
        let base = vec3i(floor(departure / u.grid.xyz - .5));
        let bounds = smokeDonorBounds(base);
        textureStore(outputScalarDonors, donorTexel(id, 0u),
          vec4u(pack2x16float(bounds), 0u, 0u, 0u));
      }
    }
    storeExpansion(id, density);
    clearMissingSmokePredictorApron(id);
  }
}

@compute @workgroup_size(64)
fn evolveSmoke(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let work = smokeWork(group, lane);
  let side = brickSize();
  var support = 0u;
  if work.listed {
    for (var index = lane; index < side * side * side; index += 64u) {
      let id = smokeCell(work.id, index);
      var trace = cellTrace(id);
      var density = 0.0;
      if !solidAt(center(id)) {
        let predictor = load(auxiliary, id).x;
        var bounded = predictor;
        if SCALAR_MACCORMACK {
          let bounds = unpack2x16float(textureLoad(donorCells, donorTexel(id, 0u), 0).x);
          let minimumDensity = bounds.x;
          let maximumDensity = bounds.y;
          // A singleton donor range can only retain that value or fall back to the
          // predictor. When both are equal, reverse tracing cannot change the result.
          // Requiring predictor equality also preserves fresh-apron interpolation and
          // half-store rounding; no density cutoff or approximation is introduced.
          if minimumDensity != maximumDensity || predictor != minimumDensity {
            let original = load(smokeDensity, id).x;
            let reversedDensity = sampleAt(auxiliary, traceForward(&trace), u.grid.xyz).x;
            let corrected = predictor + .5 * (original - reversedDensity);
            bounded = select(corrected,
              predictor,
              corrected < minimumDensity || corrected > maximumDensity);
          }
        }
        let transported = truncateHalf(vec4f(bounded, 0, 0, 0)).x;
        // Fine evolution already applied half-step dissipation to source production.
        let produced = smokeProduction(id);
        density = max(0.0, transported * exp(-u.dynamics.x * u.grid.w) + produced.x);
        if produced.x == 0.0 && produced.z == 0.0 && max(density,
          max(produced.y, produced.w)) < DORMANT_CUTOFF {
          density = 0.0;
        }
        if density > u.dynamics.z {
          support |= contentSupport(id, traceVelocity(&trace));
        }
      }
      storeExpansion(id, density);
    }
  }
  recordContent(work.brick, work.listed, support, lane);
}

@group(0) @binding(74) var secondSmoke: texture_storage_3d<r16float, write>;
@group(0) @binding(75) var thirdSmoke: texture_storage_3d<r16float, write>;
fn zeroSmokeTexel(texel: vec3i) {
  textureStore(outputExpansion, texel, vec4f(0));
  textureStore(secondSmoke, texel, vec4f(0));
  textureStore(thirdSmoke, texel, vec4f(0));
}

@compute @workgroup_size(64)
fn zeroSmokeBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let index = listedBrick(group);
  if index >= zeroList[0] {
    return;
  }
  let brick = zeroList[1u + 2u * index];
  let payload = zeroList[2u + 2u * index];
  let page = i32(payload & 0xffffffu);
  enterBrick(brick, true);
  let side = brickSize();
  let first = home.brick * i32(side);
  for (var i = lane; i < side * side * side; i += 64u) {
    let id = smokeCell(first, i);
    zeroSmokeTexel(slotTexel(id, page));
    if apronCopies(id) {
      for (var m = 1u; m < 8u; m++) {
        let step = apronStep(id, m);
        if all(step == vec3i(0)) {
          continue;
        }
        let neighbor = brickPage(home.brick - step);
        if neighbor >= 0 {
          zeroSmokeTexel(slotTexel(id, neighbor) + step * i32(side));
        }
      }
    }
    if (payload & FRESH_SLOT) != 0u {
      let cell = ZeroCell(id, page, true, true);
      for (var m = 1u; m < 8u; m++) {
        let texel = apronTexel(cell, m);
        if texel.w != 0 {
          zeroSmokeTexel(texel.xyz);
        }
      }
    }
  }
}
