// Corrected scalar evolution: emission, combustion, cooling, dissipation, and expansion.
// Assembled with the other fluid modules by ShaderSources.ts.
struct FieldEvolution {
  value: vec4f,
  expansion: f32,
  injected: bool
};

fn evolvedFields(id: vec3i, trace: ptr<function, CellTrace>) -> FieldEvolution {
  let position = center(id);
  if solidAt(position) {
    return FieldEvolution(vec4f(0), 0, false);
  }
  // The corrected scalars, rounded to half precision as the field textures held them.
  let transported = truncateHalf(correctedScalars(id, trace));
  let timeStep = u.grid.w;
  var lifetime = transported.z;
  var sourceProducts = vec2f(0);
  var injectedFuel = 0.0;
  let list = tileList(EMITTERS);
  for (var i = 0u; i < list.y; i++) {
    let e = emitters[tileLists[list.x + i]];
    let weight = emitterWeight(e, position);
    if weight == 0.0 {
      continue;
    }
    let variation = burstVariation(e, position); // heat, remaining lifetime
    // Max replenishment is a prescribed lifetime, not fuel concentration.
    lifetime = max(lifetime, e.rates.x * weight * variation.y);
    sourceProducts += vec2f(e.rates.z, e.rates.y * variation.x) * weight * timeStep;
    injectedFuel += e.variation.z * weight * timeStep;
  }
  let mesh = meshSourceAt(position);
  lifetime = max(lifetime, mesh.scalar.x);
  sourceProducts += vec2f(mesh.scalar.z, mesh.scalar.y) * timeStep;
  injectedFuel += mesh.scalar.w * timeStep;
  // Fuel in a voxel hot enough to ignite it is flame: its amount, up to 1, is the flame's
  // lifetime, as an emitter's flame setting is, and the flame produces heat, smoke and
  // expansion. It burns away at the burn rate, and the gas expands as it burns: the
  // expansion rate for each unit of fuel burned a second. Without fuel, sources add none.
  var remaining = 0.0;
  var burned = 0.0;
  if u.fuel.z > .5 {
    remaining = transported.w + injectedFuel;
    if transported.y + sourceProducts.y >= u.fuel.x {
      lifetime = max(lifetime, min(remaining, 1.0));
      let unburned = remaining * exp(-u.fuel.y * timeStep);
      burned = remaining - unburned;
      remaining = unburned;
    }
  }
  let flame = evaluateFlame(clamp(lifetime, 0.0, 1.0));
  let expansion = flame.products.z + flameParams.extra.x * burned / timeStep;
  let losses = vec2f(u.dynamics.x, u.forces.y);
  // Symmetric exponential loss around this step's additive outputs. Exact
  // piecewise-linear flame integrals carry no extra dt.
  let densityAndHeat = transported.xy * exp(-losses * timeStep) + (sourceProducts + flame.products.xy) * exp(-losses * timeStep * .5);
  return FieldEvolution(vec4f(max(densityAndHeat, vec2f(0)), flame.lifetime, remaining),
    expansion,
    any(sourceProducts > vec2f(0)) || lifetime > transported.z || injectedFuel > 0.0);
}

@compute @workgroup_size(8, 4, 4)
fn evolveFields(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = listedCell(group, local, vec3u(8, 4, 4));
  var support = 0u;
  if cell.listed {
    var trace = cellTrace(cell.id);
    let result = evolvedFields(cell.id, &trace);
    let value = clearDormant(result.value, result.injected);
    storeVector(cell.id, value);
    if result.expansion != 0.0 {
      noteExpansion();
    }
    if result.expansion != 0.0 || mayExpand(cell.brick) {
      storeExpansion(cell.id, result.expansion);
    }
    if holdsContent(value) {
      support = contentSupport(cell.id, traceVelocity(&trace));
    }
  }
  recordContent(cell.brick, cell.listed, support, lane);
}
