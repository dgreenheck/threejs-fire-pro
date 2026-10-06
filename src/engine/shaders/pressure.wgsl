enable f16;
// Geometric multigrid for the pressure on the velocity grid: weighted Jacobi smoothing,
// residual restriction and trilinear correction. Levels are stored only where there are
// bricks. Each brick holds successive levels down to 2³ velocity cells. The next
// three levels hold 4³, 2³ and one cell in each tile of 4x4x4 bricks (tiles.wgsl).
// Level cells span `2^level` finest velocity cells on each axis.
// Values are half precision; loads promote to f32 and only stores round.
// Per dispatch: the input, output and coarse levels and the bottom solve's iterations;
// the brick ids and how the brick list is read (bricks.w: words per entry, or 0 for every
// brick id); the tile slots.
struct Dimensions {
  levels: vec4u,
  bricks: vec4u,
  tiles: vec4u
};

@group(0) @binding(6) var<uniform> dimensions: Dimensions;
fn inputLevel() -> u32 {
  return dimensions.levels.x;
}

fn outputLevel() -> u32 {
  return dimensions.levels.y;
}

fn coarseLevel() -> u32 {
  return dimensions.levels.z;
}

// The fluid solver's parameters for the velocity grid (fluid-common.wgsl).
struct Params {
  grid: vec4f,
  forces: vec4f,
  dynamics: vec4f,
  noise: vec4f,
  counts: vec4f,
  velocityGrid: vec4f,
  fieldGrid: vec4f,
  pool: vec4f,
  bricks: vec4f,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> pressure: array<f16>;
@group(0) @binding(2) var<storage, read> rhs: array<f16>;
@group(0) @binding(3) var<storage, read> coarse: array<f16>;
@group(0) @binding(4) var<storage, read_write> outputPressure: array<f16>;
@group(0) @binding(11) var<storage, read_write> coarseFirst: array<f16>;
@group(0) @binding(8) var<storage, read> topology: array<u32>;
@group(0) @binding(9) var<storage, read> coarseTopology: array<u32>;
@group(0) @binding(10) var<storage, read_write> outputTopology: array<u32>;
// The listed bricks, or the bricks zeroed this step (fluid-zeroing.wgsl). Pages and whether bricks
// compute come from the tile directory (tiles.wgsl).
@group(0) @binding(12) var<storage, read> brickList: array<u32>;
// Finest velocity cells per brick side, as a power of two.
fn velocityBrickShift() -> u32 {
  return u32(u.velocityGrid.w);
}

fn levelShift(level: u32) -> u32 {
  if level < velocityBrickShift() {
    return velocityBrickShift() - level;
  }
  return velocityBrickShift() + 2u - level;
}

fn levelSide(level: u32) -> u32 {
  return 1u << levelShift(level);
}

// Where a level stores cell `p`: in a brick slot until one cell covers a whole brick,
// then in a tile. -1 without storage.
fn cellAddress(p: vec3i, level: u32) -> i32 {
  let shift = levelShift(level);
  let side = 1u << shift;
  let local = vec3u(p & vec3i(i32(side) - 1));
  var owner: u32;
  if level < velocityBrickShift() {
    let page = brickPage(p >> vec3u(shift));
    if page < 0 {
      return -1;
    }
    owner = poolSlot(page, u.pool.xy);
  } else {
    let tile = tileAt(p >> vec3u(shift));
    if tile < 0 {
      return -1;
    }
    owner = u32(tile);
  }
  return i32(owner * side * side * side + (local.z * side + local.y) * side + local.x);
}

fn brickActive(b: vec3i) -> bool {
  return entryComputes(brickEntry(b));
}

// Pressure unknowns. A finest-level cell needs its own brick and the bricks of its
// negative neighbors active, so projection updates all six of its faces. A coarse cell
// is an unknown only when every finest cell under it is one, which keeps each level's
// open boundary where the finest level has it (McAdams et al. 2010). All other cells
// hold zero pressure, an open boundary for their neighbors. Below the ground is solid, so
// no brick there needs to compute.
fn activeCell(p: vec3i, level: u32) -> bool {
  // A tile level's cell covers whole bricks: each one's record says whether all its cells are.
  if level >= velocityBrickShift() {
    let n = 1 << (level - velocityBrickShift());
    for (var z = 0; z < n; z++) {
      for (var y = 0; y < n; y++) {
        for (var x = 0; x < n; x++) {
          let id = brickId(p * n + vec3i(x, y, z));
          if id < 0 || entryById(u32(id)) < 0 || recordWord(u32(id), BRICK_PRESSURE_FLAG) == 0 {
            return false;
          }
        }
      }
    }
    return true;
  }
  let span = 1 << level;
  var lo = (p * span - 1) >> vec3u(velocityBrickShift());
  if u.counts.w > .5 {
    lo.y = max(lo.y, 0);
  }
  let hi = (p * span + span - 1) >> vec3u(velocityBrickShift());
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        if !brickActive(vec3i(x, y, z)) {
          return false;
        }
      }
    }
  }
  return true;
}

// A brick has a slot only while it computes (fluid-allocation.wgsl allocateBricks).
fn readPressure(p: vec3i) -> f32 {
  let a = cellAddress(p, inputLevel());
  if a < 0 {
    return 0.0;
  }
  return f32(pressure[a]);
}

fn readRhs(p: vec3i) -> f32 {
  let a = cellAddress(p, inputLevel());
  if a < 0 {
    return 0.0;
  }
  return f32(rhs[a]);
}

fn readCoarse(p: vec3i) -> f32 {
  let a = cellAddress(p, coarseLevel());
  if a < 0 {
    return 0.0;
  }
  return f32(coarse[a]);
}

fn writePressure(p: vec3i, value: f32) {
  let a = cellAddress(p, outputLevel());
  if a >= 0 {
    outputPressure[a] = f16(value);
  }
}

// Stored flags are one byte per cell, four cells per word in storage order.
// Bits 0..5 mark sides that are not blocked and bit 6 marks a solid cell.
const STORED_SOLID = 64u;
fn storedFlags(word: u32, address: u32) -> u32 {
  return (word >> ((address & 3u) * 8u)) & 0xffu;
}

// Expanded flags: bits 0..5 are open sides and bit 10 is solid. An open side's neighbor
// may hold no storage; it reads zero pressure, like every cell that is not an unknown.
const SOLID = 1024u;
fn expandFlags(stored: u32) -> u32 {
  if (stored & STORED_SOLID) != 0u {
    return SOLID;
  }
  return stored & 63u;
}

fn flagsAt(p: vec3i) -> u32 {
  let a = cellAddress(p, inputLevel());
  if a < 0 {
    return SOLID;
  }
  return expandFlags(storedFlags(topology[u32(a) >> 2u], u32(a)));
}

fn coarseSolid(p: vec3i) -> bool {
  let a = cellAddress(p, coarseLevel());
  if a < 0 {
    return false;
  }
  return (storedFlags(coarseTopology[u32(a) >> 2u], u32(a)) & STORED_SOLID) != 0u;
}

const OFFSETS = array<vec3i, 6>(vec3i(1, 0, 0),
  vec3i(-1, 0, 0),
  vec3i(0, 1, 0),
  vec3i(0, -1, 0),
  vec3i(0, 0, 1),
  vec3i(0, 0, -1));
// Whether a level's cell is solid: its center's field cell, as the fluid solver tests it.
fn blockedAt(p: vec3i, level: u32) -> bool {
  return solidAt((vec3f(p) + .5) * u.grid.xyz * f32(1u << level));
}

// Exterior colliders are tested before the ambient outlet, just like the operator.
fn cellTopology(p: vec3i, level: u32) -> u32 {
  if blockedAt(p, level) {
    return STORED_SOLID;
  }
  var flags = 0u;
  for (var side = 0u; side < 6u; side++) {
    if !blockedAt(p + OFFSETS[side], level) {
      flags |= 1u << side;
    }
  }
  return flags;
}

// The brick whose slot levels a topology dispatch builds, and its slot: from the list,
// or every brick id; the slot is -1 without one. A brick freed this step has none, and
// may have passed its slot to a fresh brick on the same list.
struct TopologyBrick {
  brick: vec3i,
  slot: i32
};

fn topologyBrick(group: vec3u) -> TopologyBrick {
  let stride = dimensions.bricks.w;
  var id: u32;
  if stride == 0u {
    id = group.x + group.y * BRICK_LIST_SPLIT;
    if id >= dimensions.bricks.x || tileRecord(id >> 6u).w == 0 {
      return TopologyBrick(vec3i(0), -1);
    }
  } else {
    let index = listedBrick(group);
    if index >= brickList[0] {
      return TopologyBrick(vec3i(0), -1);
    }
    id = brickList[1u + stride * index];
  }
  let page = entryPage(entryById(id));
  if page < 0 {
    return TopologyBrick(vec3i(0), -1);
  }
  return TopologyBrick(brickOf(id), i32(poolSlot(page, u.pool.xy)));
}

// One workgroup builds a brick's flags at the output level, a whole word per invocation.
@compute @workgroup_size(64)
fn buildBrickTopology(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let brick = topologyBrick(group);
  if brick.slot < 0 {
    return;
  }
  let level = outputLevel();
  let side = levelSide(level);
  let cells = side * side * side;
  for (var word = lane; word * 4u < cells; word += 64u) {
    var packed = 0u;
    for (var k = 0u; k < 4u; k++) {
      let offset = word * 4u + k;
      let local = vec3u(offset % side, (offset / side) % side, offset / (side * side));
      packed |= cellTopology(brick.brick * i32(side) + vec3i(local), level) << (k * 8u);
    }
    outputTopology[u32(brick.slot) * cells / 4u + word] = packed;
  }
}

// Every tile slot's flags at the output level, a whole word per invocation.
@compute @workgroup_size(256)
fn buildTileTopology(@builtin(global_invocation_id) id: vec3u) {
  let level = outputLevel();
  let side = levelSide(level);
  let cells = side * side * side;
  let total = dimensions.tiles.x * cells;
  if id.x * 4u >= total {
    return;
  }
  var packed = 0u;
  for (var k = 0u; k < 4u; k++) {
    let address = id.x * 4u + k;
    if address >= total {
      break;
    }
    let tile = tileRecord(address / cells);
    if tile.w == 0 {
      continue;
    }
    let offset = address % cells;
    let local = vec3u(offset % side, (offset / side) % side, offset / (side * side));
    packed |= cellTopology(tile.xyz * i32(side) + vec3i(local), level) << (k * 8u);
  }
  outputTopology[id.x] = packed;
}

fn axisDiagonal(flags: u32, side: u32) -> f32 {
  return f32((flags >> side) & 1u) + f32((flags >> (side + 1u)) & 1u);
}

fn axisWeights() -> vec3f {
  let inverseSpacing = 1.0 / u.grid.xyz;
  let reference = 1.0 / max(inverseSpacing.x, max(inverseSpacing.y, inverseSpacing.z));
  let normalized = reference * inverseSpacing;
  return normalized * normalized;
}

// Where a level keeps a cell that has storage, found once: its address, the first address
// of its brick or tile, and its place there. Neighbors in the same brick or tile are then
// read without another page lookup, and without checking the brick computes: callers
// reach them only from an unknown, whose brick computes, or on the tile levels.
struct PressureCellAddress {
  address: u32,
  origin: u32,
  local: vec3u,
  side: u32
};

fn pressureCellAddress(p: vec3i, level: u32) -> PressureCellAddress {
  let side = levelSide(level);
  let local = vec3u(p & vec3i(i32(side) - 1));
  let address = u32(cellAddress(p, level));
  return PressureCellAddress(address,
    address - (local.z * side + local.y) * side - local.x,
    local,
    side);
}

fn nearPressure(here: PressureCellAddress, p: vec3i, side: u32, level: u32) -> f32 {
  let q = vec3i(here.local) + OFFSETS[side];
  if all(q >= vec3i(0)) && all(q < vec3i(i32(here.side))) {
    let c = vec3u(q);
    return f32(pressure[here.origin + (c.z * here.side + c.y) * here.side + c.x]);
  }
  return readPressure(p + OFFSETS[side]);
}

fn flagsHere(here: PressureCellAddress) -> u32 {
  return expandFlags(storedFlags(topology[here.address >> 2u], here.address));
}

// Whether a cell with storage is an unknown. In a brick that computes, a cell past the
// first layer on every axis needs no other brick (activeCell).
fn unknownHere(here: PressureCellAddress, p: vec3i, level: u32, computing: bool) -> bool {
  if computing && level < velocityBrickShift() && all(here.local > vec3u(0)) {
    return true;
  }
  return activeCell(p, level);
}

fn nearValue(here: PressureCellAddress, p: vec3i, flags: u32, side: u32, level: u32) -> f32 {
  if (flags & (1u << side)) == 0u {
    return 0;
  }
  return nearPressure(here, p, side, level);
}

// The cached flags preserve zero-flux solids and half-cell ambient outlets.
// The diagonal uses per-axis cell spacing.
fn stencil(here: PressureCellAddress, p: vec3i, flags: u32, level: u32) -> vec2f {
  let weights = axisWeights();
  let sum = weights.x * (nearValue(here,
      p,
      flags,
      0u,
      level) + nearValue(here,
      p,
      flags,
      1u,
      level)) + weights.y * (nearValue(here,
      p,
      flags,
      2u,
      level) + nearValue(here,
      p,
      flags,
      3u,
      level)) + weights.z * (nearValue(here,
      p,
      flags,
      4u,
      level) + nearValue(here,
      p,
      flags,
      5u,
      level));
  let diagonal = weights.x * axisDiagonal(flags,
    0u) + weights.y * axisDiagonal(flags,
    2u) + weights.z * axisDiagonal(flags,
    4u);
  return vec2f(sum, diagonal);
}

// Cells of the dispatch. Level 0 uses 128-cell chunks; coarser brick levels use
// 64-cell chunks, packing eight 2³-cell bricks into a group. Tile levels use one
// workgroup per tile slot. `computing` marks dispatches over computing bricks. The
// brick or tile becomes home (tiles.wgsl); a workgroup inside one brick shares its record.
// Every lane must call these, before any return.
struct LevelCell {
  id: vec3i,
  valid: bool
};

// Listed brick `index` becomes home, if listed; `whole` when it fills the workgroup.
fn enterListed(index: u32, lane: u32, whole: bool) -> bool {
  let listed = index < brickList[0];
  var id = 0u;
  if listed {
    id = brickList[index + 1u];
  }
  loadHomeRecord(id, lane, listed && whole);
  if !listed {
    return false;
  }
  enterBrick(id, true);
  home.inWorkgroup = whole;
  return true;
}

fn fineCell(group: vec3u, local: vec3u) -> LevelCell {
  let lane = (local.z * 4u + local.y) * 8u + local.x;
  if !enterListed(listedBrick(group), lane, true) {
    return LevelCell(vec3i(0), false);
  }
  let side = levelSide(0u);
  let index = group.x * 128u + lane;
  let offset = vec3u(index % side, (index / side) % side, index / (side * side));
  return LevelCell(home.brick * i32(side) + vec3i(offset), index < side * side * side);
}

fn brickCell(group: vec3u, lane: u32, level: u32) -> LevelCell {
  let side = levelSide(level);
  let cells = side * side * side;
  let whole = cells >= 64u;
  var index = listedBrick(group);
  var offset = group.x * 64u + lane;
  if !whole {
    index = index * 8u + lane / 8u;
    offset = lane % 8u;
  }
  if !enterListed(index, lane, whole) {
    return LevelCell(vec3i(0), false);
  }
  let local = vec3u(offset % side, (offset / side) % side, offset / (side * side));
  return LevelCell(home.brick * i32(side) + vec3i(local), offset < cells);
}

fn tileCell(group: vec3u, lane: u32, level: u32) -> LevelCell {
  let side = levelSide(level);
  if lane >= side * side * side || tileRecord(group.x).w == 0 {
    return LevelCell(vec3i(0), false);
  }
  enterTile(group.x);
  let local = vec3u(lane % side, (lane / side) % side, lane / (side * side));
  return LevelCell(home.tileCoord * i32(side) + vec3i(local), true);
}

fn relaxCell(p: vec3i, computing: bool) {
  let level = outputLevel();
  let here = pressureCellAddress(p, level);
  let flags = flagsHere(here);
  if (flags & SOLID) != 0u || !unknownHere(here, p, level, computing) {
    outputPressure[here.address] = 0h;
    return;
  }
  let coefficients = stencil(here, p, flags, level);
  let candidate = (coefficients.x - f32(rhs[here.address])) / max(coefficients.y, 1e-12);
  outputPressure[here.address] = f16(mix(f32(pressure[here.address]), candidate, .8));
}

// The first finest-grid Jacobi iteration starts from zero every timestep. Its
// pressure neighbors and center are all zero, so initialize the iterate directly
// from the RHS and diagonal. This replaces a full-capacity reset and its following
// relaxation without changing the arithmetic or half-precision rounding.
@compute @workgroup_size(8, 4, 4)
fn initializePressure(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = fineCell(group, local);
  if !cell.valid {
    return;
  }
  let level = outputLevel();
  let here = pressureCellAddress(cell.id, level);
  let flags = flagsHere(here);
  var first = 0.0;
  if (flags & SOLID) == 0u && unknownHere(here, cell.id, level, true) {
    let weights = axisWeights();
    let diagonal = weights.x * axisDiagonal(flags,
      0u) + weights.y * axisDiagonal(flags,
      2u) + weights.z * axisDiagonal(flags,
      4u);
    first = mix(0.0, (0.0 - f32(rhs[here.address])) / max(diagonal, 1e-12), .8);
  }
  outputPressure[here.address] = f16(first);
}

@compute @workgroup_size(8, 4, 4)
fn relaxBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = fineCell(group, local);
  if cell.valid {
    relaxCell(cell.id, true);
  }
}

@compute @workgroup_size(64)
fn relaxCoarseBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = brickCell(group, lane, outputLevel());
  if cell.valid {
    relaxCell(cell.id, true);
  }
}

@compute @workgroup_size(64)
fn relaxTiles(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let cell = tileCell(group, lane, outputLevel());
  if cell.valid {
    relaxCell(cell.id, false);
  }
}

// The bottom level, one cell per tile: up to 256 tile slots fit this shared array, and one
// workgroup runs every iteration, a tile slot per lane. Every lane reaches both barriers,
// including lanes past the tiles and solid cells. Round each iterate just like the f16
// ping-pong storage.
var<workgroup> coarsePressure: array<f32, 256>;
fn sharedNeighbor(lane: u32, flags: u32, side: u32) -> f32 {
  if (flags & (1u << side)) == 0u {
    return 0.0;
  }
  let neighbor = tileNeighbor(lane, neighborIndex(OFFSETS[side]));
  if neighbor < 0 {
    return 0.0;
  }
  return coarsePressure[neighbor];
}

@compute @workgroup_size(256)
fn relaxCoarse(@builtin(local_invocation_index) lane: u32) {
  let valid = lane < dimensions.tiles.x && tileRecord(lane).w != 0;
  var p = vec3i(0);
  if valid {
    enterTile(lane);
    p = home.tileCoord;
  }
  var flags = SOLID;
  var source = 0.0;
  var value = 0.0;
  if valid && activeCell(p, outputLevel()) {
    flags = flagsAt(p);
    source = readRhs(p);
    value = readPressure(p);
  }
  coarsePressure[lane] = value;
  workgroupBarrier();
  for (var iteration = 0u; iteration < dimensions.levels.w; iteration++) {
    var next = 0.0;
    if valid && (flags & SOLID) == 0u {
      let weights = axisWeights();
      let sum = weights.x * (sharedNeighbor(lane,
          flags,
          0u) + sharedNeighbor(lane,
          flags,
          1u)) + weights.y * (sharedNeighbor(lane,
          flags,
          2u) + sharedNeighbor(lane,
          flags,
          3u)) + weights.z * (sharedNeighbor(lane,
          flags,
          4u) + sharedNeighbor(lane,
          flags,
          5u));
      let diagonal = weights.x * axisDiagonal(flags,
        0u) + weights.y * axisDiagonal(flags,
        2u) + weights.z * axisDiagonal(flags,
        4u);
      let candidate = (sum - source) / max(diagonal, 1e-12);
      next = quantizeToF16(mix(value, candidate, .8));
    }
    workgroupBarrier();
    coarsePressure[lane] = next;
    value = next;
    workgroupBarrier();
  }
  if valid {
    writePressure(p, value);
  }
}

fn restrictCell(id: vec3i, computing: bool) {
  let fine = inputLevel();
  var residual = 0.0;
  // A coarse cell is an unknown exactly when its eight fine cells are (activeCell).
  var everyUnknown = true;
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let p = id * 2 + vec3i(x, y, z);
        // On the brick levels the fine cells lie in the coarse cell's brick, which computes.
        var unknown = computing && fine < velocityBrickShift() && all((p & vec3i(i32(levelSide(fine)) - 1)) > vec3i(0));
        if !unknown {
          unknown = activeCell(p, fine);
        }
        everyUnknown = everyUnknown && unknown;
        if unknown {
          let here = pressureCellAddress(p, fine);
          let flags = flagsHere(here);
          if (flags & SOLID) == 0u {
            let coefficients = stencil(here, p, flags, fine);
            residual += f32(rhs[here.address]) - (coefficients.x - coefficients.y * f32(pressure[here.address]));
          }
        }
      }
    }
  }
  // Coarse cells are twice as wide: h_coarse^2 / h_fine^2 = 4.
  let rhs = residual * .5;
  let level = outputLevel();
  let out = pressureCellAddress(id, level);
  outputPressure[out.address] = f16(rhs);
  // The coarse level's first Jacobi iterate from a zero guess, as relaxCell computes it:
  // every neighbor holds zero, so only the cell's own right-hand side contributes.
  var first = 0.0;
  if everyUnknown {
    let flags = expandFlags(storedFlags(coarseTopology[out.address >> 2u], out.address));
    if (flags & SOLID) == 0u {
      let weights = axisWeights();
      let diagonal = weights.x * axisDiagonal(flags,
        0u) + weights.y * axisDiagonal(flags,
        2u) + weights.z * axisDiagonal(flags,
        4u);
      first = mix(0.0, (0.0 - f32(f16(rhs))) / max(diagonal, 1e-12), .8);
    }
  }
  coarseFirst[out.address] = f16(first);
}

@compute @workgroup_size(64)
fn restrictBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = brickCell(group, lane, outputLevel());
  if cell.valid {
    restrictCell(cell.id, true);
  }
}

@compute @workgroup_size(64)
fn restrictTiles(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = tileCell(group, lane, outputLevel());
  if cell.valid {
    restrictCell(cell.id, false);
  }
}

// Solid cells have no pressure unknown. Interpolate Neumann ghost values from
// adjacent fluid cells instead of treating stored solid zeros as a pressure sink. Below
// the ground, the first layer mirrors: no flux through the floor.
fn coarseValue(p: vec3i) -> f32 {
  let ground = select(-0x7fffffff, 0, u.counts.w > .5);
  let c = vec3i(p.x, max(p.y, ground), p.z);
  if !coarseSolid(c) {
    return readCoarse(c);
  }
  var sum = 0.0;
  var count = 0.0;
  for (var axis = 0u; axis < 3u; axis++) {
    for (var side = -1; side <= 1; side += 2) {
      var offset = vec3i(0);
      offset[axis] = side;
      var q = c + offset;
      q.y = max(q.y, ground);
      if !coarseSolid(q) {
        sum += readCoarse(q);
        count += 1.0;
      }
    }
  }
  return sum / max(count, 1.0);
}

// The coarse cells around a fine cell of a computing brick: on a brick level, those in the
// same brick are read without a page lookup.
struct CoarseStencil {
  origin: u32,
  corner: vec3i,
  side: u32,
  near: bool
};

fn coarsePlace(here: PressureCellAddress,
  p: vec3i,
  level: u32,
  computing: bool) -> CoarseStencil {
  if !computing || level + 1u >= velocityBrickShift() {
    return CoarseStencil(0u, vec3i(0), 0u, false);
  }
  let side = here.side / 2u;
  let slot = here.origin / (here.side * here.side * here.side);
  let corner = (p >> vec3u(countTrailingZeros(here.side))) * i32(side);
  return CoarseStencil(slot * side * side * side, corner, side, true);
}

fn coarseAt(home: CoarseStencil, c: vec3i) -> f32 {
  if home.near {
    let q = c - home.corner;
    if all(q >= vec3i(0)) && all(q < vec3i(i32(home.side))) {
      let l = vec3u(q);
      return f32(coarse[home.origin + (l.z * home.side + l.y) * home.side + l.x]);
    }
  }
  return readCoarse(c);
}

fn prolongateCell(p: vec3i, computing: bool) {
  let level = outputLevel();
  let here = pressureCellAddress(p, level);
  let flags = flagsHere(here);
  if (flags & SOLID) != 0u || !unknownHere(here, p, level, computing) {
    outputPressure[here.address] = 0h;
    return;
  }
  let q = (vec3f(p) + .5) * .5 - .5;
  let i = vec3i(floor(q));
  let f = fract(q);
  // Ordinary coarse cells need no ghost extrapolation or collider tests.
  // Keep the original interpolation order and the full boundary path.
  var correction: f32;
  if u.counts.y == 0 && (u.counts.w == 0 || i.y >= 0) {
    let home = coarsePlace(here, p, level, computing);
    correction = mix(mix(mix(coarseAt(home, i), coarseAt(home, i + vec3i(1, 0, 0)), f.x), mix(coarseAt(home, i + vec3i(0, 1, 0)), coarseAt(home, i + vec3i(1, 1, 0)), f.x), f.y),
      mix(mix(coarseAt(home, i + vec3i(0, 0, 1)), coarseAt(home, i + vec3i(1, 0, 1)), f.x),
        mix(coarseAt(home, i + vec3i(0, 1, 1)), coarseAt(home, i + vec3i(1, 1, 1)), f.x),
        f.y),
      f.z);
  } else {
    correction = mix(mix(mix(coarseValue(i), coarseValue(i + vec3i(1, 0, 0)), f.x), mix(coarseValue(i + vec3i(0, 1, 0)), coarseValue(i + vec3i(1, 1, 0)), f.x), f.y),
      mix(mix(coarseValue(i + vec3i(0, 0, 1)), coarseValue(i + vec3i(1, 0, 1)), f.x),
        mix(coarseValue(i + vec3i(0, 1, 1)), coarseValue(i + vec3i(1, 1, 1)), f.x),
        f.y),
      f.z);
  }
  outputPressure[here.address] = f16(f32(pressure[here.address]) + correction);
}

@compute @workgroup_size(8, 4, 4)
fn prolongateBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u) {
  let cell = fineCell(group, local);
  if cell.valid {
    prolongateCell(cell.id, true);
  }
}

@compute @workgroup_size(64)
fn prolongateCoarseBricks(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = brickCell(group, lane, outputLevel());
  if cell.valid {
    prolongateCell(cell.id, true);
  }
}

@compute @workgroup_size(64)
fn prolongateTiles(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let cell = tileCell(group, lane, outputLevel());
  if cell.valid {
    prolongateCell(cell.id, false);
  }
}
