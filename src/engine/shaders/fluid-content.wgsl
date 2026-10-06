// Content flags and motion support that keep sparse bricks active.
// Assembled with the other fluid modules by ShaderSources.ts.
// Content tracking and sparse brick allocation
// Cells that still hold flame, smoke or fuel above the cutoff. Their bricks
// request padding from the occupied cells, not the entire brick. Their tiles stay
// with a tile of margin. Each
// workgroup reduces first, so the global atomics see one update per group.
// Flags per brick: CONTENT, this step's evolution found content; EXPANDS, it produced a
// nonzero expansion rate; MAY_EXPAND, the brick's slot may still hold one, so evolution
// stores it even where it is zero.
const CONTENT: u32 = 1u;
const EXPANDS: u32 = 2u;
const MAY_EXPAND: u32 = 4u;
// Bits 3..29 request one of the 27 neighboring bricks (including self).
// The flags of every brick id; then per tile slot the bricks this step found content in,
// for the host's tile directory and the renderer's boxes: bits x, 4 + y and 8 + z for a
// brick at (x, y, z) in its tile.
@group(0) @binding(36) var<storage, read_write> activity: array<atomic<u32>>;
// Workgroup memory starts zeroed.
var<workgroup> groupBusy: atomic<u32>;
var<workgroup> groupExpands: atomic<u32>;
fn holdsContent(f: vec4f) -> bool {
  return max(max(select(f.x, 0.0, coarseSmoke()), f.z), f.w) > u.dynamics.z;
}

// Two velocity cells cover local interpolation/derivative stencils. Add a step of
// motion, up to the previous full-brick margin. The pressure solve uses the same
// open boundary, now closer to the material. No field value or cutoff is changed.
fn contentSupport(id: vec3i, velocity: vec3f) -> u32 {
  let ratio = u.velocityGrid.xyz / u.grid.xyz;
  let side = vec3f(f32(brickSize()));
  let local = vec3f(id & vec3i(vec3u(side) - 1u)) + .5;
  let pad = min(side, max(vec3f(2), vec3f(2) * ratio) + abs(velocity) * u.grid.w / u.grid.xyz);
  let low = local < pad;
  let high = local + pad >= side;
  let row = 2u | select(0u, 1u, low.x) | select(0u, 4u, high.x);
  let plane = (row << 3u) | select(0u, row, low.y) | select(0u, row << 6u, high.y);
  return (plane << 9u) | select(0u, plane, low.z) | select(0u, plane << 18u, high.z);
}

// Every lane of the workgroup must call this, listed or not; a workgroup lies inside
// one brick. Lanes that produced a nonzero expansion rate called noteExpansion first.
fn recordContent(brick: u32, listed: bool, support: u32, lane: u32) {
  if support != 0u {
    atomicOr(&groupBusy, support);
  }
  workgroupBarrier();
  if !listed || lane != 0u {
    return;
  }
  var flags = 0u;
  if atomicLoad(&groupBusy) != 0u {
    flags = CONTENT | (atomicLoad(&groupBusy) << 3u);
    let local = brick & 63u;
    let bits = (1u << (local & 3u)) | (16u << ((local >> 2u) & 3u)) | (256u << (local >> 4u));
    atomicOr(&activity[brickCount() + (brick >> 6u)], bits);
  }
  if atomicLoad(&groupExpands) != 0u {
    flags |= EXPANDS;
  }
  if flags != 0u {
    atomicOr(&activity[brick], flags);
  }
}

fn noteExpansion() {
  atomicStore(&groupExpands, 1u);
}

// Whether the cell's brick may hold a nonzero expansion rate from an earlier step. Where
// it holds none, storing a zero rate would change nothing.
fn mayExpand(brick: u32) -> bool {
  return (atomicLoad(&activity[brick]) & MAY_EXPAND) != 0u;
}
