// Pool growth.
// Assembled with the other fluid modules by ShaderSources.ts.
// The slots a grown pool added: the first new slot and the new slot count.
struct PoolGrowth {
  slots: vec4u,
};

@group(0) @binding(51) var<uniform> growth: PoolGrowth;
// The pool grew: its new slots go under the free stack, which the host moved up to make
// room, so the stack's top, and the slots the next steps take, do not depend on when the
// pool grew.
@compute @workgroup_size(64)
fn growFreeList(@builtin(global_invocation_id) id: vec3u) {
  let added = growth.slots.y;
  if id.x < added {
    freeSlots[id.x] = growth.slots.x + added - 1u - id.x;
  }
  if id.x == 0u {
    atomicAdd(&poolState.free, i32(added));
  }
}
