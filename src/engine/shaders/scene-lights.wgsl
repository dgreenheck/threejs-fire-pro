// Scene lights: every brick that computed adds the light of its flames to the anchor
// nearest it, an emitter or a burning explosion (SceneLights.ts). A workgroup per active
// brick samples its cells on a 4³ lattice, sums their emission, and adds the sums to its
// anchor's totals; the host reads the totals back and places each anchor's light at the
// centroid of its emission. This lights scene surfaces; it is not volumetric GI or
// geometry shadow transport. volume-common.wgsl supplies the uniform, fields, bricks and
// emission.
// The number of anchors, then each anchor's world position.
@group(0) @binding(15) var<storage, read> anchors: array<vec4f>;
// Eight floats an anchor, as bits: the emission's luminance, its luminance-weighted offset
// from the anchor, and its RGB, summed over the samples. Zeroed before each collection.
@group(0) @binding(16) var<storage, read_write> sums: array<atomic<u32>>;
var<workgroup> nearest: array<vec2f, 64>;
var<workgroup> positions: array<vec4f, 64>;
var<workgroup> colors: array<vec4f, 64>;
// Flames rise from their source: height above an anchor counts a quarter, below it in
// full.
fn anchorDistance(p: vec3f, anchor: vec3f) -> f32 {
  let d = p - anchor;
  let rise = select(d.y, d.y * .25, d.y > 0.0);
  return d.x * d.x + d.z * d.z + rise * rise;
}

fn addFloat(index: u32, value: f32) {
  var old = atomicLoad(&sums[index]);
  loop {
    let sum = bitcast<u32>(bitcast<f32>(old) + value);
    let exchange = atomicCompareExchangeWeak(&sums[index], old, sum);
    if exchange.exchanged {
      break;
    }
    old = exchange.old_value;
  }
}

@compute @workgroup_size(64)
fn collectLights(@builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) lane: u32) {
  let index = listedBrick(group);
  let count = u32(anchors[0].x);
  // Every lane takes the same branch: the brick is the workgroup's.
  if index >= activeBricks[0] || count == 0u {
    return;
  }
  let id = activeBricks[1u + index];
  let page = brickPages[id];
  if page < 0 {
    return;
  }
  let shift = fieldShift();
  let brick = brickOfId(id, tileBoxes[id >> 6u].xyz);
  let stride = 1u << (shift - 2u);
  let local = vec3u(lane & 3u, (lane >> 2u) & 3u, lane >> 4u) * stride + stride / 2u;
  var f = textureLoad(fields, poolOrigin(page, shift) + vec3i(local), 0);
  let smokeShift = renderSmokeShift();
  let ratio = f32(1u << (shift - smokeShift));
  let smokeTexel = vec3f(poolOrigin(page, smokeShift)) + (vec3f(local) + .5) / ratio;
  f.x = textureSampleLevel(smokeDensity,
    linearSampler,
    smokeTexel / vec3f(textureDimensions(smokeDensity)),
    0).x;
  let rgb = emissionAt(f);
  let luminance = dot(rgb, vec3f(.2126, .7152, .0722));
  // The anchor nearest the brick's center, the lanes taking turns through the anchors;
  // ties go to the first.
  let center = (vec3f(brick) + .5) * brickSize();
  var best = vec2f(1e38, 0);
  for (var i = lane; i < count; i += 64u) {
    let d = anchorDistance(center, anchors[1u + i].xyz);
    if d < best.x {
      best = vec2f(d, f32(i));
    }
  }
  nearest[lane] = best;
  workgroupBarrier();
  for (var reach = 32u; reach > 0u; reach >>= 1u) {
    if lane < reach {
      let other = nearest[lane + reach];
      let mine = nearest[lane];
      if other.x < mine.x || (other.x == mine.x && other.y < mine.y) {
        nearest[lane] = other;
      }
    }
    workgroupBarrier();
  }
  let anchor = u32(nearest[0].y);
  let p = (vec3f(brick * (1 << shift) + vec3i(local)) + .5) * u.grid.x;
  positions[lane] = vec4f(luminance, (p - anchors[1u + anchor].xyz) * luminance);
  colors[lane] = vec4f(rgb, 0);
  workgroupBarrier();
  for (var reach = 32u; reach > 0u; reach >>= 1u) {
    if lane < reach {
      positions[lane] += positions[lane + reach];
      colors[lane] += colors[lane + reach];
    }
    workgroupBarrier();
  }
  // Seven lanes add a total each.
  if lane < 7u {
    var totals = array<f32, 7>(
      positions[0].x, positions[0].y, positions[0].z, positions[0].w,
      colors[0].x, colors[0].y, colors[0].z,
    );
    addFloat(anchor * 8u + lane, totals[lane]);
  }
}
