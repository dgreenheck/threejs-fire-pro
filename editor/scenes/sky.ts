import {
  CubeTexture,
  DataTexture,
  HalfFloatType,
  LinearFilter,
  LinearSRGBColorSpace,
  Vector3,
} from 'three/webgpu';
import {
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
  color,
  cubeTexture,
  mix,
  screenUV,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import skyUrl from './assets/daylight-sky.bin?url';
import metadata from './assets/daylight-sky.json';

/** Fixed atmospheric radiance baked offline; no runtime scattering integration. */
export async function createPreviewSky() {
  const response = await fetch(skyUrl);
  if (!response.ok) throw new Error(`Could not load the sky lookup: ${response.status}.`);
  const buffer = await response.arrayBuffer();
  const size = metadata.faceSize;
  const faceValues = size * size * 4;
  if (buffer.byteLength !== 6 * faceValues * 2) throw new Error('Invalid sky lookup size.');
  const faces = Array.from({ length: 6 }, (_, face) => {
    const data = new Uint16Array(buffer, face * faceValues * 2, faceValues);
    const image = new DataTexture(data, size, size);
    image.type = HalfFloatType;
    image.colorSpace = LinearSRGBColorSpace;
    return image;
  });
  const lookup = new CubeTexture(faces);
  lookup.name = 'Baked Rayleigh–Mie daytime sky';
  lookup.type = HalfFloatType;
  lookup.colorSpace = LinearSRGBColorSpace;
  lookup.minFilter = lookup.magFilter = LinearFilter;
  lookup.generateMipmaps = false;
  lookup.needsUpdate = true;

  // Reconstruct the same world ray for the sky and the floor's sky blend.
  // Using surface normals here would make the floor sample the zenith everywhere.
  const clip = vec4(screenUV.x.mul(2).sub(1), screenUV.y.mul(-2).add(1), 1, 1);
  const view = cameraProjectionMatrixInverse.mul(clip).xyz;
  const world = cameraWorldMatrix.mul(vec4(view, 0)).xyz.normalize();
  // Three flips X for loaded cubemaps; these baked faces use standard cube axes.
  const radiance = cubeTexture(lookup, vec3(world.x.negate(), world.yz)).rgb.mul(0.45);
  const blend = uniform(1);
  const node = mix(color('#121820').rgb, radiance, blend);
  return {
    node,
    environment: lookup,
    blend,
    sunDirection: new Vector3(...metadata.sunDirection),
    dispose() {
      node.dispose();
      lookup.dispose();
      faces.forEach((face) => face.dispose());
    },
  };
}
