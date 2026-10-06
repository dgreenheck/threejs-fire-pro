import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { sceneAssets } from './props';
import {
  CylinderGeometry,
  Mesh,
  MeshStandardNodeMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  Scene,
  TorusGeometry,
} from 'three/webgpu';
import type { BufferGeometry, ColorRepresentation, Material, Node, Object3D } from 'three/webgpu';
import { mix, output, positionView, positionWorld, texture, vec4 } from 'three/tsl';

export function material(color: ColorRepresentation, metalness = 0, roughness = 0.8) {
  return new MeshStandardMaterial({ color, metalness, roughness });
}

export function mesh(
  scene: Object3D,
  geometry: BufferGeometry,
  surface: Material,
  position: [number, number, number] = [0, 0, 0],
) {
  const object = new Mesh(geometry, surface);
  object.position.set(...position);
  object.receiveShadow = true;
  scene.add(object);
  return object;
}

export function box(
  scene: Object3D,
  size: [number, number, number],
  position: [number, number, number],
  color: ColorRepresentation,
  metalness = 0,
) {
  return mesh(
    scene,
    new RoundedBoxGeometry(...size, 2, Math.min(...size) * 0.08),
    sceneAssets(scene).surface(metalness ? 'metal' : 'concrete', String(color)),
    position,
  );
}

export function ground(scene: Scene, size = 500) {
  const concrete = sceneAssets(scene).materials.get('concrete')!;
  const surface = new MeshStandardNodeMaterial({
    metalness: 0.08,
    roughness: 0.85,
    normalMap: concrete.normalMap,
    roughnessMap: concrete.roughnessMap,
  });
  surface.colorNode = texture(concrete.map!, positionWorld.xz.mul(0.25)).rgb;
  // Blend the complete shaded floor into the same backdrop, not a gray fog color.
  const floorFade = positionView.length().smoothstep(50, 70);
  surface.outputNode = vec4(
    mix(output.rgb, scene.backgroundNode as Node<'vec3'>, floorFade),
    output.a,
  );
  const geometry = new PlaneGeometry(size, size);
  const positions = geometry.getAttribute('position');
  const uvs = geometry.getAttribute('uv');
  for (let i = 0; i < uvs.count; i++)
    uvs.setXY(i, positions.getX(i) * 0.25, -positions.getY(i) * 0.25);
  const floor = mesh(scene, geometry, surface);
  floor.rotation.x = -Math.PI / 2;
  return floor;
}

export function cylinder(
  scene: Object3D,
  radius: number,
  height: number,
  position: [number, number, number],
  color: ColorRepresentation,
  metalness = 0,
) {
  return mesh(
    scene,
    new CylinderGeometry(radius, radius, height, 64),
    sceneAssets(scene).surface(metalness ? 'metal' : 'concrete', String(color)),
    position,
  );
}

export function ring(
  scene: Object3D,
  radius: number,
  thickness: number,
  y: number,
  color: ColorRepresentation,
) {
  const object = mesh(
    scene,
    new TorusGeometry(radius, thickness, 12, 96),
    sceneAssets(scene).surface('metal', String(color)),
    [0, y, 0],
  );
  object.rotation.x = Math.PI / 2;
  return object;
}

/** The example host owns scene resources; shared geometry/materials are disposed once. */
export function disposeScene(scene: Scene) {
  const geometries = new Set<BufferGeometry>();
  const materials = new Set<Material>();
  scene.traverse((object) => {
    if (!(object instanceof Mesh)) return;
    geometries.add(object.geometry);
    for (const value of Array.isArray(object.material) ? object.material : [object.material]) {
      materials.add(value as Material);
    }
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((surface) => surface.dispose());
  scene.clear();
}
