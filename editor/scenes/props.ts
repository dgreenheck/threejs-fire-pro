import {
  Box3,
  Group,
  Mesh,
  MeshStandardMaterial,
  RepeatWrapping,
  SRGBColorSpace,
  TextureLoader,
  Vector3,
  type Object3D,
  type Texture,
} from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import rockUrl from './assets/rock_07.glb?url';
import woodUrl from './assets/dead_tree_trunk_02.glb?url';
import brickColor from './assets/red_brick_03-diff.jpg';
import brickNormal from './assets/red_brick_03-nor_gl.jpg';
import brickRough from './assets/red_brick_03-rough.jpg';
import concreteColor from './assets/concrete_wall_006-diff.jpg';
import concreteNormal from './assets/concrete_wall_006-nor_gl.jpg';
import concreteRough from './assets/concrete_wall_006-rough.jpg';
import metalColor from './assets/metal_plate_02-diff.jpg';
import metalNormal from './assets/metal_plate_02-nor_gl.jpg';
import metalRough from './assets/metal_plate_02-rough.jpg';

/** Scene-owned assets. Geometry is shared by instances; every GPU resource is released on exit. */
export class PropAssets {
  readonly materials = new Map<string, MeshStandardMaterial>();
  readonly models = new Map<string, Group>();
  private textures = new Set<Texture>();
  async load(natural: boolean) {
    const loader = new TextureLoader();
    const jobs = [
      ['brick', brickColor, brickNormal, brickRough],
      ['concrete', concreteColor, concreteNormal, concreteRough],
      ['metal', metalColor, metalNormal, metalRough],
    ].map(async ([name, color, normal, rough]) => {
      const maps = await Promise.all(
        [color, normal, rough].map(async (url) => {
          const map = await loader.loadAsync(url);
          map.wrapS = map.wrapT = RepeatWrapping;
          map.anisotropy = 8;
          this.textures.add(map);
          return map;
        }),
      );
      maps[0].colorSpace = SRGBColorSpace;
      this.materials.set(
        name,
        new MeshStandardMaterial({
          map: maps[0],
          normalMap: maps[1],
          roughnessMap: maps[2],
          roughness: 1,
          metalness: name === 'metal' ? 0.7 : 0,
        }),
      );
    });
    if (natural)
      for (const [name, url] of [
        ['rock', rockUrl],
        ['wood', woodUrl],
      ])
        jobs.push(
          (async () => {
            const { scene } = await new GLTFLoader().loadAsync(url);
            // Center the source once; instances retain the artist's UVs, normals and silhouette.
            if (name === 'wood') scene.rotation.y = Math.PI / 2;
            scene.updateMatrixWorld(true);
            const bounds = new Box3().setFromObject(scene);
            const center = bounds.getCenter(new Vector3());
            const size = bounds.getSize(new Vector3());
            const normalized = new Group();
            scene.position.sub(center);
            normalized.add(scene);
            normalized.userData.size = size;
            scene.traverse((object) => {
              if (object instanceof Mesh)
                for (const material of Array.isArray(object.material)
                  ? object.material
                  : [object.material]) {
                  for (const value of Object.values(material))
                    if ((value as Texture)?.isTexture) this.textures.add(value as Texture);
                }
            });
            this.models.set(name, normalized);
          })(),
        );
    const results = await Promise.allSettled(jobs);
    const failure = results.find((r) => r.status === 'rejected');
    if (failure?.status === 'rejected') {
      this.dispose();
      throw failure.reason;
    }
  }
  surface(name: 'brick' | 'concrete' | 'metal', color = '#ffffff') {
    const surface = this.materials.get(name)!.clone();
    surface.color.set(color);
    return surface;
  }
  model(name: 'rock' | 'wood', dimensions: [number, number, number]) {
    const source = this.models.get(name)!;
    const model = source.clone(true);
    const size = source.userData.size as Vector3;
    model.scale.set(dimensions[0] / size.x, dimensions[1] / size.y, dimensions[2] / size.z);
    // Keep the collider/placement frame rigid even when the visible mesh is resized.
    const root = new Group();
    root.add(model);
    return root;
  }
  dispose() {
    for (const model of this.models.values())
      model.traverse((object) => {
        if (object instanceof Mesh) {
          object.geometry.dispose();
          for (const m of Array.isArray(object.material) ? object.material : [object.material])
            m.dispose();
        }
      });
    this.materials.forEach((m) => m.dispose());
    this.textures.forEach((t) => t.dispose());
    this.models.clear();
    this.materials.clear();
    this.textures.clear();
  }
}
export function sceneAssets(object: Object3D): PropAssets {
  let root = object;
  while (root.parent) root = root.parent;
  return root.userData.props as PropAssets;
}
