import { SphereGeometry } from 'three/webgpu';
import { box, ground, material } from './primitives';
import { createNozzle } from './nozzle';
import type { SceneContext } from './types';
export const recipe = {
  buildScene({ scene }: SceneContext) {
    ground(scene);
    const { nozzle, muzzle } = createNozzle(scene);
    const target = box(scene, [0.3, 2.5, 2.8], [3.6, 1.25, 0], '#888d91', 1);
    for (const z of [-1.25, 1.25]) box(scene, [0.65, 0.16, 0.4], [3.6, 0.08, z], '#6f757b', 1);
    for (const y of [0.25, 2.25])
      for (const z of [-1.1, 1.1]) box(scene, [0.04, 0.09, 0.09], [3.43, y, z], '#bbb7a7', 1);
    const projectileGeometry = new SphereGeometry(0.1, 24, 16);
    const projectileMaterial = material('#ffd0a1', 0.2, 0.25);
    projectileMaterial.emissive.set('#ff792d');
    projectileMaterial.emissiveIntensity = 2;
    return {
      objects: { nozzle, muzzle, target, projectileGeometry, projectileMaterial },
      dispose() {
        projectileGeometry.dispose();
        projectileMaterial.dispose();
      },
    };
  },
};
