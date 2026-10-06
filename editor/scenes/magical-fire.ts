import { OctahedronGeometry } from 'three/webgpu';
import { cylinder, ground, material, mesh, ring } from './primitives';

import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene);
    cylinder(scene, 1.15, 0.18, [0, 0.09, 0], '#41404e', 0.4);
    cylinder(scene, 0.73, 0.55, [0, 0.45, 0], '#353342', 0.5);
    cylinder(scene, 0.95, 0.12, [0, 0.78, 0], '#676071', 0.6);
    ring(scene, 1.4, 0.012, 0.015, '#65617e');
    ring(scene, 1.62, 0.008, 0.015, '#49495f');
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3;
      const crystal = mesh(scene, new OctahedronGeometry(0.2, 0), material('#66627d', 0.4, 0.25), [
        Math.cos(angle) * 1.2,
        0.25,
        Math.sin(angle) * 1.2,
      ]);
      crystal.rotation.y = angle;
    }
    return { objects: {}, ground: floor.control };
  },
};
