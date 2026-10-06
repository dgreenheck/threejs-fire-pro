import { box, cylinder, ground, ring } from './primitives';
import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene);
    cylinder(scene, 1.3, 0.16, [0, 0.08, 0], '#a1a3a4', 1);
    ring(scene, 1.15, 0.025, 0.17, '#b7b7b7');
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      cylinder(scene, 0.035, 0.04, [Math.cos(a) * 1.21, 0.17, Math.sin(a) * 1.21], '#c6c7c8', 1);
    }
    for (const x of [-2, 2]) box(scene, [0.5, 0.08, 0.1], [x, 0.04, 0], '#d1b26c', 1);
    return { objects: {}, ground: floor.control };
  },
};
