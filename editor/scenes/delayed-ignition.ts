import { box, cylinder, ground, ring } from './primitives';

import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene);
    cylinder(scene, 1.25, 0.15, [0, 0.075, 0], '#48515a', 0.5);
    ring(scene, 1.05, 0.025, 0.16, '#69858d');
    cylinder(scene, 0.15, 0.28, [0, 0.29, 0], '#6c747a', 0.7);
    for (const x of [-1.4, 1.4]) {
      box(scene, [0.06, 0.7, 0.06], [x, 0.35, 0], '#636d78', 0.5);
      box(scene, [0.14, 0.08, 0.14], [x, 0.72, 0], '#c69a62', 0.3);
    }
    for (let i = 0; i < 12; i++) {
      const a = (i * Math.PI) / 6;
      cylinder(scene, 0.025, 0.025, [Math.cos(a) * 1.13, 0.16, Math.sin(a) * 1.13], '#a8a9aa', 1);
    }
    return { objects: {}, ground: floor.control };
  },
};
