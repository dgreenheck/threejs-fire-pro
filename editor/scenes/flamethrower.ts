import { ground, box } from './primitives';
import { createNozzle } from './nozzle';
import type { SceneContext } from './types';
export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene);
    const rig = createNozzle(scene);
    for (const z of [-2.1, 2.1]) box(scene, [7.5, 0.12, 0.1], [0, 0.06, z], '#a49162', 1);
    return { objects: rig, ground: floor.control };
  },
};
