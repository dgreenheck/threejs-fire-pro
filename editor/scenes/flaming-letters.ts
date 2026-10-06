import { sceneAssets } from './props';
import { ground, mesh } from './primitives';
import { createLetterGeometry } from './letters';

import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    ground(scene);
    const letters = mesh(
      scene,
      createLetterGeometry(),
      sceneAssets(scene).surface('metal'),
      [0, 0, 0],
    );
    return { objects: { letters } };
  },
};
