import { ground } from './primitives';
import { sceneAssets } from './props';
import type { SceneContext } from './types';
export const recipe = {
  buildScene({ scene }: SceneContext) {
    ground(scene);
    const assets = sceneAssets(scene);
    for (let i = 0; i < 15; i++) {
      const angle = (i / 15) * Math.PI * 2;
      const rock = assets.model('rock', [0.65, 0.3, 0.5]);
      rock.position.set(Math.cos(angle) * 1.55, 0.15, Math.sin(angle) * 1.55);
      rock.rotation.y = angle;
      scene.add(rock);
    }
    return { objects: {} };
  },
};
