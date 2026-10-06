import { ground } from './primitives';
import { sceneAssets } from './props';
import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene);
    const assets = sceneAssets(scene);
    const rocks = [];
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const rock = assets.model('rock', [0.72, 0.36 + 0.04 * Math.sin(i), 0.45]);
      rock.position.set(Math.cos(angle), 0.18, Math.sin(angle));
      rock.rotation.y = angle + i * 0.7;
      scene.add(rock);
      rocks.push(rock);
    }
    for (let i = 0; i < 5; i++) {
      const log = assets.model('wood', [0.28, 0.27, 1.5]);
      log.position.set(Math.cos(i * 2.4) * 0.24, 0.24 + (i % 2) * 0.17, Math.sin(i * 2.4) * 0.24);
      log.rotation.set(0.08, i * 1.26, 0.05);
      scene.add(log);
    }
    return { objects: { rocks }, ground: floor.control };
  },
};
