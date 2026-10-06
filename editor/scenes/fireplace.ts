import { box, ground, mesh } from './primitives';
import { sceneAssets } from './props';
import { BoxGeometry } from 'three/webgpu';
import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene),
      assets = sceneAssets(scene);
    const wallGeometry = new BoxGeometry(12, 6, 0.25);
    const uv = wallGeometry.getAttribute('uv');
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * 6, uv.getY(i) * 3);
    mesh(scene, wallGeometry, assets.surface('brick'), [0, 3, -1.55]);
    box(scene, [3.6, 0.22, 2], [0, 0.11, -0.1], '#88837c');
    const left = box(scene, [0.52, 2.1, 1.1], [-1.42, 1.27, -0.55], '#a39b8c');
    const right = box(scene, [0.52, 2.1, 1.1], [1.42, 1.27, -0.55], '#a39b8c');
    const mantel = box(scene, [3.5, 0.3, 1.5], [0, 2.47, -0.55], '#b6aa96');
    box(scene, [2.3, 1.95, 0.08], [0, 1.2, -1.1], '#302b27');
    // Separate masonry courses and recessed joints give the surround real edge depth.
    for (const x of [-1.42, 1.42])
      for (let i = 0; i < 7; i++)
        box(scene, [0.55, 0.27, 1.13], [x, 0.38 + i * 0.29, -0.55], i % 2 ? '#aaa092' : '#b3aa9c');
    for (let i = 0; i < 4; i++) {
      const log = assets.model('wood', [0.27, 0.26, 1.35]);
      log.position.set((i - 1.5) * 0.29, 0.48, -0.2);
      log.rotation.y = Math.PI / 2 + (i % 2 ? 1 : -1) * 0.45;
      scene.add(log);
    }
    for (const x of [-0.8, 0.8]) box(scene, [0.06, 0.28, 1], [x, 0.36, -0.2], '#535457', 1);
    for (let i = 0; i < 9; i++)
      box(scene, [0.04, 0.05, 1], [-0.8 + i * 0.2, 0.38, -0.2], '#444749', 1);
    return { objects: { left, right, mantel }, ground: floor.control };
  },
};
