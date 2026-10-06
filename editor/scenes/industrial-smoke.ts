import { CylinderGeometry, TorusGeometry } from 'three/webgpu';
import { sceneAssets } from './props';
import { box, ground, material, mesh } from './primitives';

import type { SceneContext } from './types';

export const recipe = {
  buildScene({ scene }: SceneContext) {
    const floor = ground(scene);
    box(scene, [2.4, 0.25, 2.4], [0, 0.125, 0], '#3c4249', 0.35);
    box(scene, [1.4, 1.2, 1.4], [0, 0.8, 0], '#4a6671', 0.5);
    const pipe = mesh(
      scene,
      new CylinderGeometry(0.32, 0.32, 1.25, 32, 1, true),
      sceneAssets(scene).surface('metal'),
      [0, 1.9, 0],
    );
    for (const y of [1.32, 2.48]) {
      const flange = mesh(
        scene,
        new TorusGeometry(0.33, 0.06, 10, 32),
        sceneAssets(scene).surface('metal', '#b1b2b3'),
        [0, y, 0],
      );
      flange.rotation.x = Math.PI / 2;
    }
    // Inner liner and rolled lip make the chimney a hollow manufactured object.
    mesh(
      scene,
      new CylinderGeometry(0.285, 0.285, 1.18, 48, 1, true),
      material('#25282a', 0.8, 0.8),
      [0, 1.9, 0],
    ).material.side = 2;
    for (const y of [1.32, 2.48])
      for (let i = 0; i < 8; i++) {
        const a = (i * Math.PI) / 4;
        mesh(
          scene,
          new CylinderGeometry(0.025, 0.025, 0.1, 6),
          sceneAssets(scene).surface('metal'),
          [Math.cos(a) * 0.34, y, Math.sin(a) * 0.34],
        );
      }
    for (let i = 0; i < 5; i++)
      box(scene, [0.035, 0.7, 0.02], [-0.5 + i * 0.25, 0.82, 0.711], '#20282e');
    box(scene, [0.27, 0.1, 0.02], [0.38, 1.12, 0.715], '#bda258');
    return { objects: { pipe }, ground: floor.control };
  },
};
