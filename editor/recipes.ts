import { ground } from './scenes/primitives';
import { recipe as letters } from './scenes/flaming-letters';
import { recipe as fireplace } from './scenes/fireplace';
import { recipe as campfire } from './scenes/campfire';
import { recipe as projectiles } from './scenes/projectiles';
import { recipe as magic } from './scenes/magical-fire';
import { recipe as delayed } from './scenes/delayed-ignition';
import { recipe as smoke } from './scenes/industrial-smoke';
import { recipe as flamethrower } from './scenes/flamethrower';
import { recipe as tornado } from './scenes/fire-tornado';
import { recipe as burstStage } from './scenes/burst-stage';
import type { Recipe } from './document';
import type { DemoScene, SceneContext } from './scenes/types';
const empty = {
  buildScene({ scene }: SceneContext): DemoScene {
    return { objects: {}, ground: ground(scene).control };
  },
};
export const recipes: Record<Recipe, { buildScene(ctx: SceneContext): DemoScene }> = {
  empty,
  'emitter-forces': empty,
  'flaming-letters': letters,
  fireplace,
  campfire,
  projectiles,
  flamethrower,
  'fire-tornado': tornado,
  'magical-fire': magic,
  'delayed-ignition': delayed,
  'industrial-smoke': smoke,
  'fuel-fireball': burstStage,
  'moving-emitter': empty,
};
