import { validateDocument, type SimulationDocument } from './document';
import campfire from './presets/campfire.json';
import smallFlame from './presets/small-flame.json';
import letters from './presets/flaming-letters.json';
import fireplace from './presets/fireplace.json';
import smoke from './presets/industrial-smoke.json';
import magic from './presets/magical-fire.json';
import delayed from './presets/delayed-ignition.json';
import projectiles from './presets/projectiles.json';
import fireball from './presets/fuel-fireball.json';
import moving from './presets/moving-emitter.json';
import flamethrower from './presets/flamethrower.json';
import tornado from './presets/fire-tornado.json';
export const presets = [
  smallFlame,
  campfire,
  letters,
  fireplace,
  smoke,
  magic,
  delayed,
  projectiles,
  flamethrower,
  tornado,
  fireball,
  moving,
].map((value) => ({
  document: validateDocument(value),
  image: `/thumbnails/${value.id}.png`,
}));
export function presetImage(document: SimulationDocument) {
  return (
    presets.find((p) => p.document.id === document.id) ??
    presets.find((p) => p.document.scene.recipe === document.scene.recipe)
  )?.image;
}
