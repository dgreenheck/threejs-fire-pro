import type { Recipe, SimulationDocument } from './document';

/** Presentation choices belong to each demo, not to the editable simulation schema. */
const authored: Record<Recipe, { start: string; stop: string }> = {
  empty: { start: 'Start emitters', stop: 'Stop emitters' },
  campfire: {
    start: 'Light campfire',
    stop: 'Extinguish',
  },
  fireplace: { start: 'Light fireplace', stop: 'Extinguish' },
  'flaming-letters': {
    start: 'Ignite letters',
    stop: 'Extinguish letters',
  },
  'industrial-smoke': {
    start: 'Start exhaust',
    stop: 'Stop exhaust',
  },
  'magical-fire': {
    start: 'Ignite',
    stop: 'Extinguish',
  },
  'delayed-ignition': {
    start: 'Ignite',
    stop: 'Extinguish',
  },
  projectiles: { start: '', stop: '' },
  flamethrower: {
    start: 'Start spraying',
    stop: 'Stop spraying',
  },
  'fire-tornado': {
    start: 'Ignite vortex',
    stop: 'Extinguish vortex',
  },
  'fuel-fireball': { start: '', stop: '' },
  'moving-emitter': {
    start: 'Ignite sphere',
    stop: 'Extinguish sphere',
  },
};
export function demoActions(document: SimulationDocument) {
  const sources = document.emitters;
  const continuous = sources.filter((e) => e.mode === 'continuous');
  const recipe = document.scene.recipe;
  return {
    ...authored[recipe],
    continuous: continuous.length > 0,
    emitting: continuous.some((e) => e.options.active !== false),
    launch: recipe === 'projectiles',
    burst: recipe !== 'projectiles' && sources.some((e) => e.mode === 'burst'),
    resetOnBurst: recipe === 'fuel-fireball',
  };
}
