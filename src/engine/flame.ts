/** FLAME-LIFETIME-001: authored response masks, not chemical reaction rates.
 */
export type FlameRamp = ReadonlyArray<readonly [lifetime: number, response: number]>;
/** The renderer derives the flame's glow from its lifetime (fire-emission.wgsl). */
export const FLAME_CHANNELS = ['smoke', 'heat', 'expansion'] as const;
export const MAX_FLAME_KNOTS = 8;
export interface FlameSettings {
  lifespan: number;
  smokeRate: number;
  heatRate: number;
  expansionRate: number;
  ramps: Record<(typeof FLAME_CHANNELS)[number], FlameRamp>;
}

/** Authored starting look, not coefficients inferred from a combustion paper. */
export const DEFAULT_FLAME_SETTINGS: FlameSettings = {
  lifespan: 1.2,
  smokeRate: 2,
  heatRate: 4,
  expansionRate: 1,
  ramps: {
    smoke: [
      [0, 0],
      [0.2, 1],
      [0.5, 0],
      [1, 0],
    ],
    heat: [
      [0, 0],
      [0.2, 1],
      [1, 1],
    ],
    expansion: [
      [0, 0],
      [0.5, 0],
      [0.75, 1],
      [1, 1],
    ],
  },
};

/** Serialize one step atomically; validate the values actually sent as f32. */
export function packFlameStep(
  settings: FlameSettings,
  dt: number,
): { uniform: ArrayBuffer; ramps: Float32Array<ArrayBuffer> } {
  const finite = (name: string, value: number): number => {
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value)))
      throw new Error(`${name} must be finite in f32.`);
    return Math.fround(value);
  };
  const step = finite('Flame timestep', dt),
    lifespan = finite('Flame lifespan', settings.lifespan);
  if (step <= 0 || lifespan <= 0)
    throw new Error('Flame timestep and lifespan must be positive in f32.');
  finite('Lifetime decrement', step / lifespan);
  const smoke = finite('Smoke rate', settings.smokeRate),
    heat = finite('Heat rate', settings.heatRate);
  const expansion = finite('Expansion rate', settings.expansionRate);
  if (smoke < 0 || heat < 0) throw new Error('Smoke and heat rates must be nonnegative.');
  for (const rate of [smoke, heat, expansion]) finite('Integrated flame output', rate * lifespan);
  const uniform = new ArrayBuffer(48),
    floats = new Float32Array(uniform),
    counts = new Uint32Array(uniform, 32, 4);
  floats.set([step, lifespan, smoke, heat, expansion, 0, 0, 0]);
  const ramps = new Float32Array(FLAME_CHANNELS.length * MAX_FLAME_KNOTS * 2);
  FLAME_CHANNELS.forEach((channel, index) => {
    const knots = settings.ramps?.[channel];
    if (!Array.isArray(knots) || knots.length < 2 || knots.length > MAX_FLAME_KNOTS)
      throw new Error(`${channel} ramp needs 2–${MAX_FLAME_KNOTS} knots.`);
    let previous = -1;
    knots.forEach((knot, i) => {
      if (!Array.isArray(knot) || knot.length !== 2)
        throw new Error(`${channel} ramp needs [lifetime,response] pairs.`);
      const x = finite(`${channel} lifetime`, knot[0]),
        y = finite(`${channel} response`, knot[1]);
      if (x < 0 || x > 1 || y < 0 || y > 1 || x <= previous)
        throw new Error(`${channel} knots must increase in f32 with coordinates in [0,1].`);
      if ((i === 0 && x !== 0) || (i === knots.length - 1 && x !== 1))
        throw new Error(`${channel} ramp must span lifetime 0–1.`);
      ramps.set([x, y], (index * MAX_FLAME_KNOTS + i) * 2);
      previous = x;
    });
    counts[index] = knots.length;
  });
  return { uniform, ramps };
}
