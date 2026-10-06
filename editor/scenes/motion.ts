/** Seeded control points and a cubic B-spline keep the wandering path smooth and bounded. */
export function wanderPosition(seconds: number): [number, number, number] {
  const time = Math.max(0, seconds) / 0.7;
  const segment = Math.floor(time),
    t = time - segment;
  const weights = [
    (1 - t) ** 3,
    3 * t ** 3 - 6 * t * t + 4,
    -3 * t ** 3 + 3 * t * t + 3 * t + 1,
    t ** 3,
  ];
  return [0, 1, 2].map((axis) =>
    weights.reduce((sum, weight, offset) => {
      let hash = Math.imul(segment + offset + 17, 374761393) ^ Math.imul(axis + 1, 668265263);
      hash = Math.imul(hash ^ (hash >>> 13), 1274126177);
      const random = ((hash ^ (hash >>> 16)) >>> 0) / 0xffffffff;
      return sum + (weight / 6) * (random * 2 - 1);
    }, 0),
  ) as [number, number, number];
}
