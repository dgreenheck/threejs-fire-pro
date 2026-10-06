import type { Vec3 } from './types.ts';

// Multi-lobe observer fit, Eq. 4 / Table 1, Wyman, Sloan & Shirley (JCGT 2013).
// https://jcgt.org/published/0002/02/01/paper.pdf
const lobes = [
  [
    [0.362, 442, 0.0624, 0.0374],
    [1.056, 599.8, 0.0264, 0.0323],
    [-0.065, 501.1, 0.049, 0.0382],
  ],
  [
    [0.821, 568.8, 0.0213, 0.0247],
    [0.286, 530.9, 0.0613, 0.0322],
  ],
  [
    [1.217, 437, 0.0845, 0.0278],
    [0.681, 459, 0.0385, 0.0725],
  ],
];
const observer = Array.from({ length: 95 }, (_, i) => {
  const wavelength = 360 + i * 5;
  const xyz = lobes.map((channel) =>
    channel.reduce((sum, [height, center, left, right]) => {
      const distance = (wavelength - center) * (wavelength < center ? left : right);
      return sum + height * Math.exp(-0.5 * distance * distance);
    }, 0),
  ) as Vec3;
  return { wavelength, xyz };
});

/** Planck spectral radiance in W sr^-1 m^-3; wavelength is supplied in nm. */
function planckRadiance(wavelengthNm: number, kelvin: number): number {
  if (
    !Number.isFinite(wavelengthNm) ||
    wavelengthNm <= 0 ||
    !Number.isFinite(kelvin) ||
    kelvin <= 0
  )
    throw new Error('Wavelength and temperature must be positive and finite.');
  const wavelength = wavelengthNm * 1e-9;
  // Exact SI h, c and k; the second radiation constant is h*c/k.
  const h = 6.62607015e-34,
    c = 299792458,
    k = 1.380649e-23;
  return (2 * h * c * c) / (wavelength ** 5 * Math.expm1((h * c) / (wavelength * k * kelvin)));
}

/** Visible spectral integration against the analytic CIE 1931 observer fit. */
function blackbodyXYZ(kelvin: number): Vec3 {
  const xyz: Vec3 = [0, 0, 0];
  for (const sample of observer) {
    const radiance = planckRadiance(sample.wavelength, kelvin) * 5e-9;
    for (let c = 0; c < 3; c++) xyz[c] += sample.xyz[c] * radiance;
  }
  return xyz;
}

function linearRGB([x, y, z]: Vec3): Vec3 {
  return [
    3.2406 * x - 1.5372 * y - 0.4986 * z,
    -0.9689 * x + 1.8758 * y + 0.0415 * z,
    0.0557 * x - 0.204 * y + 1.057 * z,
  ].map((v) => Math.max(0, v)) as Vec3;
}
const BLACKBODY_MIN_K = 500;
const BLACKBODY_MAX_K = 10000;
export const BLACKBODY_TABLE_SIZE = 512;
const referencePeak = Math.max(...linearRGB(blackbodyXYZ(1800)));

/** Peak-normalized linear RGB and log2 radiance relative to 1800 K. */
function blackbodySample(kelvin: number): [number, number, number, number] {
  const rgb = linearRGB(blackbodyXYZ(kelvin)),
    peak = Math.max(...rgb);
  if (peak <= 0) return [0, 0, 0, -32];
  return [...rgb.map((v) => v / peak), Math.max(-32, Math.log2(peak / referencePeak))] as [
    number,
    number,
    number,
    number,
  ];
}

let table: Float32Array | undefined;
/** Shared CPU data; treat as read-only. GPU ownership belongs to each renderer. */
export function blackbodyTable(): Float32Array {
  if (!table) {
    table = new Float32Array(BLACKBODY_TABLE_SIZE * 4);
    for (let i = 0; i < BLACKBODY_TABLE_SIZE; i++)
      table.set(
        blackbodySample(
          BLACKBODY_MIN_K + ((BLACKBODY_MAX_K - BLACKBODY_MIN_K) * i) / (BLACKBODY_TABLE_SIZE - 1),
        ),
        i * 4,
      );
  }
  return table;
}
