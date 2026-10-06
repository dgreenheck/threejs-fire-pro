/** Fuel that becomes flame where it is hot enough: its amount, up to 1, is the flame's
 * lifetime (fluid-field-evolution.wgsl evolvedFields). */
export interface FuelSettings {
  enabled: boolean;
  ignitionHeat: number;
  /** First-order fuel consumption, in inverse seconds. */
  burnRate: number;
}

export const DEFAULT_FUEL: Readonly<FuelSettings> = {
  enabled: false,
  ignitionHeat: 0.5,
  burnRate: 4,
};
