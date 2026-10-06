import type { SimulationDocument } from './document';

interface PreviewPreferences {
  lights?: boolean;
}
const storageKey = 'fire-pro-preview-preferences';

export function loadPreviewPreferences(): PreviewPreferences {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) ?? '{}');
    return { lights: typeof value?.lights === 'boolean' ? value.lights : undefined };
  } catch {
    return {};
  }
}

/** Whether a scene makes smoke but no flame or fuel: it is only visible lit. */
function smokeOnly(document: SimulationDocument) {
  const amounts = document.emitters.map((e) =>
    e.mode === 'burst'
      ? { ...e.burst.charge }
      : {
          flame: e.options.emission?.flame,
          smoke: e.options.emission?.smokeRate,
          fuel: e.options.emission?.fuelRate,
        },
  );
  return (
    amounts.some((a) => (a.smoke ?? 0) > 0) &&
    amounts.every((a) => !(a.flame ?? 0) && !(a.fuel ?? 0))
  );
}

export function applyPreviewPreferences(
  document: SimulationDocument,
  preferences: PreviewPreferences,
) {
  if (smokeOnly(document)) document.scene.sky = true;
  else if (preferences.lights !== undefined) document.scene.sky = preferences.lights;
  return document;
}

export function rememberPreviewPreferences(
  before: SimulationDocument,
  after: SimulationDocument,
  preferences: PreviewPreferences,
) {
  if (before.scene.sky === after.scene.sky) return;
  preferences.lights = after.scene.sky;
  try {
    localStorage.setItem(storageKey, JSON.stringify(preferences));
  } catch {
    // The in-memory preferences still work when browser storage is unavailable.
  }
}
