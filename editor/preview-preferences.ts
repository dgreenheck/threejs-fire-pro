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

export function applyPreviewPreferences(
  document: SimulationDocument,
  preferences: PreviewPreferences,
) {
  if (preferences.lights !== undefined) document.scene.sky = preferences.lights;
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
