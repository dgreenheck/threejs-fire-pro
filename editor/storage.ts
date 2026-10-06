import { openDB } from 'idb';
import { validateDocument, type SimulationDocument } from './document';
const db = () =>
  openDB('fire-pro-editor', 1, {
    upgrade(db) {
      db.createObjectStore('documents');
      db.createObjectStore('presets', { keyPath: 'id' });
    },
  });
export async function saveDraft(document: SimulationDocument) {
  const database = await db();
  const tx = database.transaction('documents', 'readwrite');
  await tx.store.put(document, document.id);
  await tx.store.put(document, 'draft');
  await tx.done;
  database.close();
}
export async function loadDraft() {
  const database = await db();
  const value = await database.get('documents', 'draft');
  database.close();
  return value ? validateDocument(value) : null;
}
export async function savePreset(document: SimulationDocument) {
  const database = await db();
  await database.put('presets', document);
  database.close();
}
/** Saved presets in the current format, and the names of saved presets in an older one,
 * which can't be opened. */
export async function loadPresets() {
  const database = await db();
  const values = await database.getAll('presets');
  database.close();
  const presets: SimulationDocument[] = [],
    obsolete: string[] = [];
  for (const value of values) {
    try {
      presets.push(validateDocument(value));
    } catch {
      obsolete.push(typeof value?.name === 'string' ? value.name : 'Unnamed preset');
    }
  }
  return { presets, obsolete };
}
export async function deletePreset(id: string) {
  const database = await db();
  await database.delete('presets', id);
  database.close();
}
