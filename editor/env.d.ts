import type { SimulationDocument } from './document';
import type { SimulationRuntime } from './runtime';
declare global {
  interface Window {
    __FIRE_EDITOR__?: {
      document: SimulationDocument | null;
      runtime: SimulationRuntime | null;
      error: string;
      loading: boolean;
    };
  }
}
