import type { GpuTimingRow } from '../src/engine/GpuProfiler.ts';
import { createGpuTimingTree, updateGpuTimingTree, type GpuTimingNode } from './gpu-timing-tree.ts';

export const GPU_TIMING_WINDOW = 100;

export interface GpuTimingStatistics {
  samples: number;
  mean: number;
  standardDeviation: number;
  minimum: number;
  maximum: number;
}

export function isValidGpuTimingRow(row: GpuTimingRow): boolean {
  return (
    Number.isFinite(row.milliseconds) &&
    row.milliseconds >= 0 &&
    Number.isFinite(row.calls) &&
    row.calls > 0
  );
}

/** A bounded window with stable running variance; rescan only when an extreme expires. */
class TimingWindow {
  private values = new Float64Array(GPU_TIMING_WINDOW);
  private cursor = 0;
  private samples = 0;
  private mean = 0;
  private squaredDeviations = 0;
  private minimum = 0;
  private maximum = 0;

  add(value: number): void {
    const old = this.values[this.cursor];
    this.values[this.cursor] = value;
    this.cursor = (this.cursor + 1) % GPU_TIMING_WINDOW;
    if (this.samples < GPU_TIMING_WINDOW) {
      this.samples++;
      const delta = value - this.mean;
      this.mean += delta / this.samples;
      this.squaredDeviations += delta * (value - this.mean);
      this.minimum = this.samples === 1 ? value : Math.min(this.minimum, value);
      this.maximum = this.samples === 1 ? value : Math.max(this.maximum, value);
      return;
    }
    const previousMean = this.mean;
    this.mean += (value - old) / this.samples;
    this.squaredDeviations += (value - old) * (value - this.mean + old - previousMean);
    if ((old === this.minimum && value > old) || (old === this.maximum && value < old)) {
      this.minimum = Infinity;
      this.maximum = -Infinity;
      for (const sample of this.values) {
        this.minimum = Math.min(this.minimum, sample);
        this.maximum = Math.max(this.maximum, sample);
      }
    } else {
      this.minimum = Math.min(this.minimum, value);
      this.maximum = Math.max(this.maximum, value);
    }
  }

  snapshot(): GpuTimingStatistics {
    return {
      samples: this.samples,
      mean: this.mean,
      standardDeviation: Math.sqrt(Math.max(0, this.squaredDeviations / this.samples)),
      minimum: this.minimum,
      maximum: this.maximum,
    };
  }
}

/** Record valid per-frame group totals and the latest 100 nonnegative samples per pass. */
export class GpuTimingHistory {
  samples = 0;
  private tree = createGpuTimingTree();
  private windows = new Map<string, TimingWindow>();

  add(rows: Iterable<GpuTimingRow>): void {
    const valid = Array.from(rows).filter(isValidGpuTimingRow);
    if (!valid.length) return;
    updateGpuTimingTree(this.tree, valid);
    const visit = (node: GpuTimingNode) => {
      const value = node.milliseconds ?? 0;
      // A recorded zero duration is valid; absent passes have no calls.
      if (node.calls > 0 && Number.isFinite(value) && value >= 0) {
        let window = this.windows.get(node.id);
        if (!window) {
          window = new TimingWindow();
          this.windows.set(node.id, window);
        }
        window.add(value);
      }
      node.children?.forEach(visit);
    };
    visit(this.tree);
    this.samples = Math.min(GPU_TIMING_WINDOW, this.samples + 1);
  }

  snapshot(): ReadonlyMap<string, GpuTimingStatistics> {
    return new Map(Array.from(this.windows, ([id, window]) => [id, window.snapshot()]));
  }

  reset(): void {
    this.samples = 0;
    this.tree = createGpuTimingTree();
    this.windows.clear();
  }
}
