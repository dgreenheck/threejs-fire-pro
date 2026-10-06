export type GpuTimingCategory = 'simulation' | 'lighting' | 'render';
export interface GpuTimingRow {
  label: string;
  category: GpuTimingCategory;
  milliseconds: number;
  calls: number;
}
export interface GpuTimingFrame {
  rows: GpuTimingRow[];
  skipped: number;
}

const activeProfilers = new WeakMap<GPUDevice, GpuProfiler>();
const QUERY_COUNT = 4096;
type ComputePass = Pick<
  GPUComputePassEncoder,
  'setPipeline' | 'setBindGroup' | 'dispatchWorkgroups' | 'dispatchWorkgroupsIndirect' | 'end'
>;

/** Native passes normally; one timestamped pass per dispatch during a sampled frame. */
export function beginComputePass(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  descriptor: GPUComputePassDescriptor,
  category: Exclude<GpuTimingCategory, 'render'> = 'simulation',
): ComputePass {
  const profiler = activeProfilers.get(device);
  if (!profiler) return encoder.beginComputePass(descriptor);
  let pipeline: GPUComputePipeline;
  const bindings = new Map<number, { group: GPUBindGroup | null; offsets: number[] }>();
  const timed = (dispatch: (pass: GPUComputePassEncoder) => void) => {
    const label = pipeline.label || descriptor.label || 'Compute kernel';
    const pass = encoder.beginComputePass({
      ...descriptor,
      label,
      timestampWrites: profiler.timestamps(label, category),
    });
    pass.setPipeline(pipeline);
    for (const [index, { group, offsets }] of bindings) pass.setBindGroup(index, group, offsets);
    dispatch(pass);
    pass.end();
  };
  return {
    setPipeline(value) {
      pipeline = value;
    },
    setBindGroup(
      index: number,
      group: GPUBindGroup | null,
      offsets: Iterable<number> | Uint32Array = [],
      start: number = 0,
      length?: number,
    ) {
      const values = Array.from(offsets);
      bindings.set(index, {
        group,
        offsets: values.slice(start, length === undefined ? undefined : start + length),
      });
    },
    dispatchWorkgroups(x, y = 1, z = 1) {
      timed((pass) => pass.dispatchWorkgroups(x, y, z));
    },
    dispatchWorkgroupsIndirect(buffer, offset) {
      timed((pass) => pass.dispatchWorkgroupsIndirect(buffer, offset));
    },
    end() {},
  };
}

/**
 * GPU time per pass label, counting each moment once. Pass timestamps overlap: a render
 * pass starts its vertex work while earlier passes still run and then waits for them, and
 * on tile-based GPUs the next pass starts before the last one's fragments finish. Each
 * moment goes to the running pass that finishes first, the one making progress, so the rows
 * add up to the time the GPU was busy. Passes whose end precedes their start are left out
 * and counted as invalid.
 */
function exclusiveTimings(
  entries: readonly { label: string; category: GpuTimingCategory }[],
  times: ArrayLike<bigint>,
): { rows: GpuTimingRow[]; invalid: number } {
  const rows = new Map<string, GpuTimingRow>();
  const spans: { row: GpuTimingRow; start: bigint; end: bigint }[] = [];
  let invalid = 0;
  entries.forEach(({ label, category }, index) => {
    const start = times[index * 2];
    const end = times[index * 2 + 1];
    if (end < start) {
      invalid++;
      return;
    }
    const key = `${category}:${label}`;
    const row = rows.get(key) ?? { label, category, milliseconds: 0, calls: 0 };
    row.calls++;
    rows.set(key, row);
    spans.push({ row, start, end });
  });
  const moments = [...new Set(spans.flatMap(({ start, end }) => [start, end]))].sort((a, b) =>
    a < b ? -1 : a > b ? 1 : 0,
  );
  for (let i = 0; i + 1 < moments.length; i++) {
    const [from, to] = [moments[i], moments[i + 1]];
    let owner: (typeof spans)[number] | undefined;
    for (const span of spans)
      if (span.start <= from && span.end >= to && (!owner || span.end < owner.end)) owner = span;
    if (owner) owner.row.milliseconds += Number(to - from) / 1e6;
  }
  return { rows: [...rows.values()], invalid };
}

/** Bounded, asynchronous timestamp readback. No GPU waits in the render loop. */
export class GpuProfiler {
  readonly supported: boolean;
  renderLabel = 'Scene rendering';
  private resources?: { queries: GPUQuerySet; resolve: GPUBuffer; readback: GPUBuffer };
  private entries: { label: string; category: GpuTimingCategory }[] = [];
  private skipped = 0;
  private recording = false;
  private pending = false;
  private disposed = false;
  private originalCreateEncoder?: GPUDevice['createCommandEncoder'];
  private readonly device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
    this.supported = device.features.has('timestamp-query');
  }

  beginFrame(): boolean {
    if (!this.supported || this.pending || this.recording || this.disposed) return false;
    if (!this.resources) {
      this.resources = {
        queries: this.device.createQuerySet({
          label: 'Live GPU timings',
          type: 'timestamp',
          count: QUERY_COUNT,
        }),
        resolve: this.device.createBuffer({
          label: 'GPU timing resolve',
          size: QUERY_COUNT * 8,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        readback: this.device.createBuffer({
          label: 'GPU timing readback',
          size: QUERY_COUNT * 8,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
      };
    }
    this.entries = [];
    this.skipped = 0;
    this.recording = true;
    activeProfilers.set(this.device, this);
    // Three.js owns its render-pass encoders. Instrument their pass boundaries
    // only during this frame, then restore the exact original device method.
    const createEncoder = this.device.createCommandEncoder;
    this.originalCreateEncoder = createEncoder;
    this.device.createCommandEncoder = (descriptor) => {
      const encoder = createEncoder.call(this.device, descriptor);
      const encoderLabel = descriptor?.label;
      const beginRenderPass = encoder.beginRenderPass.bind(encoder);
      encoder.beginRenderPass = (renderDescriptor) =>
        beginRenderPass({
          ...renderDescriptor,
          timestampWrites: this.timestamps(
            encoderLabel === 'mipmapEncoder' ? 'Texture mipmaps' : this.renderLabel,
            'render',
          ),
        });
      return encoder;
    };
    return true;
  }

  timestamps(
    label: string,
    category: GpuTimingCategory,
  ): GPUComputePassTimestampWrites | undefined {
    if (!this.recording) return undefined;
    const index = this.entries.length * 2;
    if (index + 2 > QUERY_COUNT) {
      this.skipped++;
      return undefined;
    }
    this.entries.push({ label, category });
    return {
      querySet: this.resources!.queries,
      beginningOfPassWriteIndex: index,
      endOfPassWriteIndex: index + 1,
    };
  }

  async endFrame(): Promise<GpuTimingFrame | null> {
    if (!this.recording) return null;
    this.stopRecording();
    const entries = this.entries;
    const skipped = this.skipped;
    if (!entries.length) return { rows: [], skipped };
    const { queries, resolve, readback } = this.resources!;
    const bytes = entries.length * 16;
    const encoder = this.device.createCommandEncoder({ label: 'Resolve live GPU timings' });
    encoder.resolveQuerySet(queries, 0, entries.length * 2, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, readback, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    this.pending = true;
    try {
      await readback.mapAsync(GPUMapMode.READ, 0, bytes);
      if (this.disposed) return null;
      const times = new BigUint64Array(readback.getMappedRange(0, bytes));
      const { rows, invalid } = exclusiveTimings(entries, times);
      return { rows, skipped: skipped + invalid };
    } catch (error) {
      if (!this.disposed) throw error;
      return null;
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      this.pending = false;
    }
  }

  private stopRecording(): void {
    activeProfilers.delete(this.device);
    if (this.originalCreateEncoder) this.device.createCommandEncoder = this.originalCreateEncoder;
    this.originalCreateEncoder = undefined;
    this.recording = false;
  }

  dispose(): void {
    this.disposed = true;
    if (this.recording) this.stopRecording();
    this.resources?.queries.destroy();
    this.resources?.resolve.destroy();
    this.resources?.readback.destroy();
    this.resources = undefined;
  }
}
