import type { Camera, Object3D, WebGPURenderer } from 'three/webgpu';
import { GpuProfiler, type GpuTimingFrame, type GpuTimingRow } from '../src/engine/GpuProfiler';
import { createGpuTimingTree, updateGpuTimingTree, type GpuTimingNode } from './gpu-timing-tree';
import {
  GpuTimingHistory,
  GPU_TIMING_WINDOW,
  isValidGpuTimingRow,
  type GpuTimingStatistics,
} from './gpu-timing-history';

const WARMUP_SAMPLES = 10;

/** What each kernel does, for the profiler's tooltips. */
export const KERNEL_NOTES: Record<string, string> = {
  markContent: 'Find the bricks that hold content after a diagnostic wrote the fields.',
  buildBricks: 'Decide which bricks compute and which hold storage this step.',
  freeBricks: 'Release the storage of bricks nothing needs any more.',
  allocateBricks: 'Give storage to bricks that need it and list the bricks that compute.',
  linkBricks: "Record each brick's neighbors for the step's kernels.",
  buildSolids:
    'Draw the colliders into the solid cells of new bricks, or of every brick after they move.',
  finishBricks: 'Size the dispatches over the listed bricks.',
  zeroVelocityBricks: 'Clear the velocity and pressure of new or idle bricks.',
  zeroFieldBricks: 'Clear the fields of new or idle bricks.',
  advectVelocityWithBounds: 'Transport velocity and collect correction bounds.',
  correctVelocity: 'Correct velocity transport and clamp overshoot.',
  advectScalars: 'Transport heat, flame lifetime and fuel, plus smoke in Full mode.',
  advectSmoke: 'Transport smoke on its separate coarser grid.',
  evolveSmoke: 'Correct smoke transport, apply dissipation and average fine-cell smoke production.',
  zeroSmokeBricks: 'Clear the separate smoke pools of new or idle bricks.',
  smoothSmoke: 'Smooth full-resolution smoke into the rendering cache.',
  smoothCoarseSmoke: 'Smooth smoke on its coarse grid without a fine-grid copy.',
  evolveFields:
    'Correct scalar transport, inject sources, burn fuel, evolve flames and produce heat, smoke and expansion.',
  computeCurl: 'Measure local rotation for vorticity forces.',
  applyForces:
    'Evaluate buoyancy, wind, turbulence and vorticity, and apply them to the velocity grid.',
  computeDivergence: 'Measure flow divergence before the pressure solve.',
  'Multigrid initializePressure':
    'Compute the first pressure iteration directly from a zero guess.',
  'Multigrid buildBrickTopology':
    'Find solid and open neighbors in new bricks, or in every brick after the tiles or geometry change.',
  'Multigrid buildTileTopology':
    'Find solid and open neighbors on the tile levels after the tiles or geometry change.',
  'Multigrid relaxBricks': 'Smooth pressure on the active bricks of the finest level.',
  'Multigrid relaxCoarseBricks': 'Smooth pressure on the coarser levels inside each brick.',
  'Multigrid relaxTiles': 'Smooth pressure on the levels stored per tile of bricks.',
  'Multigrid relaxCoarse': 'Solve the bottom level, one cell per tile, in one workgroup.',
  'Multigrid restrictBricks':
    'Transfer remaining pressure error to a coarser level inside each brick and take its first smoothing step.',
  'Multigrid restrictTiles':
    'Transfer remaining pressure error to a coarser tile level and take its first smoothing step.',
  'Multigrid prolongateBricks': 'Add coarse corrections to the active finest bricks.',
  'Multigrid prolongateCoarseBricks': 'Add coarse corrections to a finer level inside each brick.',
  'Multigrid prolongateTiles': 'Add coarse corrections to a finer tile level.',
  project: 'Correct velocity using the solved pressure.',
  exportPages: "Update the page tables of the renderer's boxes.",
  buildOccupancy:
    'Find the blocks of each brick holding visible smoke or flame, which the camera marches.',
  buildLighting: 'Cache the light that reaches the fire and smoke, in every brick.',
  lightFresh: 'Light the bricks the latest step gave storage.',
  buildSlices: 'March the camera rays at half resolution.',
  collectLights: "Sum the flames' light by emitter for the scene lights.",
};

export interface GpuTimingsSnapshot {
  /** Unavailable without GPU timestamps; waiting until the first sample arrives. */
  state: 'off' | 'unavailable' | 'waiting' | 'live' | 'failed';
  tree: GpuTimingNode;
  /** Statistics over up to the last 100 complete captured frames. */
  statisticsSamples: number;
  statistics: ReadonlyMap<string, GpuTimingStatistics>;
  message?: string;
}

/**
 * GPU pass timings for the preview, sampled with timestamp queries up to five frames a
 * second. Discard the first 10 complete captures after starting or resetting.
 * Statistics use the latest 100 nonnegative, finite samples per pass after warmup.
 * Nothing is measured while it is off.
 */
export function createGpuTimings(renderer: WebGPURenderer) {
  const device = (renderer.backend as unknown as { device?: GPUDevice }).device;
  const supported = Boolean(device?.features.has('timestamp-query'));
  let profiler: GpuProfiler | undefined;
  let tree = createGpuTimingTree();
  const history: GpuTimingFrame[] = [];
  const rolling = new GpuTimingHistory();
  let statistics = rolling.snapshot();
  let discardedSamples = 0;
  let active = false;
  let capturing = false;
  let nextSample = 0;
  let generation = 0;
  let frameGeneration = 0;
  let failure: string | undefined;
  let disposed = false;
  let originalRender: WebGPURenderer['render'] | undefined;

  const show = (frame: GpuTimingFrame) => {
    // An incomplete frame would understate the total; wait for the next one.
    if (frame.skipped) return;
    const validRows = frame.rows.filter(isValidGpuTimingRow);
    if (!validRows.length) return;
    if (discardedSamples < WARMUP_SAMPLES) {
      discardedSamples++;
      return;
    }
    rolling.add(validRows);
    statistics = rolling.snapshot();
    history.push({ ...frame, rows: validRows });
    if (history.length > GPU_TIMING_WINDOW) history.shift();
    const rows = new Map<string, GpuTimingRow>();
    for (const sample of history)
      for (const value of sample.rows) {
        const key = `${value.category}:${value.label}`;
        const row = rows.get(key) ?? { ...value, milliseconds: 0, calls: 0 };
        row.milliseconds += value.milliseconds / history.length;
        row.calls += value.calls / history.length;
        rows.set(key, row);
      }
    tree = createGpuTimingTree();
    updateGpuTimingTree(tree, rows.values());
  };
  const reset = () => {
    generation++;
    history.length = 0;
    discardedSamples = 0;
    rolling.reset();
    statistics = rolling.snapshot();
    nextSample = 0;
    tree = createGpuTimingTree();
  };

  return {
    reset,
    get active() {
      return active;
    },
    setActive(value: boolean) {
      if (value === active) return;
      active = value;
      failure = undefined;
      if (active && device && supported) profiler = new GpuProfiler(device);
      else {
        if (originalRender) renderer.render = originalRender;
        originalRender = undefined;
        capturing = false;
        profiler?.dispose();
        profiler = undefined;
      }
      reset();
    },
    snapshot(): GpuTimingsSnapshot {
      const state = !active
        ? 'off'
        : !supported
          ? 'unavailable'
          : failure
            ? 'failed'
            : history.length
              ? 'live'
              : 'waiting';
      return {
        state,
        tree,
        statisticsSamples: rolling.samples,
        statistics,
        message: failure,
      };
    },
    beginFrame(now: number) {
      if (!active || failure || disposed || now < nextSample || !profiler?.beginFrame()) return;
      const frameProfiler = profiler;
      capturing = true;
      frameGeneration = generation;
      nextSample = now + 200;
      // Name render passes after the scene they draw.
      originalRender = renderer.render;
      const render = originalRender;
      renderer.render = (scene: Object3D, camera: Camera) => {
        const previous = frameProfiler.renderLabel;
        frameProfiler.renderLabel =
          scene.name || ('material' in scene ? 'Post-processing' : 'Scene + fire');
        try {
          render.call(renderer, scene, camera);
        } finally {
          frameProfiler.renderLabel = previous;
        }
      };
    },
    endFrame() {
      if (!capturing || !profiler) return;
      capturing = false;
      if (originalRender) renderer.render = originalRender;
      originalRender = undefined;
      const captured = frameGeneration;
      void profiler
        .endFrame()
        .then((frame) => {
          if (frame && active && !disposed && captured === generation) show(frame);
        })
        .catch((error: unknown) => {
          if (active && !disposed && captured === generation)
            failure = error instanceof Error ? error.message : String(error);
        });
    },
    dispose() {
      disposed = true;
      generation++;
      if (originalRender) renderer.render = originalRender;
      profiler?.dispose();
    },
  };
}
