import { useEffect, useState, type CSSProperties, type ReactNode, type RefObject } from 'react';
import { Activity, ChevronDown, ChevronRight } from 'lucide-react';
import type { RuntimeStats, SimulationRuntime } from '../runtime';
import type { GpuTimingNode } from '../gpu-timing-tree';
import { KERNEL_NOTES, type GpuTimingsSnapshot } from '../gpu-timings';
import { GPU_TIMING_WINDOW, type GpuTimingStatistics } from '../gpu-timing-history';
import type { FrameTiming } from '../frame-timing';

const countFormat = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 });
const exactCountFormat = new Intl.NumberFormat('en');
const memory = (bytes: number) =>
  bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GiB`
    : `${(bytes / 1024 ** 2).toFixed(1)} MiB`;

/** Own the refresh state here so metrics don't rerender the editor and inspector. */
export function PerformanceStats({ runtime }: { runtime: RefObject<SimulationRuntime | null> }) {
  const [stats, setStats] = useState<RuntimeStats | null>(null);
  const [profiling, setProfiling] = useState(false);
  const [profile, setProfile] = useState<GpuTimingsSnapshot | null>(null);
  useEffect(() => {
    const refresh = () => {
      const current = runtime.current;
      // A rebuilt runtime starts with the profiler off.
      current?.setProfiling(profiling);
      setStats(current?.stats ?? null);
      setProfile(profiling ? (current?.profile ?? null) : null);
    };
    refresh();
    const timer = window.setInterval(refresh, 250);
    return () => window.clearInterval(timer);
  }, [runtime, profiling]);
  useEffect(() => () => runtime.current?.setProfiling(false), [runtime]);
  const toggleProfiler = () => {
    const next = !profiling;
    setProfiling(next);
    runtime.current?.setProfiling(next);
  };
  const share = stats && stats.voxelBudget ? stats.activeVoxelCount / stats.voxelBudget : 0;
  return (
    <div className={`performance-stats ${profiling ? 'profiling' : ''}`}>
      <dl aria-label="Performance statistics">
        <FrameRateHeader runtime={runtime} />
        <div
          className={`stat-row ${stats?.gridLimited ? 'stat-warning' : ''}`}
          title={
            (stats
              ? `${exactCountFormat.format(stats.activeVoxelCount)} voxels computed near the fire, of a budget of ${exactCountFormat.format(stats.voxelBudget)}.`
              : 'Voxels computed near the fire, of the voxel budget.') +
            (stats?.gridLimited
              ? ' The simulation has reached its voxel budget; raise Max voxels or the voxel size.'
              : '')
          }
        >
          <dt>Voxels</dt>
          <dd aria-label="Total voxel count">
            {stats
              ? `${countFormat.format(stats.activeVoxelCount)} / ${countFormat.format(stats.voxelBudget)}`
              : '—'}
            {stats?.gridLimited && ' · at budget'}
          </dd>
          {/* The share of the budget the solver computed. */}
          <span className="stat-meter" aria-hidden="true">
            <span style={{ width: `${Math.min(100, share * 100)}%` }} />
          </span>
        </div>
        <div
          className="stat-row"
          title="GPU allocation for the simulation, volume lighting and volume slices. Excludes browser/driver overhead, temporary growth overlap and Three.js scene buffers."
        >
          <dt>Memory</dt>
          <dd aria-label="Estimated simulation memory">
            {stats ? `≈ ${memory(stats.estimatedMemoryBytes)}` : '—'}
          </dd>
        </div>
      </dl>
      <button
        type="button"
        className="profiler-toggle"
        aria-pressed={profiling}
        aria-expanded={profiling}
        title={profiling ? 'Stop measuring GPU passes' : 'Measure how long each GPU pass takes'}
        onClick={toggleProfiler}
      >
        <Activity size={12} />
        GPU profiler
        {profiling ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {profiling && <GpuProfile profile={profile} stats={stats} />}
    </div>
  );
}

const FRAME_GRAPH_WINDOW = 5000;
interface FrameSample {
  time: number;
  milliseconds: number | null;
}

/** Refresh only this header at 10 Hz; the GPU profiler keeps its slower cadence. */
function FrameRateHeader({ runtime }: { runtime: RefObject<SimulationRuntime | null> }) {
  const [state, setState] = useState<FrameTiming & { samples: FrameSample[]; now: number }>({
    fps: null,
    frameTimeMs: null,
    samples: [],
    now: 0,
  });
  useEffect(() => {
    let source: SimulationRuntime | null = null;
    const refresh = () => {
      const current = runtime.current;
      const changed = source !== current;
      source = current;
      const now = performance.now();
      const timing = current?.frameTiming ?? { fps: null, frameTimeMs: null };
      setState((previous) => ({
        ...timing,
        now,
        samples: [
          ...(changed
            ? []
            : previous.samples.filter((sample) => sample.time > now - FRAME_GRAPH_WINDOW)),
          { time: now, milliseconds: timing.frameTimeMs },
        ],
      }));
    };
    refresh();
    const timer = window.setInterval(refresh, 100);
    return () => window.clearInterval(timer);
  }, [runtime]);

  const { fps, frameTimeMs, samples, now } = state;
  const tone = fps === null ? '' : fps >= 55 ? 'good' : fps >= 30 ? 'fair' : 'slow';
  // Always show the 60 FPS reference; expand the scale to retain slow-frame spikes.
  const maximum = Math.max(33.3, ...samples.map((sample) => sample.milliseconds ?? 0));
  const ceiling = Math.ceil(maximum / 10) * 10;
  const y = (value: number) => 32 - (value / ceiling) * 30;
  let path = '';
  let connected = false;
  let lastTime = 0;
  for (const sample of samples) {
    if (sample.milliseconds === null) {
      connected = false;
      continue;
    }
    const x = 160 * (1 - (now - sample.time) / FRAME_GRAPH_WINDOW);
    const continuous = connected && sample.time - lastTime < 250;
    path += `${continuous ? 'L' : 'M'}${x.toFixed(1)},${y(sample.milliseconds).toFixed(1)} `;
    connected = true;
    lastTime = sample.time;
  }
  return (
    <div className="stat-headline">
      <div className="stat-fps-counter" title="Rendered frames per second over the last second.">
        <dt>FPS</dt>
        <dd aria-label="Frames per second" className={`stat-fps ${tone}`}>
          {fps?.toFixed(0) ?? '—'}
        </dd>
        <span className="stat-window">1 s average</span>
      </div>
      <div
        className="stat-frame-graph"
        title="Average frame interval in the preceding 100 ms, including simulation, rendering and browser scheduling. Dashed line: 60 FPS."
      >
        <dt>Frame time</dt>
        <dd aria-label="Frame time">
          {frameTimeMs === null ? '—' : `${frameTimeMs.toFixed(1)} ms`}
        </dd>
        <svg
          className="frame-time-chart"
          viewBox="0 0 160 36"
          preserveAspectRatio="none"
          role="img"
          aria-label={`Frame time over the last five seconds, sampled every 100 milliseconds. Vertical scale zero to ${ceiling} milliseconds; dashed line is 16.7 milliseconds, or 60 FPS.`}
        >
          <line className="frame-time-baseline" x1="0" x2="160" y1="32" y2="32" />
          <line
            className="frame-time-reference"
            x1="0" x2="160"
            y1={y(1000 / 60)} y2={y(1000 / 60)}
          />
          <path className="frame-time-line" d={path} />
        </svg>
        <span className="stat-graph-scale">
          <span>5 s · 0–{ceiling} ms</span>
          <span>now</span>
        </span>
      </div>
    </div>
  );
}

const profileMessages: Record<string, string> = {
  unavailable: 'This browser or GPU does not report GPU timestamps.',
  waiting: 'Waiting for GPU samples…',
};

function GpuTimingErrorBar({ statistics }: { statistics: GpuTimingStatistics | undefined }) {
  if (!statistics) return <span className="profile-history">—</span>;
  const { mean, standardDeviation, minimum, maximum, samples } = statistics;
  const range = maximum - minimum;
  const x = (value: number) =>
    range > 0 ? 4 + Math.max(0, Math.min(1, (value - minimum) / range)) * 152 : 80;
  const low = x(mean - standardDeviation);
  const high = x(mean + standardDeviation);
  const description = `${samples} samples: mean ${mean.toFixed(3)} ms, standard deviation ${standardDeviation.toFixed(3)} ms, min ${minimum.toFixed(3)} ms, max ${maximum.toFixed(3)} ms. Each whisker spans its full min–max range; the band shows mean ± one standard deviation, clipped to that range.`;
  return (
    <span className="profile-history" role="img" aria-label={description} title={description}>
      <svg viewBox="0 0 160 22" preserveAspectRatio="none" aria-hidden="true">
        <line className="profile-range" x1="4" x2="156" y1="11" y2="11" />
        <line className="profile-range" x1="4" x2="4" y1="4" y2="18" />
        <line className="profile-range" x1="156" x2="156" y1="4" y2="18" />
        <rect className="profile-deviation" x={low} y="6" width={high - low} height="10" rx="2" />
        <line className="profile-mean" x1={x(mean)} x2={x(mean)} y1="2" y2="20" />
      </svg>
      <span className="profile-history-values">
        <span>{minimum.toFixed(3)}</span>
        <span>{maximum.toFixed(3)}</span>
      </span>
    </span>
  );
}

/** GPU time by stage, then by group and kernel; click a group to open it. Idle work is hidden. */
function GpuProfile({
  profile,
  stats,
}: {
  profile: GpuTimingsSnapshot | null;
  stats: RuntimeStats | null;
}) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [open, setOpen] = useState<Set<string>>(
    () => new Set(['all', 'simulation', 'lighting', 'render']),
  );
  if (!profile || profile.state !== 'live')
    return (
      <p className="profiler-status" role="status">
        {profile?.state === 'failed'
          ? `GPU timing stopped: ${profile.message}`
          : profileMessages[profile?.state ?? 'waiting']}
      </p>
    );
  const total = profile.statistics.get('all')?.mean ?? 0;
  const copyJson = async () => {
    const metadata = new Map<string, GpuTimingNode>();
    const collect = (node: GpuTimingNode) => {
      metadata.set(node.id, node);
      node.children?.forEach(collect);
    };
    collect(profile.tree);
    const snapshot = {
      capturedAt: new Date().toISOString(),
      url: window.location.href,
      units: 'ms',
      window: GPU_TIMING_WINDOW,
      warmupSamplesDiscarded: 10,
      samplePolicy: 'Nonnegative finite timings per pass; population standard deviation.',
      statisticsSamples: profile.statisticsSamples,
      runtime: stats,
      rows: Array.from(profile.statistics, ([id, statistics]) => {
        const node = metadata.get(id);
        return {
          id,
          name: node?.label ?? id,
          category: node?.category,
          kernel: node?.kernel,
          averageCallsPerCapture: node?.calls,
          ...statistics,
        };
      }),
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(snapshot, null, 2));
      setCopyStatus('copied');
    } catch {
      setCopyStatus('failed');
    }
  };
  const toggle = (id: string) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const rows: ReactNode[] = [];
  const visit = (node: GpuTimingNode, depth: number) => {
    if (!node.calls) return;
    const expandable = Boolean(node.children?.some((child) => child.calls));
    const expanded = expandable && open.has(node.id);
    const statistics = profile.statistics.get(node.id);
    const average = statistics?.mean ?? 0;
    const deviation = statistics?.standardDeviation ?? 0;
    const share = total > 0 ? (average / total) * 100 : 0;
    const content = (
      <>
        <span className="profile-name">
          {expandable ? (
            expanded ? (
              <ChevronDown size={11} />
            ) : (
              <ChevronRight size={11} />
            )
          ) : (
            <span className="profile-spacer" />
          )}
          <span className="profile-label">{node.label}</span>
        </span>
        <span className="profile-ms">{average.toFixed(3)}</span>
        <span className="profile-std">{deviation.toFixed(3)}</span>
        <GpuTimingErrorBar statistics={statistics} />
      </>
    );
    const style = { '--depth': depth } as CSSProperties;
    rows.push(
      expandable ? (
        <button
          key={node.id}
          type="button"
          className={`profile-row depth-${depth}`}
          style={style}
          aria-expanded={expanded}
          title={`${node.label}: ${average.toFixed(3)} ms average, ${deviation.toFixed(3)} ms standard deviation, ${share.toFixed(1)}% of the GPU time over ${statistics?.samples ?? 0} valid samples`}
          onClick={() => toggle(node.id)}
        >
          {content}
        </button>
      ) : (
        <div
          key={node.id}
          className={`profile-row depth-${depth}`}
          style={style}
          title={`${KERNEL_NOTES[node.kernel ?? ''] ?? node.label} ${node.calls.toFixed(1)} calls, ${average.toFixed(3)} ms average, ${deviation.toFixed(3)} ms standard deviation over ${statistics?.samples ?? 0} valid samples.`}
        >
          {content}
        </div>
      ),
    );
    if (expanded) node.children!.forEach((child) => visit(child, depth + 1));
  };
  visit(profile.tree, 0);
  return (
    <div className="profiler" aria-label="GPU time per frame">
      <div className="profiler-heading">
        <span>GPU timings · ms</span>
        <span>
          {profile.statisticsSamples.toLocaleString()} / {GPU_TIMING_WINDOW.toLocaleString()}{' '}
          samples
        </span>
        <button
          type="button"
          className="profiler-copy"
          onClick={copyJson}
          title="Copy statistics for every GPU pass as JSON, including collapsed rows"
        >
          {copyStatus === 'copied' ? 'Copied!' : 'Copy JSON'}
        </button>
      </div>
      {copyStatus === 'failed' && (
        <p className="profiler-status" role="status">
          Clipboard access failed. Try Copy JSON again.
        </p>
      )}
      <div className="profiler-legend">White: range · Blue: ±1σ · Orange: mean</div>
      <div className="profile-tree">
        <div className="profile-column-headings">
          <span>Pass</span>
          <span className="profile-ms">Avg</span>
          <span className="profile-std">Std dev</span>
          <span className="profile-distribution-heading">Range / ±1σ</span>
        </div>
        {rows}
      </div>
    </div>
  );
}
