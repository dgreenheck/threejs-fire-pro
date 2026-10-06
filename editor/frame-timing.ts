export interface FrameTiming {
  fps: number | null;
  frameTimeMs: number | null;
}

/** Rendered frame intervals, with separate windows for FPS and graph samples. */
export class FrameTimingHistory {
  private frames: { time: number; interval: number }[] = [];
  private started: number | null = null;

  record(time: number, interval: number): void {
    if (!Number.isFinite(time) || !Number.isFinite(interval) || interval <= 0) return;
    this.started ??= time - interval;
    this.frames.push({ time, interval });
    this.trim(time);
  }

  sample(now: number): FrameTiming {
    this.trim(now);
    if (this.started === null) return { fps: null, frameTimeMs: null };
    const elapsed = Math.min(1000, now - this.started);
    let total = 0;
    let count = 0;
    for (const frame of this.frames) {
      if (frame.time > now - 100) {
        total += frame.interval;
        count++;
      }
    }
    return {
      fps: elapsed > 0 ? (this.frames.length * 1000) / elapsed : null,
      frameTimeMs: count ? total / count : null,
    };
  }

  private trim(now: number): void {
    let expired = 0;
    while (expired < this.frames.length && this.frames[expired].time <= now - 1000) expired++;
    if (expired) this.frames.splice(0, expired);
  }
}
