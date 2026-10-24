/** Browser render cadence, separate from GPU duration and simulation throughput. */
export interface FrameReport {
  intervalsMs: number[];
  durationMs: number;
  fps: number;
  medianMs: number;
  p95Ms: number;
  over25Ms: number;
  simulationSeconds: number;
  simulationRate: number;
}
export function frameReport(intervalsMs: readonly number[], simulationSeconds: number): FrameReport {
  if (!intervalsMs.length || intervalsMs.some(value => !Number.isFinite(value) || value <= 0)
    || !Number.isFinite(simulationSeconds) || simulationSeconds < 0) throw new Error('Invalid browser frame sample');
  const sorted = [...intervalsMs].sort((a, b) => a - b);
  const durationMs = intervalsMs.reduce((sum, value) => sum + value, 0);
  return { intervalsMs: [...intervalsMs], durationMs, fps: intervalsMs.length * 1000 / durationMs,
    medianMs: sorted[Math.ceil(sorted.length * .5) - 1], p95Ms: sorted[Math.ceil(sorted.length * .95) - 1],
    over25Ms: intervalsMs.filter(value => value > 25).length,
    simulationSeconds, simulationRate: simulationSeconds * 1000 / durationMs };
}

/** A finite capture includes one second of warmup, then four seconds of actual render intervals. */
export class FrameCapture {
  readonly warmupMs = 1000;
  readonly sampleMs = 4000;
  private previous: number | null = null;
  private intervals: number[] = [];
  private simulated = 0;
  private done = false;
  readonly startedAt: number;
  constructor(startedAt: number) {
    if (!Number.isFinite(startedAt) || startedAt < 0) throw new Error('Invalid browser capture start');
    this.startedAt = startedAt;
  }
  record(time: number, simulatedSeconds: number): FrameReport | null {
    if (this.done) throw new Error('Browser capture already finished');
    if (!Number.isFinite(time) || time < this.startedAt || !Number.isFinite(simulatedSeconds) || simulatedSeconds < 0)
      throw new Error('Invalid browser capture frame');
    if (time < this.startedAt + this.warmupMs) return null;
    if (this.previous !== null) {
      const interval = time - this.previous;
      if (interval <= 0) throw new Error('Browser capture timestamps must increase');
      this.intervals.push(interval); this.simulated += simulatedSeconds;
    }
    this.previous = time;
    if (time < this.startedAt + this.warmupMs + this.sampleMs) return null;
    this.done = true;
    return frameReport(this.intervals, this.simulated);
  }
}

export interface BrowserFrameReport extends FrameReport {
  context: Record<string, unknown>;
  measuredAt: string;
  userAgent: string;
  viewport: { width: number; height: number; devicePixelRatio: number; drawingWidth: number; drawingHeight: number };
  warmupMs: number;
  requestedSampleMs: number;
  gpuDrainMs: number;
  fpsIncludingDrain: number;
  scope: string;
}

/** A stalled device must not leave an explicit capture disabled indefinitely. */
export async function waitForGpuDrain(queue: Pick<GPUQueue, 'onSubmittedWorkDone'>, timeoutMs = 5000): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([queue.onSubmittedWorkDone(), new Promise<never>((_, reject) => {
      timeout = setTimeout(() => reject(new Error(`GPU queue did not finish within ${timeoutMs} ms`)), timeoutMs);
    })]);
  } finally { clearTimeout(timeout); }
}

/** Explicit opt-in capture; no background benchmark, automatic download or pose readback. */
export function browserPerformance(options: {
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context(): Record<string, unknown>;
  start(): boolean;
  pause(): void;
}) {
  const button = document.querySelector<HTMLButtonElement>('#measure')!;
  const output = document.querySelector<HTMLOutputElement>('#performance')!;
  const data = document.querySelector<HTMLPreElement>('#performance-data')!;
  const save = document.querySelector<HTMLButtonElement>('#save-performance')!;
  let capture: FrameCapture | null = null, watchdog: ReturnType<typeof setTimeout> | undefined;
  let context: Record<string, unknown> = {}, viewport: BrowserFrameReport['viewport'];
  let report: BrowserFrameReport | null = null, finishing = false;
  let previous: number | null = null, live: number[] = [], lastLabel = 0, revision = 0;
  const stopTimer = () => { clearTimeout(watchdog); watchdog = undefined; };
  const clear = () => { previous = null; live = []; };
  function cancel(reason: string) {
    revision++;
    if (!capture) return;
    capture = null; stopTimer(); button.disabled = false; button.textContent = 'Measure 4s';
    output.textContent = reason; clear();
  }
  function reset(reason: string) {
    cancel(reason); report = null; data.textContent = 'No capture yet.'; save.disabled = true;
    output.textContent = reason; clear();
  }
  save.addEventListener('click', () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `sdf-${report.context.demo}-frames.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  button.addEventListener('click', () => {
    if (capture || finishing || !options.start()) return;
    context = options.context();
    viewport = { width: innerWidth, height: innerHeight, devicePixelRatio,
      drawingWidth: options.canvas.width, drawingHeight: options.canvas.height };
    report = null; save.disabled = true; data.textContent = ''; capture = new FrameCapture(performance.now()); clear();
    button.disabled = true; button.textContent = 'Warming up…'; output.textContent = '1s warmup · 4s sample · then pause';
    watchdog = setTimeout(() => { cancel('Capture stopped: no complete visible frame sample.'); options.pause(); }, 12000);
  });
  async function finish(sample: FrameReport, measured: FrameCapture) {
    const version = revision;
    capture = null; finishing = true; stopTimer(); options.pause(); clear();
    output.textContent = 'Finishing GPU work…';
    const began = performance.now();
    try {
      await waitForGpuDrain(options.device.queue);
      if (version !== revision) return;
      const gpuDrainMs = performance.now() - began;
      report = { ...sample, context, measuredAt: new Date().toISOString(), userAgent: navigator.userAgent, viewport,
        warmupMs: measured.warmupMs, requestedSampleMs: measured.sampleMs, gpuDrainMs,
        fpsIncludingDrain: sample.intervalsMs.length * 1000 / (sample.durationMs + gpuDrainMs),
        scope: 'Visible browser render submissions at requestAnimationFrame timestamps. Includes simulation, rendering, UI and enabled overlays. Not GPU duration or verified compositor presentation. Simulation rate is fixed-step simulated seconds / sample wall seconds; GPU drain is recorded separately.' };
      output.textContent = `${report.fps.toFixed(1)} FPS · p95 ${report.p95Ms.toFixed(1)} ms · simulation ${report.simulationRate.toFixed(2)}×`;
      data.textContent = JSON.stringify(report, null, 2);
      save.disabled = false;
    } catch (error) { if (version === revision) output.textContent = `Capture failed: ${String(error)}`; }
    finally { finishing = false; button.disabled = false; button.textContent = 'Measure 4s'; }
  }
  return {
    get active() { return capture !== null || finishing; },
    get report() { return report; },
    cancel, reset,
    idle() { clear(); },
    frame(time: number, simulatedSeconds: number) {
      if (previous !== null && time > previous) { live.push(time - previous); if (live.length > 90) live.shift(); }
      previous = time;
      if (capture) {
        const measured = capture;
        let sample: FrameReport | null;
        try { sample = measured.record(time, simulatedSeconds); }
        catch { cancel('Capture stopped: insufficient valid frame samples.'); options.pause(); return; }
        button.textContent = time < measured.startedAt + measured.warmupMs ? 'Warming up…' : 'Measuring…';
        if (sample) void finish(sample, measured);
      } else if (!finishing && time - lastLabel > 500 && live.length > 1) {
        lastLabel = time;
        output.textContent = `${(live.length * 1000 / live.reduce((a, b) => a + b, 0)).toFixed(0)} FPS · browser cadence`;
      }
    },
  };
}
