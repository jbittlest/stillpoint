/// <reference lib="webworker" />
/**
 * Analysis worker: openMp4 -> loadTelemetry -> buildPlan, off the UI thread. Keeps the Telemetry here and answers
 * re-plan requests (coalesced: while a plan is being built only the newest pending request is kept).
 */
import type { Telemetry } from '../types';
import type { StabParams } from '../ui/contracts';
import { openMp4 } from '../mp4';
import { loadTelemetry } from '../telemetry';
import { buildPlan } from '../plan';
import type { AnalysisIn, AnalysisOut, PlanSummary, TelemetrySummary } from './protocol';

type Body = AnalysisOut extends infer T ? (T extends unknown ? Omit<T, 'gen'> : never) : never;
/** current clip generation; results of an older clip are dropped */
let curGen = 0;
const postFor = (gen: number, m: Body, transfer: Transferable[] = []) => {
  if (gen !== curGen) return;
  (self as unknown as DedicatedWorkerGlobalScope).postMessage({ ...m, gen }, transfer);
};

let tel: Telemetry | null = null;
let ready: Promise<void> | null = null;
let busy = false;
let pending: { gen: number; id: number; params: StabParams } | null = null;

function median(a: ArrayLike<number>) {
  if (!a.length) return 0;
  const s = Array.from(a).sort((x, y) => x - y);
  return s[s.length >> 1];
}

function summarize(t: Telemetry): TelemetrySummary {
  return {
    camera: t.camera, width: t.width, height: t.height, fps: t.fps, frames: t.framePts.length, imuRate: t.imuRate,
    hasHighrate: t.hasHighrate, eisBaked: t.eisBaked, readoutS: t.readoutS, lensModel: t.lens.model, warnings: t.warnings,
    medianExposureS: median(t.exposureS),
  };
}

async function analyze(gen: number, file: File) {
  const post = (m: Body, t: Transferable[] = []) => postFor(gen, m, t);
  tel = null;
  let info;
  try {
    info = await openMp4(file);
  } catch (e) {
    post({ type: 'error', stage: 'open', message: (e as Error).message ?? String(e) });
    return;
  }
  post({ type: 'info', info });
  try {
    let last = 0;
    const t = await loadTelemetry(file, info, f => {
      const now = performance.now();
      if (now - last > 100 || f >= 1) { last = now; post({ type: 'progress', stage: 'telemetry', f }); }
    });
    if (gen !== curGen) return;
    tel = t;
    post({ type: 'telemetry', summary: summarize(tel) });
  } catch (e) {
    console.error(e);
    post({ type: 'error', stage: 'telemetry', message: (e as Error).message ?? String(e) });
  }
}

function planSummary(p: import('../types').Plan): PlanSummary {
  const fx = median(p.outFx);
  return { outW: p.outW, outH: p.outH, srcW: p.srcW, srcH: p.srcH, frames: p.framePts.length, outFx: fx, hfovDeg: (2 * Math.atan((p.outW / 2) / fx) * 180) / Math.PI };
}

async function runPlans() {
  if (busy) return;
  busy = true;
  try {
    while (pending) {
      const req = pending;
      pending = null;
      const post = (m: Body, t: Transferable[] = []) => postFor(req.gen, m, t);
      if (ready) await ready;
      if (req.gen !== curGen) continue;
      if (!tel) { post({ type: 'error', stage: 'plan', id: req.id, message: 'No telemetry loaded' }); continue; }
      const t0 = performance.now();
      try {
        let last = 0;
        const built = buildPlan(tel, req.params, f => {
          const now = performance.now();
          if (now - last > 100) { last = now; post({ type: 'progress', stage: 'plan', f, id: req.id }); }
        });
        const ms = performance.now() - t0;
        // copy the small arrays (they may alias Telemetry arrays); transfer rowMats when it owns its buffer
        const plan = { ...built, framePts: Float64Array.from(built.framePts), outFx: Float32Array.from(built.outFx), lens: { ...built.lens } };
        const ownsBuffer = plan.rowMats.byteOffset === 0 && plan.rowMats.byteLength === plan.rowMats.buffer.byteLength && plan.rowMats.buffer instanceof ArrayBuffer;
        if (!ownsBuffer) plan.rowMats = Float32Array.from(plan.rowMats);
        const summary = planSummary(plan);
        post({ type: 'plan', id: req.id, plan, summary, ms }, [plan.rowMats.buffer as ArrayBuffer, plan.framePts.buffer, plan.outFx.buffer]);
      } catch (e) {
        console.error(e);
        post({ type: 'error', stage: 'plan', id: req.id, message: (e as Error).message ?? String(e) });
      }
      // yield so newer requests that arrived meanwhile replace `pending`
      await new Promise(r => setTimeout(r, 0));
    }
  } finally {
    busy = false;
  }
}

self.onmessage = (ev: MessageEvent<AnalysisIn>) => {
  const m = ev.data;
  if (m.type === 'analyze') {
    curGen = m.gen;
    ready = analyze(m.gen, m.file);
  } else if (m.type === 'plan') {
    pending = { gen: m.gen, id: m.id, params: m.params };
    void runPlans();
  }
};
