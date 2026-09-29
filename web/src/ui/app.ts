/**
 * Stillpoint web UI controller (main thread). Owns no heavy work: the analysis worker parses the file and builds the
 * camera plan; the engine worker owns the GPU, the preview canvases and the export pipeline.
 */
import type { Mp4Info, Mp4Track, Plan } from '../types';
import type { StabParams } from './contracts';
import type { AnalysisIn, AnalysisOut, DecoderInfo, EngineDebug, EngineIn, EngineOut, ErrorDetails, PlanSummary, TelemetrySummary } from '../io/protocol';
import type { RenderProgress, RenderResult } from '../io/pipeline';
import type { SinkKind } from '../io/mux';
import { getOpfsExport } from '../io/mux';
import { bitratePresets } from '../io/encode';
import { detectCaps, verdict, type Caps } from './caps';
import { aspectLabel, fmtBytes, fmtDuration, fmtEta, fmtFps, fmtRate, fmtShutter, fmtTime } from './format';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type Quality = 'balanced' | 'high' | 'max';

interface ClipState {
  file: File;
  info?: Mp4Info;
  video?: Mp4Track;
  audio?: Mp4Track;
  tel?: TelemetrySummary;
  telError?: string;
  frames: number;
  fps: number;
  duration: number;
  pts?: Float64Array;
  plan?: PlanSummary;
  planId: number;
  appliedPlanId: number;
  encoder?: { codec: string | null; label: string; hardware: boolean; width: number; height: number; fps: number };
  inP: number;
  outP: number;
  pres: number;
  /** the engine's video decoder for this clip (hardware / software, why) */
  decoder?: DecoderInfo;
  /** the clip can't be decoded on this computer (friendly reason): no preview, no export */
  decodeError?: string;
  /** engine notes about the file (e.g. incomplete copy) */
  engineNotes?: string[];
}

/** test / debugging hook */
export interface DebugHook {
  view: string;
  caps?: Caps;
  gpu?: { webgpu: boolean; adapter: string; error?: string };
  clip?: { name: string; frames: number; fps: number; telemetry?: TelemetrySummary; plan?: PlanSummary; encoder?: ClipState['encoder']; planMs?: number };
  planReady: boolean;
  exportState: 'idle' | 'running' | 'done' | 'error';
  progress?: RenderProgress;
  result?: RenderResult;
  error?: string;
  events: string[];
  lastFrame?: { pres: number; ms: number; hasWarp: boolean };
  playStats?: { drawn: number; dropped: number; seconds: number };
  downloadUrl?: string;
  /** the decoder in use (from the engine) */
  decoder?: DecoderInfo;
  /** diagnostics attached to the last error (what "Copy details" copies, with browser facts) */
  errorDetails?: ErrorDetails;
  /** the last diagnostics text built by "Copy details" */
  lastDiagnostics?: string;
  playing?: boolean;
}

export class App {
  private analysis!: Worker;
  private engine!: Worker;
  /** ?sink=opfs|memory|fsa forces a sink; 'fsa-test' = the File System Access code path with an OPFS handle (headless tests) */
  private forcedSink = new URLSearchParams(location.search).get('sink') as SinkKind | 'fsa-test' | null;
  private wakeLock: { release(): Promise<void> } | null = null;
  private caps?: Caps;
  private clip: ClipState | null = null;
  /** clip generation (bumped per opened file) */
  private gen = 0;
  private planSeq = 0;
  private quality: Quality = 'high';
  private codec: 'hevc' | 'avc' = 'hevc';
  private params: StabParams = { smoothness: 1, footprint: 0.6, horizonLock: false };
  private planTimer = 0;
  private seekBusy = false;
  private seekPending: number | null = null;
  private seekTimer = 0;
  private playing = false;
  private exporting = false;
  private downloadUrl: string | null = null;
  private aspect = 16 / 9;
  private errorDetails: ErrorDetails | null = null;
  private uaData: Record<string, unknown> | null = null;
  readonly debug: DebugHook = { view: 'boot', planReady: false, exportState: 'idle', events: [] };

  async start() {
    (window as any).__stillpoint = this.debug;
    // test hooks (headless e2e)
    (window as any).__sp_seek = (p: number) => { this.pause(); this.seek(p); };
    (window as any).__sp_params = (p: Partial<StabParams>) => { this.params = { ...this.params, ...p }; this.requestPlan(true); };
    (window as any).__sp_range = (a: number, b: number) => {
      const c = this.clip; if (!c) return;
      c.inP = Math.max(0, Math.min(c.frames - 2, a)); c.outP = Math.max(c.inP + 1, Math.min(c.frames - 1, b));
      this.renderTimeline(); this.renderExportFacts();
    };
    window.addEventListener('beforeunload', e => { if (this.exporting) { e.preventDefault(); e.returnValue = ''; } });
    void this.loadUaData();
    this.bindLanding();
    this.bindWorkspace();
    this.startWorkers();
    this.caps = await detectCaps();
    this.debug.caps = this.caps;
    this.showCaps(this.caps);
    this.setView('landing');
  }

  private log(e: string) {
    this.debug.events.push(`${(performance.now() / 1000).toFixed(2)} ${e}`);
    if (this.debug.events.length > 400) this.debug.events.splice(0, 100);
  }

  private setView(v: 'landing' | 'work') {
    document.body.dataset.view = v;
    this.debug.view = v;
    $('landing').hidden = v !== 'landing';
    $('workspace').hidden = v !== 'work';
    $('btn-open-another').hidden = v !== 'work';
    if (v === 'work') requestAnimationFrame(() => this.resizeCanvases());
  }

  // ───────────────────────── workers ─────────────────────────

  private startWorkers() {
    this.analysis = new Worker(new URL('../io/analysis.worker.ts', import.meta.url), { type: 'module', name: 'stillpoint-analysis' });
    this.engine = new Worker(new URL('../io/engine.worker.ts', import.meta.url), { type: 'module', name: 'stillpoint-engine' });
    this.analysis.onmessage = e => this.onAnalysis(e.data as AnalysisOut);
    this.engine.onmessage = e => this.onEngine(e.data as EngineOut);
    this.analysis.onerror = e => this.fail('The analysis worker crashed: ' + (e.message || 'unknown error'));
    this.engine.onerror = e => this.fail('The render engine crashed: ' + (e.message || 'unknown error'));
    const before = ($('cv-before') as HTMLCanvasElement).transferControlToOffscreen();
    const after = ($('cv-after') as HTMLCanvasElement).transferControlToOffscreen();
    // test / support switches: ?sp_fault=hw-first (simulate a failing hardware decoder), ?sp_stall=3000 (watchdog ms)
    const q = new URLSearchParams(location.search);
    const debug: EngineDebug = {};
    if (q.get('sp_fault')) debug.fault = q.get('sp_fault')!;
    if (q.get('sp_stall')) debug.stallMs = Math.max(500, +q.get('sp_stall')! || 0);
    this.postEngine({ type: 'init', before, after, debug }, [before, after]);
  }

  private postEngine(m: EngineIn, transfer: Transferable[] = []) { this.engine.postMessage(m, transfer); }
  private postAnalysis(m: AnalysisIn) { this.analysis.postMessage(m); }

  private onAnalysis(m: AnalysisOut) {
    const c = this.clip;
    if (!c || m.gen !== this.gen) return; // a previous clip's late result
    switch (m.type) {
      case 'info': {
        this.log('info');
        c.info = m.info;
        c.video = m.info.tracks.find(t => t.kind === 'video' && !!t.codecString && (t.width ?? 0) >= 320) ?? m.info.tracks.find(t => t.kind === 'video');
        c.audio = m.info.tracks.find(t => t.kind === 'audio' && t.sampleCount > 0);
        if (c.video?.width && c.video.height) this.setAspect(c.video.width / c.video.height);
        this.postEngine({ type: 'open', file: c.file, info: m.info });
        this.renderFacts();
        this.status('Reading gyro…', 0);
        break;
      }
      case 'progress':
        if (m.stage === 'telemetry') this.status('Reading gyro…', m.f);
        else if (m.id === c.planId) { this.status('Planning the camera path…', m.f); }
        break;
      case 'telemetry':
        this.log('telemetry');
        c.tel = m.summary;
        this.debug.clip = { ...this.debug.clip!, telemetry: m.summary };
        this.renderBadge();
        this.renderFacts();
        if (!c.decodeError) this.requestPlan(true);
        break;
      case 'plan': {
        if (m.id !== c.planId) return; // stale
        this.log(`plan ${m.id} ${m.ms.toFixed(0)} ms`);
        c.plan = m.summary;
        this.debug.clip = { ...this.debug.clip!, plan: m.summary, planMs: m.ms };
        this.setAspect(m.summary.outW / m.summary.outH);
        const p: Plan = m.plan;
        this.postEngine({ type: 'plan', plan: p, id: m.id }, [p.rowMats.buffer as ArrayBuffer]);
        this.setPlanState(`${m.summary.outW}×${m.summary.outH} · ≈${Math.round(m.summary.hfovDeg)}° wide`, false);
        this.renderExportFacts();
        this.postEngine({ type: 'probe-encoder', bitrate: this.bitrate(), prefer: this.codec });
        break;
      }
      case 'error':
        this.log(`analysis error ${m.stage}: ${m.message}`);
        if (m.stage === 'open') {
          // DJI cameras write the file's index (moov) last: a copy that stopped early has none
          const incomplete = /no moov|truncat|unexpected end|beyond the end/i.test(m.message);
          this.fail(incomplete
            ? `This file looks incomplete — it may not have finished copying from the card, or the recording was cut off (the camera writes the file’s index at the very end). Copy the clip from the SD card again. (${m.message})`
            : `Couldn’t read this file: ${m.message}`, { stage: 'analysis-open', error: { name: 'Error', message: m.message } });
          this.status(null);
        }
        else if (m.stage === 'telemetry') {
          c.telError = m.message;
          this.renderBadge();
          this.renderFacts();
          this.status(null);
        } else if (m.id === c.planId) {
          this.setPlanState('Planning failed', false);
          this.fail(`Couldn’t plan a stabilized path: ${m.message}`);
          this.status(null);
        }
        break;
    }
  }

  private onEngine(m: EngineOut) {
    switch (m.type) {
      case 'ready':
        this.debug.gpu = m.caps;
        this.log(`engine ready webgpu=${m.caps.webgpu} ${m.caps.adapter} ${m.caps.error ?? ''}`);
        if (!m.caps.webgpu && this.caps?.webgpu) this.toast('WebGPU could not start in the render worker: ' + (m.caps.error ?? 'unknown'));
        break;
      case 'opened': {
        const c = this.clip; if (!c) return;
        const d = m.decoder;
        this.log(`opened ${m.frames} frames, decoder ${d.codec} ${d.variant} (${d.label}) sw=${d.software} fallback=${d.fallback} keyframes=${JSON.stringify(d.rap)} preflight=${d.preflight ? `${d.preflight.ms} ms ${d.preflight.tried.map(t => `${t.variant}:${t.ok ? 'ok' : 'fail'}`).join(',')}` : '-'}`);
        c.decoder = d; this.debug.decoder = d;
        c.engineNotes = m.notes;
        if (d.note) this.toast(d.software ? 'Using software video decoding for this clip (slower) — details in the clip notes.' : d.note);
        c.frames = m.frames; c.fps = m.fps; c.duration = m.duration;
        c.inP = 0; c.outP = m.frames - 1;
        if (c.video) c.pts = Float64Array.from(c.video.cts).sort();
        this.debug.clip = { ...this.debug.clip!, frames: m.frames, fps: m.fps };
        ($('btn-play') as HTMLButtonElement).disabled = false;
        this.renderTimeline();
        this.renderFacts();
        break;
      }
      case 'decoder': {
        const c = this.clip; if (!c) return;
        const was = c.decoder;
        c.decoder = m.decoder; this.debug.decoder = m.decoder;
        this.log(`decoder switched ${was?.variant ?? '?'} -> ${m.decoder.variant} (${m.decoder.label})`);
        if (m.decoder.software && !was?.software) this.toast('The hardware video decoder failed — switched to software decoding (slower).');
        this.renderFacts();
        break;
      }
      case 'open-error': {
        this.status(null);
        const c = this.clip;
        if (c) {
          c.decodeError = m.message;
          clearTimeout(this.planTimer);
          ($('btn-play') as HTMLButtonElement).disabled = true;
          this.setPlanState('', false);
          this.renderFacts();
          this.updateExportButton();
        }
        this.fail(m.message, m.details ?? { stage: 'open' });
        break;
      }
      case 'frame': {
        const c = this.clip; if (!c) return;
        c.pres = m.pres;
        this.debug.lastFrame = { pres: m.pres, ms: m.ms, hasWarp: m.hasWarp };
        this.renderPlayhead();
        if (!this.playing && !this.exporting) this.seekDone();
        break;
      }
      case 'playing':
        this.playing = m.playing;
        this.debug.playing = m.playing;
        if (m.stats) { this.debug.playStats = m.stats; this.log(`play stopped: ${m.stats.drawn} drawn, ${m.stats.dropped} dropped in ${m.stats.seconds.toFixed(2)} s`); }
        $('btn-play').classList.toggle('is-playing', m.playing);
        $('btn-play').setAttribute('aria-label', m.playing ? 'Pause preview' : 'Play preview');
        break;
      case 'plan-applied': {
        const c = this.clip; if (!c) return;
        if (m.id > c.planId) return;
        c.appliedPlanId = Math.max(c.appliedPlanId, m.id);
        if (m.id === c.planId) { this.debug.planReady = true; this.status(null); $('viewer').classList.remove('no-split'); }
        this.updateExportButton();
        break;
      }
      case 'encoder': {
        const c = this.clip; if (!c) return;
        c.encoder = m;
        this.debug.clip = { ...this.debug.clip!, encoder: m };
        this.renderExportFacts();
        this.updateExportButton();
        break;
      }
      case 'export-progress':
        this.debug.progress = m.p;
        this.renderProgress(m.p);
        break;
      case 'export-done':
        this.log(`export done ${m.result.frames} frames ${m.result.fps.toFixed(1)} fps`);
        void this.exportDone(m.result);
        break;
      case 'export-error':
        this.log(`export error: ${m.message}`);
        this.exporting = false;
        this.releaseWakeLock();
        document.title = 'Stillpoint — stabilize DJI footage in your browser';
        this.debug.exportState = m.cancelled ? 'idle' : 'error';
        if (!m.cancelled) this.fail(m.message, m.details ?? { stage: 'export' });
        else { this.showCard('export'); this.toast('Export cancelled.'); }
        this.lockControls(false);
        this.updateExportButton();
        break;
      case 'error':
        this.log(`engine error: ${m.message}`);
        this.seekDone();
        // decode failures come with diagnostics: show them in the error card (with "Copy details"), not a toast
        if (m.details && !this.exporting) this.fail(m.message, m.details);
        else this.toast(m.message);
        break;
    }
  }

  // ───────────────────────── landing ─────────────────────────

  private showCaps(c: Caps) {
    const set = (k: string, v: 'yes' | 'no' | 'partial', label?: string) => {
      const el = document.querySelector<HTMLElement>(`.cap[data-cap="${k}"]`)!;
      el.dataset.ok = v;
      if (label) el.querySelector('span')!.textContent = label;
    };
    set('gpu', c.webgpu ? 'yes' : 'no');
    set('hevc', c.decodeHevc10 ? 'yes' : c.decodeH264 ? 'partial' : 'no', c.decodeHevc10 ? 'HEVC 10-bit decode' : c.decodeH264 ? 'H.264 decode only' : 'No video decoder');
    set('enc', c.encodeHevc ? 'yes' : c.encodeH264 ? 'yes' : c.encodeAny ? 'partial' : 'no', c.encodeHevc ? 'HEVC encode' : c.encodeH264 ? 'H.264 encode' : c.encodeAny ? 'Software encode' : 'No encoder');
    set('save', c.saveToDisk ? 'yes' : c.opfs ? 'partial' : 'partial', c.saveToDisk ? 'Streams to disk' : 'Downloads when done');
    const v = verdict(c);
    const box = $('unsupported');
    if (!v.ok || v.partial) {
      box.hidden = false;
      box.classList.toggle('is-warn', v.ok);
      $('unsupported-title').textContent = v.title;
      $('unsupported-body').innerHTML = v.body;
    }
    if (!v.ok) {
      $('drop').setAttribute('aria-disabled', 'true');
      ($('file-input') as HTMLInputElement).disabled = true;
      $('drop').style.opacity = '0.55';
    }
  }

  private bindLanding() {
    const input = $('file-input') as HTMLInputElement;
    input.addEventListener('change', () => { const f = input.files?.[0]; if (f) this.openFile(f); input.value = ''; });
    // drag & drop anywhere
    let depth = 0;
    const veil = $('drag-veil');
    const hasFiles = (e: DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files');
    window.addEventListener('dragenter', e => { if (!hasFiles(e)) return; e.preventDefault(); depth++; document.body.classList.add('dragging'); if (this.debug.view === 'work') veil.hidden = false; });
    window.addEventListener('dragover', e => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer!.dropEffect = this.exporting ? 'none' : 'copy'; });
    window.addEventListener('dragleave', e => { if (!hasFiles(e)) return; depth = Math.max(0, depth - 1); if (!depth) { document.body.classList.remove('dragging'); veil.hidden = true; } });
    window.addEventListener('drop', e => {
      if (!hasFiles(e)) return;
      e.preventDefault(); depth = 0; document.body.classList.remove('dragging'); veil.hidden = true;
      if (this.exporting) { this.toast('Finish or cancel the current export first.'); return; }
      if (this.caps && !verdict(this.caps).ok) return;
      const f = e.dataTransfer!.files[0];
      if (f) this.openFile(f);
    });
    const another = () => { if (this.exporting) { this.toast('Finish or cancel the current export first.'); return; } input.click(); };
    $('btn-open-another').addEventListener('click', another);
    $('btn-another').addEventListener('click', another);
  }

  // ───────────────────────── open a clip ─────────────────────────

  openFile(file: File) {
    if (!/\.(mp4|mov|m4v)$/i.test(file.name) && !/^video\//.test(file.type)) {
      this.toast(`“${file.name}” doesn’t look like a video file.`);
      return;
    }
    this.log(`open ${file.name} ${file.size}`);
    this.clip = { file, frames: 0, fps: 0, duration: 0, planId: 0, appliedPlanId: -1, inP: 0, outP: 0, pres: 0 };
    this.debug.clip = { name: file.name, frames: 0, fps: 0 };
    this.debug.planReady = false;
    this.debug.exportState = 'idle';
    this.debug.result = undefined; this.debug.error = undefined; this.debug.progress = undefined;
    this.debug.decoder = undefined; this.debug.errorDetails = undefined; this.errorDetails = null;
    this.playing = false;
    $('clip-name').textContent = file.name;
    $('clip-meta').innerHTML = '';
    $('facts').innerHTML = '';
    $('warnings').innerHTML = '';
    $('card-notes').hidden = true;
    $('export-facts').innerHTML = '';
    $('viewer').classList.add('no-split');
    ($('btn-play') as HTMLButtonElement).disabled = true;
    this.setPlanState('', false);
    this.renderBadge();
    this.showCard('export');
    this.updateExportButton();
    this.setView('work');
    this.status('Opening…', 0);
    this.postAnalysis({ type: 'analyze', gen: ++this.gen, file });
  }

  private requestPlan(immediate = false) {
    const c = this.clip;
    if (!c || !c.tel || c.decodeError) return;
    clearTimeout(this.planTimer);
    if (c.tel.eisBaked) { this.setPlanState('Not available for this clip', false); this.status(null); return; }
    const go = () => {
      c.planId = ++this.planSeq; // globally increasing, so a stale reply from an older clip can never match
      this.debug.planReady = false;
      this.setPlanState('Planning…', true);
      this.updateExportButton();
      this.postAnalysis({ type: 'plan', gen: this.gen, id: c.planId, params: { ...this.params } });
    };
    if (immediate) go(); else this.planTimer = window.setTimeout(go, 220);
  }

  // ───────────────────────── workspace UI ─────────────────────────

  private bindWorkspace() {
    // split slider: drag anywhere on the picture
    const viewer = $('viewer');
    const split = $('split');
    const setSplit = (f: number) => {
      f = Math.max(0, Math.min(1, f));
      viewer.style.setProperty('--split', `${(f * 100).toFixed(2)}%`);
      split.setAttribute('aria-valuenow', String(Math.round(f * 100)));
    };
    let dragging = false;
    viewer.addEventListener('pointerdown', e => {
      if (viewer.classList.contains('no-split')) return;
      dragging = true; split.classList.add('is-drag'); viewer.setPointerCapture(e.pointerId);
      const r = viewer.getBoundingClientRect(); setSplit((e.clientX - r.left) / r.width);
    });
    viewer.addEventListener('pointermove', e => { if (!dragging) return; const r = viewer.getBoundingClientRect(); setSplit((e.clientX - r.left) / r.width); });
    const end = () => { dragging = false; split.classList.remove('is-drag'); };
    viewer.addEventListener('pointerup', end); viewer.addEventListener('pointercancel', end);
    split.addEventListener('keydown', e => {
      const cur = parseFloat(viewer.style.getPropertyValue('--split')) / 100 || 0.5;
      if (e.key === 'ArrowLeft') { setSplit(cur - 0.05); e.preventDefault(); e.stopPropagation(); }
      if (e.key === 'ArrowRight') { setSplit(cur + 0.05); e.preventDefault(); e.stopPropagation(); }
    });
    new ResizeObserver(() => this.resizeCanvases()).observe(viewer);

    // timeline
    const tl = $('timeline');
    let mode: 'seek' | 'in' | 'out' | null = null;
    const presAt = (x: number) => {
      const c = this.clip; if (!c || c.frames < 1) return 0;
      const r = tl.getBoundingClientRect();
      return Math.round(Math.max(0, Math.min(1, (x - r.left) / r.width)) * (c.frames - 1));
    };
    tl.addEventListener('pointerdown', e => {
      const c = this.clip; if (!c || !c.frames) return;
      const t = e.target as HTMLElement;
      if (this.exporting && !t.classList.contains('tl-handle')) return;
      mode = t.id === 'tl-in' ? 'in' : t.id === 'tl-out' ? 'out' : 'seek';
      if (this.exporting && mode !== 'seek') { mode = null; return; }
      tl.setPointerCapture(e.pointerId);
      if (mode === 'seek') { this.pause(); this.seek(presAt(e.clientX)); }
    });
    tl.addEventListener('pointermove', e => {
      const c = this.clip; if (!mode || !c) return;
      const p = presAt(e.clientX);
      if (mode === 'seek') this.seek(p);
      else if (mode === 'in') { c.inP = Math.min(p, c.outP - 1); this.renderTimeline(); this.renderExportFacts(); this.seek(c.inP); }
      else { c.outP = Math.max(p, c.inP + 1); this.renderTimeline(); this.renderExportFacts(); this.seek(c.outP); }
    });
    const tlEnd = () => { mode = null; };
    tl.addEventListener('pointerup', tlEnd); tl.addEventListener('pointercancel', tlEnd);

    // transport
    $('btn-play').addEventListener('click', () => this.togglePlay());
    window.addEventListener('keydown', e => {
      if (this.debug.view !== 'work' || !this.clip) return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if ((tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY') && (e.key === ' ' || e.key === 'Enter')) return; // let the control act
      const c = this.clip;
      if (e.key === ' ') { e.preventDefault(); this.togglePlay(); }
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault(); this.pause();
        const step = e.shiftKey ? Math.round(c.fps || 30) : 1;
        this.seek(c.pres + (e.key === 'ArrowLeft' ? -step : step));
      } else if ((e.key === 'i' || e.key === 'I') && !this.exporting) { c.inP = Math.min(c.pres, c.outP - 1); this.renderTimeline(); this.renderExportFacts(); }
      else if ((e.key === 'o' || e.key === 'O') && !this.exporting) { c.outP = Math.max(c.pres, c.inP + 1); this.renderTimeline(); this.renderExportFacts(); }
    });

    // stabilization controls
    const smooth = $('in-smooth') as HTMLInputElement, fov = $('in-fov') as HTMLInputElement, hz = $('in-horizon') as HTMLInputElement;
    const fill = (el: HTMLInputElement) => el.style.setProperty('--fill', `${((+el.value - +el.min) / (+el.max - +el.min)) * 100}%`);
    const sync = () => {
      this.params = { smoothness: +smooth.value, footprint: +fov.value, horizonLock: hz.checked };
      $('out-smooth').textContent = (+smooth.value).toFixed(2);
      $('out-fov').textContent = `${Math.round(+fov.value * 100)}% of sensor`;
      fill(smooth); fill(fov);
    };
    sync();
    for (const el of [smooth, fov]) el.addEventListener('input', () => { sync(); this.requestPlan(); });
    hz.addEventListener('change', () => { sync(); this.requestPlan(true); });

    // format
    for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#seg-codec button'))) {
      b.addEventListener('click', () => {
        this.codec = b.dataset.c as 'hevc' | 'avc';
        for (const o of Array.from(document.querySelectorAll('#seg-codec button'))) o.setAttribute('aria-checked', String(o === b));
        this.renderExportFacts();
        if (this.clip?.plan) this.postEngine({ type: 'probe-encoder', bitrate: this.bitrate(), prefer: this.codec });
      });
    }
    // quality
    for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#seg-quality button'))) {
      b.addEventListener('click', () => {
        this.quality = b.dataset.q as Quality;
        for (const o of Array.from(document.querySelectorAll('#seg-quality button'))) o.setAttribute('aria-checked', String(o === b));
        this.renderExportFacts();
        if (this.clip?.plan) this.postEngine({ type: 'probe-encoder', bitrate: this.bitrate(), prefer: this.codec });
      });
    }

    $('btn-export').addEventListener('click', () => void this.startExport());
    $('btn-cancel').addEventListener('click', () => this.postEngine({ type: 'cancel' }));
    $('btn-again').addEventListener('click', () => { this.showCard('export'); this.updateExportButton(); });
    $('btn-error-ok').addEventListener('click', () => { this.showCard('export'); this.updateExportButton(); });
  }

  private resizeCanvases() {
    const v = $('viewer');
    if (!v.clientWidth) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.min(2560, Math.round(v.clientWidth * dpr));
    const h = Math.round(w / this.aspect);
    this.postEngine({ type: 'resize', w, h });
  }

  private setAspect(a: number) {
    if (!Number.isFinite(a) || a <= 0 || Math.abs(a - this.aspect) < 1e-3) return;
    this.aspect = a;
    $('viewer').style.setProperty('--ar', String(a));
    this.resizeCanvases();
  }

  private status(text: string | null, f?: number) {
    const el = $('viewer-status');
    if (text === null) { el.hidden = true; return; }
    el.hidden = false;
    $('viewer-status-text').textContent = text;
    ($('viewer-status-bar') as HTMLElement).style.width = `${Math.round(Math.max(0, Math.min(1, f ?? 0)) * 100)}%`;
  }

  private setPlanState(text: string, busy: boolean) {
    const el = $('plan-state');
    el.textContent = text;
    el.classList.toggle('is-busy', busy);
  }

  private toast(msg: string) {
    const t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout((t as any)._timer);
    (t as any)._timer = setTimeout(() => (t.hidden = true), 4200);
  }

  private fail(msg: string, details?: ErrorDetails) {
    this.debug.error = msg;
    this.errorDetails = details ?? null;
    this.debug.errorDetails = details;
    this.log('fail: ' + msg);
    if (this.debug.view !== 'work') { this.toast(msg); return; }
    this.ensureErrorExtras();
    $('error-text').textContent = msg;
    const copy = $('btn-error-copy') as HTMLButtonElement;
    copy.hidden = !details;
    copy.textContent = 'Copy details';
    $('error-details-pre').hidden = true;
    this.showCard('error');
  }

  /** "Copy details" + a fallback text box in the error card (built here so the page markup stays unchanged). */
  private ensureErrorExtras() {
    if (document.getElementById('btn-error-copy')) return;
    const card = $('card-error');
    const ok = $('btn-error-ok');
    const row = document.createElement('div');
    row.className = 'error-actions';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.id = 'btn-error-copy';
    copy.className = 'btn btn-ghost';
    copy.textContent = 'Copy details';
    copy.title = 'Copies technical details (browser, graphics card, clip format, the decoder’s error) to send to whoever helps you';
    copy.hidden = true;
    ok.replaceWith(row);
    ok.classList.remove('btn-block');
    row.append(copy, ok);
    const pre = document.createElement('pre');
    pre.id = 'error-details-pre';
    pre.className = 'error-details';
    pre.hidden = true;
    pre.tabIndex = 0;
    card.append(pre);
    copy.addEventListener('click', () => void this.copyDetails(copy));
  }

  private async loadUaData() {
    try {
      const ud = (navigator as any).userAgentData;
      if (!ud) return;
      this.uaData = { brands: ud.brands, mobile: ud.mobile, platform: ud.platform };
      const hi = await ud.getHighEntropyValues?.(['platformVersion', 'architecture', 'bitness', 'model', 'fullVersionList']);
      if (hi) this.uaData = { ...this.uaData, ...hi };
    } catch { /* optional */ }
  }

  /** Everything someone helping remotely needs to know about a failure, as pretty JSON. */
  diagnostics(): string {
    const c = this.clip;
    const nav = navigator as any;
    const v = c?.video;
    const out = {
      app: 'Stillpoint web',
      page: location.origin + location.pathname,
      time: new Date().toISOString(),
      browser: {
        userAgent: navigator.userAgent, uaData: this.uaData ?? undefined, platform: navigator.platform, language: navigator.language,
        cores: navigator.hardwareConcurrency, memoryGB: nav.deviceMemory, screen: `${screen.width}×${screen.height} @${window.devicePixelRatio}x`,
      },
      gpu: this.debug.gpu,
      caps: this.caps,
      clip: c ? {
        name: c.file.name, bytes: c.file.size, codec: v?.codecString, width: v?.width, height: v?.height, frames: c.frames, fps: c.fps,
        duration: c.duration, camera: c.tel?.camera, colr: v?.colr, samples: v?.sampleCount,
      } : undefined,
      decoder: c?.decoder,
      error: { message: this.debug.error, details: this.errorDetails ?? undefined },
      events: this.debug.events.slice(-30),
    };
    return JSON.stringify(out, (_k, val) => (val instanceof Float64Array || val instanceof Uint8Array || val instanceof Uint32Array ? `[${val.length} values]` : val), 2);
  }

  private async copyDetails(btn: HTMLButtonElement) {
    const text = this.diagnostics();
    this.debug.lastDiagnostics = text;
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch { /* fall back below */ }
    if (!ok) {
      const ta = document.createElement('textarea');
      ta.value = text; ta.readOnly = true;
      ta.style.position = 'fixed'; ta.style.opacity = '0'; ta.style.pointerEvents = 'none';
      document.body.append(ta);
      ta.select();
      try { ok = document.execCommand('copy'); } catch { ok = false; }
      ta.remove();
    }
    const pre = $('error-details-pre');
    if (!ok) {
      pre.textContent = text;
      pre.hidden = false;
      const r = document.createRange(); r.selectNodeContents(pre);
      const sel = window.getSelection(); sel?.removeAllRanges(); sel?.addRange(r);
    }
    btn.textContent = ok ? 'Copied — paste it in a message' : 'Select the text below and copy it';
    this.log(`copy details ${ok ? 'ok' : 'fallback'} (${text.length} chars)`);
    setTimeout(() => { btn.textContent = 'Copy details'; }, 3000);
  }

  private showCard(which: 'export' | 'progress' | 'done' | 'error') {
    $('card-export').hidden = which !== 'export';
    $('card-progress').hidden = which !== 'progress';
    $('card-done').hidden = which !== 'done';
    $('card-error').hidden = which !== 'error';
  }

  private renderBadge() {
    const b = $('gyro-badge');
    const c = this.clip;
    let level = 'pending', text = 'Reading gyro…';
    if (c?.telError) { level = 'bad'; text = 'No gyro data found'; }
    else if (c?.tel) {
      const t = c.tel;
      if (t.eisBaked) { level = 'bad'; text = 'In-camera EIS on — not supported'; }
      else if (t.hasHighrate && t.imuRate > 0) { level = 'good'; text = `${fmtRate(t.imuRate)} gyro`; }
      else if (t.imuRate > 0) { level = 'limited'; text = `${fmtRate(t.imuRate)} attitude — limited`; }
      else { level = 'bad'; text = 'No gyro data'; }
    }
    b.dataset.level = level;
    b.querySelector('span')!.textContent = text;
  }

  private codecName(t?: Mp4Track): string {
    const s = t?.codecString ?? '';
    if (s.startsWith('avc1') || s.startsWith('avc3')) {
      const p = parseInt(s.slice(5, 7), 16);
      return `H.264 ${p === 100 ? 'High' : p === 77 ? 'Main' : p === 66 ? 'Baseline' : p === 110 ? 'High 10' : ''}`.trim();
    }
    if (s.startsWith('hvc1') || s.startsWith('hev1')) {
      const prof = s.split('.')[1]?.replace(/^[ABC]/, '');
      return prof === '2' ? 'HEVC Main 10' : prof === '1' ? 'HEVC Main' : 'HEVC';
    }
    return s || t?.fourcc || '—';
  }

  private renderFacts() {
    const c = this.clip; if (!c) return;
    const v = c.video;
    const chips: string[] = [];
    if (v?.width) chips.push(`${v.width}×${v.height}`);
    if (c.fps) chips.push(`${fmtFps(c.fps)} fps`);
    else if (v && v.sampleCount > 1 && v.durationS) chips.push(`${fmtFps((v.sampleCount) / v.durationS)} fps`);
    if (c.duration || c.info) chips.push(fmtTime(c.duration || c.info!.durationS, false));
    if (v) chips.push(this.codecName(v));
    chips.push(fmtBytes(c.file.size));
    $('clip-meta').innerHTML = chips.map(x => `<span class="chip">${esc(x)}</span>`).join('');

    const rows: Array<[string, string]> = [];
    const t = c.tel;
    rows.push(['Camera', t ? t.camera : c.telError ? 'Unknown' : '…']);
    if (v?.width) rows.push(['Resolution', `${v.width}×${v.height} · ${aspectLabel(v.width, v.height!)}`]);
    if (c.fps) rows.push(['Frame rate', `${fmtFps(c.fps)} fps`]);
    if (c.frames) rows.push(['Length', `${fmtDuration(c.duration)} · ${c.frames.toLocaleString()} frames`]);
    if (v) rows.push(['Video', `${this.codecName(v)}${v.colr?.transfer === 18 ? ' · HLG' : v.colr?.transfer === 16 ? ' · PQ' : ''}`]);
    if (c.decoder) rows.push(['Decoding', c.decoder.software ? 'Software · slower' : c.decoder.label.startsWith('hardware') ? 'Graphics card (hardware)' : 'Automatic']);
    else if (c.decodeError) rows.push(['Decoding', 'Not possible on this computer']);
    rows.push(['Audio', c.audio ? `AAC · ${Math.round((c.audio.sampleRate ?? 48000) / 1000)} kHz${c.audio.channels === 2 ? ' stereo' : c.audio.channels === 1 ? ' mono' : ''}` : 'None']);
    if (t) {
      rows.push(['Gyro', t.imuRate > 0 ? `${fmtRate(t.imuRate)}${t.hasHighrate ? '' : ' (per-frame)'}` : 'None']);
      if (t.medianExposureS > 0) rows.push(['Shutter', fmtShutter(t.medianExposureS)]);
      if (t.readoutS > 0) rows.push(['Readout', `${(t.readoutS * 1000).toFixed(1)} ms`]);
      rows.push(['Lens', t.lensModel === 'kb4' ? 'Fisheye (KB4)' : 'Rectilinear']);
    }
    $('facts').innerHTML = rows.map(([k, val]) => `<dt>${esc(k)}</dt><dd title="${esc(val)}">${esc(val)}</dd>`).join('');

    const w: string[] = [];
    if (c.decodeError) w.push(`<li class="is-bad">${esc(c.decodeError)}</li>`);
    for (const n of c.engineNotes ?? []) w.push(`<li>${esc(n)}</li>`);
    if (c.decoder?.note) w.push(`<li class="is-info">${esc(c.decoder.note)}</li>`);
    if (c.telError) w.push(`<li class="is-bad">This file has no gyro data Stillpoint can use. It may be an edited or already-stabilized copy, or not from a DJI O3, O4 Pro or Osmo Action 4. Open the original MP4 from the camera’s card (recorded with EIS off).</li>`);
    if (t?.eisBaked) w.push('<li class="is-bad">This clip was recorded with in-camera stabilization (RockSteady / HorizonSteady) on. Its gyro no longer matches the picture, so it can’t be stabilized. Record with EIS off.</li>');
    else if (t && !t.hasHighrate && t.imuRate > 0) w.push('<li>Only per-frame attitude is stored in this clip (no high-rate gyro), so fast vibration and rolling-shutter jello can’t be fully removed. For the best result record 4:3 with EIS off.</li>');
    // engine diagnostics are informational: they live in the collapsible clip details, not the top of the panel
    const diag = [...(t?.warnings ?? []), ...(c.telError ? [`Telemetry: ${c.telError}`] : [])];
    $('clip-notes').innerHTML = diag.map(x => `<li>${esc(x)}</li>`).join('');
    $('clip-notes').hidden = !diag.length;
    const noStab = !!c.telError || !!t?.eisBaked;
    for (const id of ['in-smooth', 'in-fov', 'in-horizon']) ($(id) as HTMLInputElement).disabled = noStab || this.exporting;
    $('warnings').innerHTML = w.join('');
    $('card-notes').hidden = w.length === 0;
    $('clip-summary').textContent = t ? t.camera : '';
  }

  private bitrate(): number {
    const c = this.clip;
    const w = c?.plan?.outW ?? 3840, h = c?.plan?.outH ?? 2160, fps = c?.fps || 59.94;
    return this.presets(w, h, fps)[this.quality];
  }

  /** H.264 needs ~40 % more bits than HEVC for the same quality */
  private presets(w: number, h: number, fps: number) {
    const p = bitratePresets(w, h, fps);
    if (this.codec !== 'avc') return p;
    const up = (b: number) => Math.round((b * 1.4) / 1e6) * 1e6;
    return { balanced: up(p.balanced), high: up(p.high), max: up(p.max) };
  }

  private rangeSeconds(): number {
    const c = this.clip;
    if (!c?.pts || !c.frames) return c?.duration ?? 0;
    return c.pts[c.outP] - c.pts[c.inP] + 1 / (c.fps || 30);
  }

  private renderExportFacts() {
    const c = this.clip; if (!c) return;
    const presets = this.presets(c.plan?.outW ?? 3840, c.plan?.outH ?? 2160, c.fps || 59.94);
    $('q-balanced').textContent = `${Math.round(presets.balanced / 1e6)} Mb/s`;
    $('q-high').textContent = `${Math.round(presets.high / 1e6)} Mb/s`;
    $('q-max').textContent = `${Math.round(presets.max / 1e6)} Mb/s`;
    const rows: Array<[string, string]> = [];
    if (c.plan) rows.push(['Output', `${c.plan.outW}×${c.plan.outH} · ${fmtFps(c.fps)} fps`]);
    const e = c.encoder;
    const got = e?.codec ? (/^(avc1|avc3)/.test(e.codec) ? 'avc' : /^(hvc1|hev1)/.test(e.codec) ? 'hevc' : 'other') : null;
    const fallback = got && got !== this.codec ? ` · ${this.codec === 'avc' ? 'H.264' : 'HEVC'} unavailable at this size` : '';
    rows.push(['Codec', e ? (e.codec ? `${e.label}${e.hardware ? ' · hardware' : ' · software (slow)'}${fallback}` : 'No encoder available') : '…']);
    const whole = c.inP === 0 && c.outP === c.frames - 1;
    const secs = this.rangeSeconds();
    rows.push(['Range', c.frames ? (whole ? `Whole clip · ${fmtDuration(secs)}` : `${fmtTime(c.pts![c.inP] - c.pts![0])} – ${fmtTime(c.pts![c.outP] - c.pts![0])} · ${fmtDuration(secs)}`) : '…']);
    rows.push(['Audio', c.audio ? 'Original, copied' : 'None']);
    rows.push(['Est. size', c.frames ? `≈ ${fmtBytes((this.bitrate() * secs) / 8)}` : '…']);
    $('export-facts').innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd>`).join('');
    const sink = this.sinkKind();
    $('btn-export-label').textContent = sink === 'fsa' ? 'Stabilize & save…' : 'Stabilize';
    $('export-note').textContent = sink === 'fsa'
      ? 'You’ll pick where to save. The file streams straight to disk, so clip length doesn’t matter.'
      : sink === 'opfs' ? 'The finished MP4 downloads when it’s done. It’s written to this browser’s private storage first.'
        : 'The finished MP4 downloads when it’s done. Long clips need a lot of memory in this browser.';
  }

  private updateExportButton() {
    const c = this.clip;
    const ok = !!c && !this.exporting && !!c.plan && c.appliedPlanId === c.planId && !!c.encoder?.codec && !c.tel?.eisBaked && !c.telError && !c.decodeError && !!c.decoder;
    ($('btn-export') as HTMLButtonElement).disabled = !ok;
  }

  private lockControls(lock: boolean) {
    for (const id of ['in-smooth', 'in-fov', 'in-horizon', 'btn-play']) ($(id) as HTMLInputElement).disabled = lock;
    if (!lock && (this.clip?.decodeError || !this.clip?.frames)) ($('btn-play') as HTMLButtonElement).disabled = true;
    for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#seg-quality button, #seg-codec button'))) b.disabled = lock;
  }

  private renderTimeline() {
    const c = this.clip; if (!c || c.frames < 2) return;
    const f = (p: number) => `${(p / (c.frames - 1)) * 100}%`;
    const range = $('tl-range');
    range.style.left = f(c.inP);
    range.style.right = `${100 - (c.outP / (c.frames - 1)) * 100}%`;
    $('tl-in').style.left = f(c.inP);
    $('tl-out').style.left = f(c.outP);
    this.renderPlayhead();
  }

  private renderPlayhead() {
    const c = this.clip; if (!c || c.frames < 2) return;
    $('tl-head').style.left = `${(c.pres / (c.frames - 1)) * 100}%`;
    const t = c.pts ? c.pts[c.pres] - c.pts[0] : c.pres / (c.fps || 30);
    $('time').innerHTML = `${fmtTime(t)} <span>/ ${fmtTime(c.duration, c.duration < 60)}</span>`;
  }

  // ───────────────────────── scrub / play ─────────────────────────

  private seek(p: number) {
    const c = this.clip; if (!c || !c.frames) return;
    p = Math.max(0, Math.min(c.frames - 1, Math.round(p)));
    c.pres = p;
    this.renderPlayhead();
    if (this.exporting) return;
    if (this.seekBusy) { this.seekPending = p; return; }
    this.seekBusy = true;
    this.postEngine({ type: 'seek', pres: p });
    clearTimeout(this.seekTimer);
    this.seekTimer = window.setTimeout(() => this.seekDone(), this.clip?.decoder?.software ? 6000 : 2500);
  }

  private seekDone() {
    clearTimeout(this.seekTimer);
    this.seekBusy = false;
    if (this.seekPending !== null) { const p = this.seekPending; this.seekPending = null; this.seek(p); }
  }

  private togglePlay() {
    const c = this.clip; if (!c || !c.frames || this.exporting) return;
    if (this.playing) { this.pause(); return; }
    const from = c.pres >= c.outP ? c.inP : c.pres;
    this.postEngine({ type: 'play', from, to: c.outP, loop: false });
  }

  private pause() { if (this.playing) this.postEngine({ type: 'pause' }); }

  // ───────────────────────── export ─────────────────────────

  private sinkKind(): SinkKind {
    if (this.forcedSink === 'fsa-test') return 'fsa';
    if (this.forcedSink === 'fsa' || this.forcedSink === 'opfs' || this.forcedSink === 'memory') return this.forcedSink;
    if (this.caps?.saveToDisk) return 'fsa';
    if (this.caps?.opfs) return 'opfs';
    return 'memory';
  }

  private outName(): string {
    const base = this.clip!.file.name.replace(/\.[^.]+$/, '');
    return `${base}_stillpoint.mp4`;
  }

  private async startExport() {
    const c = this.clip;
    if (!c || this.exporting || !c.plan) return;
    this.pause();
    const kind = this.sinkKind();
    let handle: FileSystemFileHandle | undefined;
    if (this.forcedSink === 'fsa-test') {
      handle = await (await navigator.storage.getDirectory()).getFileHandle(this.outName(), { create: true });
    } else if (kind === 'fsa') {
      try {
        handle = await (window as any).showSaveFilePicker({ suggestedName: this.outName(), types: [{ description: 'MP4 video', accept: { 'video/mp4': ['.mp4'] } }] });
      } catch (e) {
        if ((e as DOMException).name === 'AbortError') return; // user closed the dialog
        this.toast('Couldn’t open the save dialog — the file will download instead.');
      }
    }
    const est = (this.bitrate() * this.rangeSeconds()) / 8;
    const sink = handle ? { kind: 'fsa' as const, name: handle.name, handle } : { kind: (kind === 'fsa' ? (this.caps?.opfs ? 'opfs' : 'memory') : kind) as SinkKind, name: this.outName() };
    if (sink.kind === 'memory' && est > 2e9 && !confirm(`This export will be about ${fmtBytes(est)}, which has to fit in memory in this browser. Continue?`)) return;
    if (sink.kind === 'opfs') {
      // the file is written to browser storage first, then copied to Downloads: check there is room for it
      try {
        const { quota = Infinity, usage = 0 } = await navigator.storage.estimate();
        if (quota - usage < est * 1.1) {
          this.fail(`Not enough free space for this export: it needs about ${fmtBytes(est * 1.1)} of browser storage and ${fmtBytes(quota - usage)} is available. Free up disk space, choose a lower quality, or export a shorter range.`);
          return;
        }
      } catch { /* estimate unavailable: go ahead */ }
    }
    this.exporting = true;
    this.debug.exportState = 'running';
    this.debug.result = undefined; this.debug.error = undefined;
    this.log(`export start sink=${sink.kind}`);
    if (this.downloadUrl) { URL.revokeObjectURL(this.downloadUrl); this.downloadUrl = null; }
    this.lockControls(true);
    this.updateExportButton();
    this.showCard('progress');
    try { this.wakeLock = await (navigator as any).wakeLock?.request('screen') ?? null; } catch { this.wakeLock = null; }
    this.renderProgress({ phase: 'starting', done: 0, total: c.outP - c.inP + 1, fps: 0, etaS: NaN, elapsedS: 0, bytes: 0 });
    $('viewer').classList.remove('no-split');
    this.postEngine({ type: 'export', settings: { bitrate: this.bitrate(), prefer: this.codec, first: c.inP, last: c.outP, includeAudio: true, sink } });
  }

  private renderProgress(p: RenderProgress) {
    const f = p.total ? p.done / p.total : 0;
    $('prog-label').textContent = p.phase === 'finalizing' ? 'Finishing the file' : p.phase === 'starting' ? 'Starting' : 'Stabilizing';
    $('prog-pct').textContent = `${Math.floor(f * 100)}%`;
    ($('prog-bar') as HTMLElement).style.width = `${(f * 100).toFixed(2)}%`;
    $('prog-fps').textContent = p.fps > 0 ? `${p.fps.toFixed(p.fps < 10 ? 1 : 0)} fps` : '— fps';
    $('prog-eta').textContent = p.phase === 'finalizing' ? 'writing…' : fmtEta(p.etaS);
    $('prog-bytes').textContent = fmtBytes(p.bytes);
    document.title = p.phase === 'done' ? 'Stillpoint' : `${Math.floor(f * 100)}% · Stillpoint`;
  }

  private releaseWakeLock() { void this.wakeLock?.release().catch(() => {}); this.wakeLock = null; }

  private async exportDone(r: RenderResult) {
    this.exporting = false;
    this.releaseWakeLock();
    this.debug.exportState = 'done';
    this.debug.result = { ...r, blob: null };
    document.title = 'Stillpoint — stabilize DJI footage in your browser';
    this.lockControls(false);
    this.updateExportButton();
    $('done-title').textContent = r.name;
    $('done-sub').textContent = `${fmtDuration(r.frames / (this.clip?.fps || 30))} of video in ${fmtDuration(r.seconds)} · ${r.fps.toFixed(0)} fps · ${fmtBytes(r.bytes)}`;
    const dl = $('btn-download') as HTMLAnchorElement;
    dl.hidden = true;
    try {
      let blob: Blob | null = null;
      if (r.sinkKind === 'opfs') blob = await getOpfsExport(r.name);
      else if (this.forcedSink === 'fsa-test') blob = await (await (await navigator.storage.getDirectory()).getFileHandle(r.name)).getFile();
      else if (r.sinkKind === 'memory') blob = r.blob;
      if (blob) {
        this.downloadUrl = URL.createObjectURL(blob);
        this.debug.downloadUrl = this.downloadUrl;
        dl.href = this.downloadUrl;
        dl.download = r.name;
        dl.hidden = false;
        dl.textContent = `Download · ${fmtBytes(blob.size)}`;
        if (new URLSearchParams(location.search).get('autodownload') !== '0') dl.click();
      } else {
        $('done-title').textContent = `Saved ${r.name}`;
      }
    } catch (e) {
      this.fail('The export finished but the file could not be opened for download: ' + (e as Error).message);
      return;
    }
    this.showCard('done');
  }
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]!);
}
