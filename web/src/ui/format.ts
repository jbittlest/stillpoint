export function fmtTime(s: number, tenths = true): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  // round first, so 59.97 s reads 1:00.0, not 0:60.0
  s = tenths ? Math.round(s * 10) / 10 : Math.floor(s + 1e-6);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  const sec = tenths ? r.toFixed(1).padStart(4, '0') : String(Math.floor(r)).padStart(2, '0');
  if (m >= 60) return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}:${sec}`;
  return `${m}:${sec}`;
}

export function fmtDuration(s: number): string {
  if (!Number.isFinite(s)) return '—';
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60), r = Math.round(s - m * 60);
  return `${m} min ${String(r).padStart(2, '0')} s`;
}

export function fmtEta(s: number): string {
  if (!Number.isFinite(s) || s < 0) return '—';
  if (s < 1) return 'almost done';
  if (s < 60) return `${Math.ceil(s)} s left`;
  const m = Math.floor(s / 60), r = Math.round(s - m * 60);
  if (m < 60) return `${m}:${String(r).padStart(2, '0')} left`;
  return `${Math.floor(m / 60)} h ${m % 60} min left`;
}

export function fmtBytes(b: number): string {
  if (!Number.isFinite(b)) return '—';
  if (b < 1e6) return `${Math.round(b / 1e3)} KB`;
  if (b < 1e9) return `${(b / 1e6).toFixed(b < 1e8 ? 1 : 0)} MB`;
  return `${(b / 1e9).toFixed(2)} GB`;
}

export function fmtFps(f: number): string {
  if (!Number.isFinite(f)) return '—';
  return f >= 100 ? f.toFixed(0) : f.toFixed(2).replace(/\.?0+$/, '') || '0';
}

export function fmtRate(hz: number): string {
  if (hz >= 1000) return `${(hz / 1000).toFixed(hz % 1000 === 0 ? 0 : 1)} kHz`;
  return `${Math.round(hz)} Hz`;
}

export function fmtShutter(s: number): string {
  if (!(s > 0)) return '—';
  return `1/${Math.round(1 / s)} s`;
}

export function aspectLabel(w: number, h: number): string {
  const r = w / h;
  if (Math.abs(r - 16 / 9) < 0.02) return '16:9';
  if (Math.abs(r - 4 / 3) < 0.02) return '4:3';
  if (Math.abs(r - 1) < 0.02) return '1:1';
  if (Math.abs(r - 9 / 16) < 0.02) return '9:16';
  return r.toFixed(2) + ':1';
}
