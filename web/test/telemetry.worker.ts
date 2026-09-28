// TELEMETRY browser harness: openMp4 + loadTelemetry inside a module Worker (the app's analysis path).
import { openMp4 } from '../src/mp4';
import { loadTelemetry } from '../src/telemetry';

self.onmessage = async (e: MessageEvent<{ file: File; imuIdx: number[]; frameIdx: number[] }>) => {
  const { file, imuIdx, frameIdx } = e.data;
  try {
    const t0 = performance.now();
    const info = await openMp4(file);
    const tel = await loadTelemetry(file, info);
    const ms = performance.now() - t0;
    const q: number[] = [];
    for (const k of imuIdx) for (let c = 0; c < 4; c++) q.push(tel.imuQ[4 * k + c]);
    (self as unknown as Worker).postMessage({ ms, frames: tel.framePts.length, imu: tel.imuT.length,
      frameT: frameIdx.map((k) => tel.frameT[k]), imuT: imuIdx.map((k) => tel.imuT[k]), imuQ: q });
  } catch (err) {
    (self as unknown as Worker).postMessage({ error: String(err) });
  }
};
