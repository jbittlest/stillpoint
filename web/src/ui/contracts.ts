/** The engine-module shapes the APP code relies on (the shared interface spec). */
import type { StabParams } from '../plan';

export type { StabParams };

export interface WarperLike {
  warp(frame: VideoFrame, k: number): VideoFrame;
  destroy(): void;
}
