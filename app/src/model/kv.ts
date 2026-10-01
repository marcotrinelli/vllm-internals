import { countLE } from './metrics';
import type { Request } from './types';

export interface BlockTable {
  blockIds: Int32Array; // physical block of each logical block, -1 if the pool ran dry
  blockT: Float64Array; // when each logical block was materialised
  nCached: number; // leading blocks served from the prefix cache
  privateIds: number[];
}

export interface BlockPlan {
  tables: Map<number, BlockTable>;
  maxHead: number; // fresh blocks ever handed out, so the used range is [0, maxHead)
  spilled: number;
}

/* Map each request's logical blocks onto physical ones. Sizes and timings are measured; the
 * PLACEMENT is a reconstruction, because no endpoint exports a block table. Whole blocks the
 * server reported as cache hits map to one shared set, which is what makes them show up with
 * ref > 1. Free list is fresh-first then FIFO, like the engine's LRU free queue */
export function planBlocks(requests: Request[], B: number, nBlocks: number): BlockPlan {
  let head = 0;
  const freed: number[] = [];
  const pop = () => (head < nBlocks ? head++ : (freed.shift() ?? -1));
  const shared = new Map<number, number>();
  const tables = new Map<number, BlockTable>();
  const events: Array<[number, 0 | 1, Request, number]> = [];

  for (const r of requests) {
    const nb = Math.max(0, Math.ceil((r.promptTokens + r.tokenT.length) / B));
    const tb: BlockTable = {
      blockIds: new Int32Array(nb).fill(-1), blockT: new Float64Array(nb),
      nCached: Math.floor(r.cachedTokens / B), privateIds: [],
    };
    tables.set(r.id, tb);
    const tFirst = r.tFirst ?? r.tEnd;
    for (let k = 0; k < nb; k++) {
      const start = k * B;
      let t: number;
      if (start < r.promptTokens) {
        // Cache hits are mapped in at admission; the rest arrive with the prefill chunks,
        // spread over the request's measured prefill time
        t = k < tb.nCached ? r.tRunStart
          : r.tRunStart + (start / r.promptTokens) * Math.max(0, tFirst - r.tRunStart);
      } else {
        const j = start - r.promptTokens;
        t = r.tokenT.length ? r.tokenT[Math.min(j, r.tokenT.length - 1)] : r.tEnd;
      }
      tb.blockT[k] = t;
      events.push([t, 1, r, k]);
    }
    events.push([r.tEnd, 0, r, -1]);
  }
  // Releases sort before allocations at the same instant, so a finishing request's blocks
  // are available to whatever is admitted in its place
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let spilled = 0;
  for (const [, type, r, k] of events) {
    const tb = tables.get(r.id)!;
    if (type === 0) { freed.push(...tb.privateIds); continue; }
    let id: number;
    if (k < tb.nCached) {
      if (!shared.has(k)) {
        const fresh = pop();
        if (fresh < 0) { spilled++; continue; }
        shared.set(k, fresh);
      }
      id = shared.get(k)!;
    } else {
      id = pop();
      if (id < 0) { spilled++; continue; }
      tb.privateIds.push(id);
    }
    tb.blockIds[k] = id;
  }
  return { tables, maxHead: head, spilled };
}

/* 0 free, 1 cached but free, 2 allocated, 3 shared (ref > 1) */
export type BlockState = 0 | 1 | 2 | 3;

export interface Frame {
  states: Uint8Array;
  ref: Int32Array;
  owner: Int32Array; // request id, -2 for a cached block nobody references, -1 for none
  src: Int32Array; // request whose tokens the block holds, live or cached; -1 for none
  logical: Int32Array;
  allocT: Float64Array;
  used: number;
  cachedFree: number;
}

/* Which physical block belongs to whom at one instant */
export function kvFrame(requests: Request[], plan: BlockPlan, nBlocks: number, apc: boolean, t: number): Frame {
  const f: Frame = {
    states: new Uint8Array(nBlocks), ref: new Int32Array(nBlocks), owner: new Int32Array(nBlocks).fill(-1),
    src: new Int32Array(nBlocks).fill(-1), logical: new Int32Array(nBlocks).fill(-1), allocT: new Float64Array(nBlocks), used: 0, cachedFree: 0,
  };
  for (const r of requests) {
    if (t < r.tRunStart || t >= r.tEnd) { continue; }
    const tb = plan.tables.get(r.id)!;
    const held = countLE(tb.blockT, t);
    for (let k = 0; k < held; k++) {
      const id = tb.blockIds[k];
      if (id < 0) { continue; }
      if (f.ref[id] === 0) { f.used++; }
      f.ref[id]++;
      if (f.owner[id] < 0) { f.owner[id] = r.id; f.src[id] = r.id; f.logical[id] = k; f.allocT[id] = tb.blockT[k]; }
    }
  }
  for (let b = 0; b < nBlocks; b++) {
    if (f.ref[b] > 1) { f.states[b] = 3; } else if (f.ref[b] === 1) { f.states[b] = 2; }
  }
  // Shared-prefix blocks stay resident with ref 0 once their users finish: cached and free,
  // the state prefix caching exists for
  if (apc) {
    for (const r of requests) {
      if (t < r.tRunStart) { continue; }
      const tb = plan.tables.get(r.id)!;
      for (let k = 0; k < tb.nCached; k++) {
        const id = tb.blockIds[k];
        if (id >= 0 && f.states[id] === 0 && tb.blockT[k] <= t) {
          f.states[id] = 1;
          f.cachedFree++;
          f.owner[id] = -2; f.src[id] = r.id; f.logical[id] = k; f.allocT[id] = tb.blockT[k];
        }
      }
    }
  }
  return f;
}

/* The token slots of logical block k of r at t: positions [start, end), holding prefix cache
 * hits below `cached`, the rest of the prompt below `prompt` and generated tokens below
 * `filled`. Slots from `filled` on are allocated but empty: what is left of the block until
 * the sequence grows into it. A cached block with no live reader is a whole prompt block */
export interface BlockFill {
  start: number;
  end: number;
  cached: number;
  prompt: number;
  filled: number;
}

export function blockFill(r: Request, k: number, B: number, t: number, resident: boolean): BlockFill {
  const start = k * B;
  const end = start + B;
  const filled = resident ? end : Math.min(end, r.promptTokens + countLE(r.tokenT, t));
  return { start, end, cached: Math.min(end, r.cachedTokens), prompt: Math.min(end, r.promptTokens), filled: Math.max(start, filled) };
}
