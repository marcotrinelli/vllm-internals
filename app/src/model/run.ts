import { planBlocks, type BlockPlan } from './kv';
import { clamp, interpAt } from './metrics';
import { runSummary } from './trace';
import type { Metrics, MetricsRow, Request, Summary, Trace } from './types';

/* Everything the panels draw, on one clock */
export interface Run {
  metrics: Metrics | null;
  trace: Trace | null;
  requests: Request[];
  byId: Map<number, Request>;
  offset: number;
  tStart: number;
  tEnd: number;
  blockSize: number;
  nBlocks: number;
  apc: boolean;
  plan: BlockPlan | null;
  summary: Summary;
}

/* Each file carries the wall-clock instant of its own t=0, so the trace is shifted onto the
 * metrics clock by the difference of the two */
export function combine(metrics: Metrics | null, trace: Trace | null): Run {
  const offset = metrics?.t0Unix != null && trace?.t0Unix != null ? trace.t0Unix - metrics.t0Unix : 0;
  const requests = (trace?.requests ?? []).map((r) => ({
    ...r,
    tSubmit: r.tSubmit + offset,
    tFirst: r.tFirst != null ? r.tFirst + offset : null,
    tEnd: r.tEnd + offset,
    tokenT: r.tokenT.map((x) => x + offset),
    preempted: [] as Array<[number, number]>,
  })).sort((a, b) => a.tSubmit - b.tSubmit || a.id - b.id);

  if (metrics) {
    assignRunStarts(metrics.rows, requests);
    assignPreemptions(metrics.rows, requests);
  } else {
    for (const r of requests) { r.tRunStart = r.tSubmit; } // queue and prefill are indistinguishable
  }
  const blockSize = metrics?.blockSize ?? 16;
  const nBlocks = metrics?.nBlocks ?? 0;
  // With prefix caching unreported, a cache hit in the trace is proof enough that it is on
  const apc = metrics?.apc ?? requests.some((r) => r.cachedTokens > 0);
  return {
    metrics, trace, requests, offset,
    byId: new Map(requests.map((r) => [r.id, r])),
    tStart: Math.min(0, ...requests.map((r) => r.tSubmit)),
    tEnd: Math.max(metrics?.tEnd ?? 0, ...requests.map((r) => r.tEnd), 1e-3),
    blockSize, nBlocks, apc,
    plan: nBlocks && requests.length ? planBlocks(requests, blockSize, nBlocks) : null,
    summary: runSummary(requests),
  };
}

/* A request that has arrived but not streamed a token yet is either queued or being
 * prefilled, and the client cannot tell which. The server can: `num_requests_running` minus
 * the requests already decoding is the number of prefill slots in use, and
 * `num_requests_waiting` how many are queued. Hand the slots out FCFS, once, so a request's
 * entry into the batch is fixed for the whole replay. With other clients on the same server
 * this over-assigns slots: it is a reconstruction */
export function assignRunStarts(rows: MetricsRow[], requests: Request[]): void {
  const granted = new Map<number, number>();
  // the last scrape that saw a request still waiting: queued at least until then
  const waited = new Map<number, number>();
  for (const row of rows) {
    const t = row.t;
    const pending: Request[] = [];
    let decoding = 0;
    for (const r of requests) {
      if (t < r.tSubmit || t >= r.tEnd) { continue; }
      if (r.tFirst != null && t >= r.tFirst) { decoding++; } else { pending.push(r); }
    }
    pending.sort((a, b) => (granted.has(a.id) ? 0 : 1) - (granted.has(b.id) ? 0 : 1)
      || a.tSubmit - b.tSubmit || a.id - b.id);
    const slots = Math.min(pending.length, Math.max(0, Math.round(row.running) - decoding));
    for (const r of pending.slice(0, slots)) {
      if (!granted.has(r.id)) { granted.set(r.id, t); }
    }
    // The rest are queued only as far as the waiting gauge confirms, latest arrivals first: a
    // request between the HTTP layer and the scheduler is in neither gauge
    const queued = Math.min(pending.length - slots, Math.max(0, Math.round(row.waiting)));
    for (const r of pending.slice(pending.length - queued)) {
      if (!granted.has(r.id)) { waited.set(r.id, t); }
    }
  }
  // Queued from submit until the last scrape that saw it waiting, prefill from there on. No
  // such scrape (sparse scrapes, or a short wait between two) is no evidence of a queue, and
  // the whole wait for the first token counts as prefill. A lower bound on the queue
  for (const r of requests) {
    r.tRunStart = clamp(waited.get(r.id) ?? r.tSubmit, r.tSubmit, r.tFirst ?? r.tEnd);
  }
}

export type RequestState = 'absent' | 'queued' | 'prefill' | 'decode' | 'preempted' | 'finished';

/* The trace has no preemption event and /metrics only counts them, so a decode gap is read as
 * this request being preempted when it overlaps a scrape window that counted one and is far
 * longer than the request's usual gap (a preempted sequence waits, then recomputes its KV) */
export function assignPreemptions(rows: MetricsRow[], requests: Request[]): void {
  const windows: Array<[number, number]> = [];
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].dPreempt > 0) { windows.push([rows[i - 1].t, rows[i].t]); }
  }
  if (!windows.length) { return; }
  for (const r of requests) {
    const gaps = r.tokenT.slice(1).map((x, j) => x - r.tokenT[j]).sort((a, b) => a - b);
    if (!gaps.length) { continue; }
    const stall = Math.max(5 * gaps[Math.floor(gaps.length / 2)], 0.05);
    for (let j = 1; j < r.tokenT.length; j++) {
      const a = r.tokenT[j - 1];
      const b = r.tokenT[j];
      if (b - a > stall && windows.some(([w0, w1]) => w0 < b && w1 > a)) { r.preempted.push([a, b]); }
    }
  }
}

export function stateAt(r: Request, t: number): RequestState {
  if (t < r.tSubmit) { return 'absent'; }
  if (t >= r.tEnd) { return 'finished'; }
  if (t < r.tRunStart) { return 'queued'; }
  if (r.preempted.some(([a, b]) => t > a && t < b)) { return 'preempted'; }
  return r.tFirst == null || t < r.tFirst ? 'prefill' : 'decode';
}

export type AxisMode = 'time' | 'step';

export interface Axis {
  // instant -> fraction of the chart width, and back; outside [0, 1] when the instant is
  // outside the shown range, so each chart clips at its own edge
  x: (t: number) => number;
  tAt: (f: number) => number;
  step: boolean;
  // the instants at the two edges
  t0: number;
  t1: number;
}

/* The x axis every chart shares: wall clock, or the engine's own step counter (where a
 * stall, many seconds and few steps, collapses to nothing). `range` zooms to a span of the
 * run, picked on the engine timeline */
export function makeAxis(run: Run, mode: AxisMode, range: [number, number] | null = null): Axis {
  const rec = run.metrics;
  const [t0, t1] = range ?? [run.tStart, run.tEnd];
  if (mode === 'step' && rec && rec.hasSteps && rec.rows[rec.rows.length - 1].cSteps > 0) {
    const R = rec.rows;
    const s0 = interpAt(rec, 'cSteps', t0);
    const span = Math.max(1e-9, interpAt(rec, 'cSteps', t1) - s0);
    return {
      step: true, t0, t1,
      x: (t) => (interpAt(rec, 'cSteps', t) - s0) / span,
      tAt: (f) => {
        const v = s0 + clamp(f, 0, 1) * span;
        for (let i = 1; i < R.length; i++) {
          if (v <= R[i].cSteps) {
            const g = clamp((v - R[i - 1].cSteps) / Math.max(1e-9, R[i].cSteps - R[i - 1].cSteps), 0, 1);
            return R[i - 1].t + g * (R[i].t - R[i - 1].t);
          }
        }
        return R[R.length - 1].t;
      },
    };
  }
  const span = Math.max(1e-9, t1 - t0);
  return { step: false, t0, t1, x: (t) => (t - t0) / span, tAt: (f) => t0 + clamp(f, 0, 1) * span };
}
