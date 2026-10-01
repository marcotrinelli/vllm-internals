import type { Metrics, MetricsFile, MetricsRow } from './types';

export const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);

/* Number of elements <= v in a sorted array */
export function countLE(arr: ArrayLike<number>, v: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= v) { lo = mid + 1; } else { hi = mid; }
  }
  return lo;
}

const pick = (m: Record<string, number>, ...names: string[]): number => {
  for (const n of names) {
    if (m[n] != null) { return +m[n]; }
  }
  return 0;
};

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[s.length >> 1] : 0;
};

/* Round trip of each scrape. A sample without `rtt` is stamped when its response arrived, up
 * to a whole round trip after the server read the counters. The first scrape starts at t=0,
 * so that sample's `t` is a round trip; capped by the usual gap between samples in case the
 * first scrape failed and the first sample is a later one */
function roundTrips(samples: MetricsFile['samples']): { rtt: number[]; legacy: boolean } {
  if (samples.every((s) => s.rtt != null)) { return { rtt: samples.map((s) => s.rtt as number), legacy: false }; }
  const gaps = samples.slice(1).map((s, i) => s.t - samples[i].t);
  const rtt = Math.max(0, Math.min(samples[0].t, gaps.length ? median(gaps) : samples[0].t));
  return { rtt: samples.map(() => rtt), legacy: true };
}

export function parseMetrics(file: MetricsFile, name: string): Metrics {
  const samples = [...file.samples].sort((a, b) => a.t - b.t);
  const trips = roundTrips(samples);
  // a sample without `rtt` is moved back to the middle of its scrape, where one with it
  // already is
  const late = trips.legacy ? trips.rtt[0] / 2 : 0;
  const shift = samples[0].t - late;
  const raw = samples.map((s) => {
    const m = s.m;
    return {
      t: s.t - late - shift,
      m,
      h: s.h ?? {},
      l: s.l ?? {},
      e: s.e ?? {},
      running: pick(m, 'num_requests_running'),
      waiting: pick(m, 'num_requests_waiting'),
      kv: pick(m, 'kv_cache_usage_perc', 'gpu_cache_usage_perc'),
      steps: pick(m, 'iteration_tokens_total_count'),
      iterTok: pick(m, 'iteration_tokens_total_sum'),
      gen: pick(m, 'generation_tokens_total'),
      prompt: pick(m, 'prompt_tokens_total'),
      cached: pick(m, 'prompt_tokens_cached_total'),
      queries: pick(m, 'prefix_cache_queries_total', 'gpu_prefix_cache_queries_total'),
      hits: pick(m, 'prefix_cache_hits_total', 'gpu_prefix_cache_hits_total'),
      preempt: pick(m, 'num_preemptions_total'),
      done: pick(m, 'request_success_total'),
    };
  });

  const base = raw[0];
  const rows: MetricsRow[] = raw.map((r, i) => {
    const p = i ? raw[i - 1] : r;
    const dt = i ? Math.max(1e-9, r.t - p.t) : 0;
    const d = (k: 'steps' | 'iterTok' | 'gen' | 'preempt' | 'done') => (i ? Math.max(0, r[k] - p[k]) : 0);
    const dDecode = d('gen');
    // The iteration histogram counts every token that went through a forward pass; whatever
    // is not a generated token was prefill. The only place /metrics separates the two
    const dPrefill = Math.max(0, d('iterTok') - dDecode);
    const dSteps = d('steps');
    const cQueries = r.queries - base.queries;
    const cPrompt = r.prompt - base.prompt;
    return {
      t: r.t, m: r.m, h: r.h, l: r.l, e: r.e, running: r.running, waiting: r.waiting, kv: r.kv,
      dSteps, dDecode, dPrefill, dPreempt: d('preempt'), dDone: d('done'),
      msPerStep: dSteps > 0 ? (dt * 1e3) / dSteps : null,
      tokPerStep: dSteps > 0 ? (dPrefill + dDecode) / dSteps : null,
      decodePerS: dt ? dDecode / dt : 0,
      prefillPerS: dt ? dPrefill / dt : 0,
      // Counters relative to the first scrape: a recording must not depend on how long the
      // server had been up before it started
      cSteps: r.steps - base.steps,
      cPreempt: r.preempt - base.preempt,
      cDone: r.done - base.done,
      hitRateRun: cQueries > 0 ? (r.hits - base.hits) / cQueries : null,
      cachedFrac: cPrompt > 0 ? (r.cached - base.cached) / cPrompt : null,
    };
  });

  const names = new Set<string>();
  for (const r of rows) { for (const k in r.m) { names.add(k); } }
  const cache = file.info?.cache_config_info ?? {};
  const blockSize = +cache.block_size || 16;
  const kvTokens = +cache.kv_cache_size_tokens || 0;
  const peak = (k: 'kv' | 'running' | 'waiting') => rows.reduce((a, r) => Math.max(a, r[k]), 0);
  const engines = new Set<string>();
  for (const r of rows) { for (const k in r.e) { engines.add(k); } }
  const edges: Record<string, number[]> = {};
  for (const [b, v] of Object.entries(file.buckets ?? {})) { edges[b] = v.map((x) => (x == null ? Infinity : x)); }
  return {
    name,
    source: file.source ?? null,
    info: file.info ?? {},
    server: file.server ?? null,
    engines: [...engines].sort((a, b) => +a - +b || a.localeCompare(b)),
    edges,
    rows,
    times: rows.map((r) => r.t),
    metricNames: [...names].sort(),
    // note that t0_unix is the recorder's t=0, and the first sample may sit a little after it
    t0Unix: file.t0_unix != null ? file.t0_unix + shift : null,
    tEnd: rows[rows.length - 1].t,
    rtt: median(trips.rtt),
    hasSteps: names.has('iteration_tokens_total_count'),
    hasKv: names.has('kv_cache_usage_perc') || names.has('gpu_cache_usage_perc'),
    blockSize,
    nBlocks: +cache.num_gpu_blocks || (kvTokens ? Math.round(kvTokens / blockSize) : 0),
    apc: cache.enable_prefix_caching == null ? null : cache.enable_prefix_caching.toLowerCase() === 'true',
    kvPeak: peak('kv'),
    runningPeak: peak('running'),
    waitingPeak: peak('waiting'),
  };
}

export const idxAt = (rec: Metrics, t: number) => clamp(countLE(rec.times, t) - 1, 0, rec.rows.length - 1);

type NumericKey = { [K in keyof MetricsRow]: MetricsRow[K] extends number ? K : never }[keyof MetricsRow];

/* Linear interpolation of a row field, so the playhead moves smoothly between scrapes */
export function interpAt(rec: Metrics, key: NumericKey, t: number): number {
  const R = rec.rows;
  const i = idxAt(rec, t);
  if (i >= R.length - 1 || t <= R[0].t) { return R[i][key]; }
  const a = R[i];
  const b = R[i + 1];
  const f = clamp((t - a.t) / Math.max(1e-9, b.t - a.t), 0, 1);
  return a[key] + (b[key] - a[key]) * f;
}

/* Rates and quantiles in the panels cover this much history before the playhead: a twentieth
 * of the recording, so a short run still gets a few scrapes and a long one stays smooth */
export const rateWindow = (rec: Metrics) => clamp(rec.tEnd / 20, 0.5, 15);

/* Samples [i0, i1] covering the `s` seconds up to scrape i1, always at least one scrape
 * window wide. i1 is the scrape that closes the window the playhead is in, the column the
 * engine timeline draws under it: the last sample at or before t would leave the window empty
 * (every rate null) until a second scrape lands in it, and lag the chart by a whole window.
 * The `s` seconds count back from i1, not from t, so the window only changes where the column
 * does: counted back from t, a scrape gap wider than `s` would pull in the column before and
 * report what the chart shows to the left of the playhead */
export function windowAt(rec: Metrics, t: number, s: number): [number, number] {
  const last = rec.rows.length - 1;
  const k = countLE(rec.times, t);
  const i1 = clamp(k > 0 && rec.times[k - 1] === t ? k - 1 : k, Math.min(1, last), last);
  return [Math.max(0, Math.min(idxAt(rec, rec.times[i1] - s), i1 - 1)), i1];
}

/* Per-second growth of a counter between two samples, pool-wide or for one engine core. Null
 * when the window holds a single sample */
export function rate(rec: Metrics, metric: string, i0: number, i1: number, engine?: string): number | null {
  const a = rec.rows[i0];
  const b = rec.rows[i1];
  const dt = b.t - a.t;
  if (dt <= 0) { return null; }
  const at = (r: MetricsRow) => (engine == null ? r.m[metric] : r.e[engine]?.[metric]) ?? 0;
  return Math.max(0, at(b) - at(a)) / dt;
}

export interface HistWindow {
  edges: number[];
  cum: number[];
  n: number;
  mean: number | null;
}

/* Prometheus buckets are cumulative, so a window is a difference of two snapshots; i0 < 0
 * reads everything up to i1 */
export function histWindow(rec: Metrics, base: string, i0: number, i1: number): HistWindow | null {
  const edges = rec.edges[base];
  const b = rec.rows[i1]?.h[base];
  if (!edges || !b) { return null; }
  const a = i0 >= 0 && i0 !== i1 ? rec.rows[i0].h[base] : undefined;
  const cum = b.map((v, k) => Math.max(0, v - (a?.[k] ?? 0)));
  const n = cum[cum.length - 1] ?? 0;
  const sumOf = (i: number) => rec.rows[i]?.m[`${base}_sum`] ?? 0;
  const sum = Math.max(0, sumOf(i1) - (a ? sumOf(i0) : 0));
  return { edges, cum, n, mean: n ? sum / n : null };
}

export function histQuantile(w: HistWindow | null, q: number): number | null {
  if (!w || !w.n) { return null; }
  const target = q * w.n;
  for (let i = 0; i < w.cum.length; i++) {
    if (w.cum[i] < target) { continue; }
    const lo = i ? w.edges[i - 1] : 0;
    const hi = w.edges[i];
    // in the overflow bucket: a lower bound only
    if (!Number.isFinite(hi)) { return lo; }
    const c0 = i ? w.cum[i - 1] : 0;
    return lo + ((target - c0) / Math.max(1e-9, w.cum[i] - c0)) * (hi - lo);
  }
  return w.edges[w.edges.length - 1];
}
