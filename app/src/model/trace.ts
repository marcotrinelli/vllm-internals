import type { Request, Summary, Trace, TraceFile } from './types';

/* Linear interpolation between closest ranks, the definition vllm_internals.trace.quantile uses */
export function quantile(values: number[], q: number): number | null {
  if (!values.length) { return null; }
  const xs = [...values].sort((a, b) => a - b);
  const pos = (xs.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo);
}

export function parseTrace(file: TraceFile, name: string): Trace {
  const requests: Request[] = file.requests.map((r) => {
    const promptIds = r.prompt_ids ?? null;
    const promptTokens = r.prompt_tokens ?? promptIds?.length ?? 0;
    return {
      id: r.idx,
      ok: r.ok,
      error: r.error ?? null,
      tSubmit: r.t_submit,
      tFirst: r.t_first,
      tEnd: r.t_end ?? r.t_submit,
      tokenT: r.token_t,
      tokenText: r.token_text,
      outputIds: r.output_ids ?? null,
      logprobs: r.token_logprob ?? null,
      promptIds,
      promptTokens,
      cachedTokens: Math.min(r.cached_tokens ?? 0, promptTokens),
      completionTokens: r.completion_tokens ?? null,
      finish: r.finish_reason ?? null,
      tRunStart: r.t_submit,
      preempted: [],
    };
  });
  return {
    name,
    model: file.model ?? null,
    run: file.run ?? {},
    t0Unix: file.t0_unix ?? null,
    requests,
    vocab: file.vocab ?? {},
  };
}

const gaps = (r: Request) => r.tokenT.slice(1).map((v, k) => v - r.tokenT[k]);

/* Same definitions as vllm_internals.trace.summary, so the notebook and the viewer agree
 * (checked against examples/batch.summary.json on both sides) */
export function runSummary(requests: Request[]): Summary {
  const ok = requests.filter((r) => r.ok);
  const ttft = ok.map((r) => (r.tFirst ?? r.tEnd) - r.tSubmit);
  const e2e = ok.map((r) => r.tEnd - r.tSubmit);
  const itl = ok.flatMap(gaps);
  const tpot = ok.filter((r) => r.tokenT.length > 1)
    .map((r) => (r.tokenT[r.tokenT.length - 1] - r.tokenT[0]) / (r.tokenT.length - 1));
  // `||`, not `??`: Python's `completion_tokens or len(token_t)` also falls back on 0
  const out = ok.reduce((a, r) => a + (r.completionTokens || r.tokenT.length), 0);
  const wall = ok.length
    ? Math.max(...ok.map((r) => r.tEnd)) - Math.min(...ok.map((r) => r.tSubmit)) : 0;
  const s: Summary = {
    n: requests.length,
    errors: requests.length - ok.length,
    wall_s: wall,
    out_tokens: out,
    out_tok_per_s: wall ? out / wall : null,
    req_per_s: wall ? ok.length / wall : null,
  };
  for (const p of [50, 95, 99]) { s[`ttft_p${p}`] = quantile(ttft, p / 100); }
  for (const p of [50, 95, 99]) { s[`itl_p${p}`] = quantile(itl, p / 100); }
  for (const p of [50, 95]) { s[`tpot_p${p}`] = quantile(tpot, p / 100); }
  for (const p of [50, 95]) { s[`e2e_p${p}`] = quantile(e2e, p / 100); }
  return s;
}

export interface RequestStats {
  queue: number;
  ttft: number | null;
  e2e: number;
  tpot: number | null;
  itl: number[];
  itlP50: number | null;
  itlMax: number | null;
}

export function requestStats(r: Request): RequestStats {
  const n = r.tokenT.length;
  const itl = gaps(r);
  return {
    queue: r.tRunStart - r.tSubmit,
    ttft: r.tFirst != null ? r.tFirst - r.tSubmit : null,
    e2e: r.tEnd - r.tSubmit,
    tpot: n > 1 ? (r.tokenT[n - 1] - r.tokenT[0]) / (n - 1) : null,
    itl,
    itlP50: quantile(itl, 0.5),
    itlMax: itl.length ? Math.max(...itl) : null,
  };
}
