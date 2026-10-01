import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { blockFill, kvFrame } from './kv';
import { histQuantile, histWindow, interpAt, parseMetrics, rate, rateWindow, windowAt } from './metrics';
import { assignPreemptions, assignRunStarts, combine, makeAxis, stateAt } from './run';
import { tokenAt, tokenRange } from './tokens';
import { parseTrace, runSummary } from './trace';
import type { MetricsFile, MetricsRow, TraceFile } from './types';
import { validate } from './validate';

const EXAMPLES = resolve(__dirname, '../../../examples');
const read = (name: string): unknown => JSON.parse(readFileSync(resolve(EXAMPLES, name), 'utf-8'));

const load = () => {
  const m = validate(read('batch.metrics.json'), 'batch.metrics.json');
  const t = validate(read('batch.trace.json'), 'batch.trace.json');
  if (m.kind !== 'metrics' || t.kind !== 'trace') { throw new Error('examples swapped'); }
  return { metrics: parseMetrics(m.file, 'm'), trace: parseTrace(t.file, 't'), traceFile: t.file };
};

describe('validate', () => {
  it('tells the two schemas apart', () => {
    expect(validate(read('batch.metrics.json'), 'm').kind).toBe('metrics');
    expect(validate(read('batch.trace.json'), 't').kind).toBe('trace');
  });

  it('rejects an unknown schema and names the file', () => {
    expect(() => validate({ schema: 'vllm-traces/0' }, 'old.json')).toThrow(/old.json: unsupported schema "vllm-traces\/0"/);
    expect(() => validate([], 'x.json')).toThrow(/\(none\)/);
  });

  it('points at the first offending field', () => {
    const bad = read('batch.trace.json') as TraceFile;
    bad.requests[2].token_text = bad.requests[2].token_text.slice(1);
    expect(() => validate(bad, 't.json')).toThrow('`requests[2].token_text` must be an array as long as `token_t`');
    const empty = { ...(read('batch.metrics.json') as MetricsFile), samples: [] };
    expect(() => validate(empty, 'm.json')).toThrow('`samples` must be a non-empty array');
  });
});

describe('the example run', () => {
  it('matches the summary the Python module computed for the same file', () => {
    const { trace } = load();
    const expected = read('batch.summary.json') as Record<string, number | null>;
    const got = runSummary(trace.requests) as unknown as Record<string, number | null>;
    expect(Object.keys(got).sort()).toEqual(Object.keys(expected).sort());
    for (const [k, v] of Object.entries(expected)) {
      if (v == null) { expect(got[k]).toBeNull(); } else { expect(got[k]).toBeCloseTo(v, 9); }
    }
  });

  it('lines the trace up with the metrics clock', () => {
    const { metrics, trace } = load();
    const run = combine(metrics, trace);
    // the notebook starts the recorder first, so the trace sits a little to the right
    expect(run.offset).toBeGreaterThan(0);
    expect(run.offset).toBeLessThan(1);
    expect(run.tEnd).toBeGreaterThanOrEqual(Math.max(...run.requests.map((r) => r.tEnd)));
    // shifting the clock must not change a single latency
    expect(run.summary.ttft_p50).toBeCloseTo(runSummary(trace.requests).ttft_p50!, 9);
    for (const r of run.requests) {
      expect(r.tRunStart).toBeGreaterThanOrEqual(r.tSubmit);
      expect(r.tRunStart).toBeLessThanOrEqual(r.tFirst!);
    }
  });

  it('reads the scheduler metrics', () => {
    const { metrics } = load();
    expect(metrics.blockSize).toBe(16);
    expect(metrics.nBlocks).toBeGreaterThan(0);
    expect(metrics.apc).toBe(true);
    expect(metrics.runningPeak).toBeGreaterThan(0);
    expect(metrics.rows[0].cSteps).toBe(0);
    expect(metrics.rows[metrics.rows.length - 1].cDone).toBe(8);
    expect(interpAt(metrics, 'kv', -1)).toBe(metrics.rows[0].kv);
  });

  it('reconstructs a KV pool no fuller than the server reported', () => {
    const { metrics, trace } = load();
    const run = combine(metrics, trace);
    expect(run.plan!.spilled).toBe(0);
    for (const row of metrics.rows) {
      const f = kvFrame(run.requests, run.plan!, run.nBlocks, run.apc, row.t);
      expect(f.used).toBeLessThanOrEqual(run.nBlocks);
    }
    // cache hits map onto one shared set of blocks
    const r = run.requests.find((q) => q.cachedTokens > 0)!;
    const mid = (r.tFirst! + r.tEnd) / 2;
    const f = kvFrame(run.requests, run.plan!, run.nBlocks, run.apc, mid);
    expect(f.states.some((s) => s === 3)).toBe(true);
  });

  it('shows prompt and output tokens with their content', () => {
    const { trace, traceFile } = load();
    const r = trace.requests[0];
    const first = tokenAt(r, trace.vocab, 0);
    expect(first.kind).toBe('prompt');
    expect(first.text).toBe(trace.vocab[String(r.promptIds![0])]);
    const out = tokenAt(r, trace.vocab, r.promptTokens);
    expect(out).toMatchObject({ kind: 'generated', j: 0, text: traceFile.requests[0].token_text[0] });
    expect(out.id).toBe(traceFile.requests[0].output_ids![0]);
    // decoding the prompt ids gives back the chat template around the messages
    const text = r.promptIds!.map((id) => trace.vocab[String(id)]).join('');
    expect(text).toContain(traceFile.requests[0].messages[1].content);
    const cut = tokenRange(r, trace.vocab, 0, r.promptTokens, 10);
    expect(cut).toHaveLength(11);
    expect(cut[6]).toEqual({ elided: r.promptTokens - 10 });
  });

  it('walks each request through its states', () => {
    const run = combine(load().metrics, load().trace);
    const r = run.requests[run.requests.length - 1];
    expect(stateAt(r, r.tSubmit - 1)).toBe('absent');
    expect(stateAt(r, r.tFirst!)).toBe('decode');
    expect(stateAt(r, r.tEnd)).toBe('finished');
    const axis = makeAxis(run, 'step');
    expect(axis.step).toBe(true);
    expect(axis.x(run.tEnd)).toBe(1);
    expect(axis.x(axis.tAt(0.5))).toBeCloseTo(0.5, 6);
  });
});

describe('panels on the metrics clock', () => {
  it('zooms the shared axis to a picked range, and maps outside it past the edges', () => {
    const { metrics, trace } = load();
    const run = combine(metrics, trace);
    const axis = makeAxis(run, 'time', [0.5, 1.5]);
    expect([axis.t0, axis.t1]).toEqual([0.5, 1.5]);
    expect(axis.x(1)).toBeCloseTo(0.5);
    // unclamped, so each chart clips at its own edge
    expect(axis.x(0)).toBeLessThan(0);
    expect(axis.tAt(2)).toBe(1.5);
    const steps = makeAxis(run, 'step', [0.5, 1.5]);
    expect(steps.x(0.5)).toBeCloseTo(0);
    expect(steps.x(1.5)).toBeCloseTo(1);
  });

  it('rates counters and reads histogram windows over the panel window', () => {
    const { metrics } = load();
    const win = rateWindow(metrics);
    expect(win).toBe(0.5);
    const [i0, i1] = windowAt(metrics, metrics.tEnd, win);
    expect(i0).toBeLessThan(i1);
    expect(rate(metrics, 'generation_tokens_total', 0, metrics.rows.length - 1)).toBeGreaterThan(0);
    expect(rate(metrics, 'generation_tokens_total', i1, i1)).toBeNull();
    const all = histWindow(metrics, 'time_to_first_token_seconds', 0, metrics.rows.length - 1);
    expect(all?.n).toBeGreaterThan(0);
    expect(histQuantile(all, 0.5)).toBeGreaterThan(0);
    expect(metrics.edges.time_to_first_token_seconds.at(-1)).toBe(Infinity);
  });

  it('reads the per-engine values and the server the recorder wrote', () => {
    const file = read('batch.metrics.json') as MetricsFile;
    const dp = parseMetrics({
      ...file,
      server: { version: '0.30.0', models: [{ id: 'm', max_model_len: 8192, parent: null }], config: null },
      samples: file.samples.map((x) => ({ ...x, e: { 0: { num_requests_running: 1 }, 1: { num_requests_running: 2 } } })),
    }, 'dp');
    expect(dp.engines).toEqual(['0', '1']);
    expect(dp.rows[0].e['1'].num_requests_running).toBe(2);
    expect(dp.server?.version).toBe('0.30.0');
    // the example has neither
    const { metrics } = load();
    expect([metrics.engines, metrics.server]).toEqual([[], null]);
  });
});

describe('sparse scrapes', () => {
  // a busy server can take most of a second to answer a scrape: a short run then has only a
  // few samples
  const sparse = () => {
    const file = read('batch.metrics.json') as MetricsFile;
    const n = file.samples.length;
    return parseMetrics({ ...file, samples: [file.samples[0], file.samples[n >> 1], file.samples[n - 1]] }, 'sparse');
  };

  it('rates the scrape window under the playhead, never an empty one', () => {
    const m = sparse();
    const t = m.rows[1].t / 2;
    expect(windowAt(m, t, rateWindow(m))).toEqual([0, 1]);
    expect(windowAt(m, -1, rateWindow(m))).toEqual([0, 1]);
    expect(windowAt(m, m.rows[1].t, rateWindow(m))).toEqual([0, 1]);
    expect(windowAt(m, m.tEnd + 1, rateWindow(m))).toEqual([1, 2]);
    expect(rate(m, 'generation_tokens_total', ...windowAt(m, t, rateWindow(m)))).toBeGreaterThan(0);
  });

  it('rates only the column under the playhead when scrapes are further apart than the window', () => {
    // a gap of 0.8 s and a 0.5 s window: counted back from the playhead, the window would
    // reach into the previous column and report its prefill where the chart shows none
    const file = read('batch.metrics.json') as MetricsFile;
    const s = file.samples[0];
    const at = (t: number, gen: number) => ({ ...s, t, rtt: 0.8, m: { ...s.m, generation_tokens_total: gen } });
    const m = parseMetrics({ ...file, samples: [at(0, 0), at(0.8, 0), at(1.6, 800), at(2.4, 800), at(3.2, 800)] }, 'gap');
    expect(windowAt(m, 2.7, rateWindow(m))).toEqual([3, 4]);
    expect(rate(m, 'generation_tokens_total', ...windowAt(m, 2.7, rateWindow(m)))).toBe(0);
    expect(rate(m, 'generation_tokens_total', ...windowAt(m, 1.2, rateWindow(m)))).toBeCloseTo(1000);
  });

  it('moves a sample without rtt back to the middle of its scrape', () => {
    const file = read('batch.metrics.json') as MetricsFile;
    const s = file.samples[0];
    // the first scrape starts at t=0 and returned after 0.8 s; t0 is the recorder's start
    const legacy = parseMetrics({ ...file, t0_unix: 100, samples: [0.8, 1.6, 2.4].map((t) => ({ ...s, t })) }, 'old');
    [0, 0.8, 1.6].forEach((v, i) => expect(legacy.times[i]).toBeCloseTo(v));
    expect(legacy.t0Unix).toBeCloseTo(100.4);
    expect(legacy.rtt).toBeCloseTo(0.8);
    // one with rtt is already there
    const stamped = parseMetrics({ ...file, t0_unix: 100, samples: [0.4, 1.2, 2].map((t) => ({ ...s, t, rtt: 0.8 })) }, 'new');
    expect(stamped.t0Unix).toBeCloseTo(100.4);
    // a fast server: nothing to speak of
    expect(load().metrics.rtt).toBeLessThan(0.01);
  });

  const row = (t: number, running: number, waiting: number) => ({ t, running, waiting }) as MetricsRow;

  it('counts the wait for the first token as prefill unless a scrape saw the request queued', () => {
    const { trace } = load();
    const [a, b] = trace.requests.slice(0, 2).map((r, i) => ({ ...r, tSubmit: 0, tFirst: 1 + i, tEnd: 3 + i }));
    // no scrape between submit and first token: no evidence of a queue
    assignRunStarts([row(-1, 0, 0), row(5, 0, 0)], [a, b]);
    expect([a.tRunStart, b.tRunStart]).toEqual([0, 0]);
    // one slot and one waiting at t=0.5: the later arrival waited at least until then
    b.tSubmit = 0.1;
    assignRunStarts([row(0.5, 1, 1), row(1.5, 2, 0)], [a, b]);
    expect([a.tRunStart, b.tRunStart]).toEqual([0, 0.5]);
    expect(stateAt(b, 0.3)).toBe('queued');
    expect(stateAt(b, 0.7)).toBe('prefill');
    // a full batch with nothing waiting: the request is not in the scheduler's queue yet
    assignRunStarts([row(0.5, 1, 0)], [a, b]);
    expect(b.tRunStart).toBe(0.1);
  });

  it('reads a decode stall over a counted preemption as this request preempted', () => {
    const { trace } = load();
    const tokenT = [1, 1.02, 1.04, 1.06, 2.5, 2.52];
    const r = { ...trace.requests[0], tSubmit: 0, tRunStart: 0, tFirst: 1, tEnd: 3, tokenT, preempted: [] as Array<[number, number]> };
    const preempt = (t: number, dPreempt: number) => ({ t, dPreempt }) as MetricsRow;
    assignPreemptions([preempt(0, 0), preempt(1, 0), preempt(2, 0)], [r]);
    expect(r.preempted).toEqual([]);
    assignPreemptions([preempt(0, 0), preempt(1, 0), preempt(2, 1)], [r]);
    expect(r.preempted).toEqual([[1.06, 2.5]]);
    expect(stateAt(r, 1.5)).toBe('preempted');
    expect(stateAt(r, 2.51)).toBe('decode');
  });

  it('fills a block slot by slot: prompt, then generated tokens as they arrive', () => {
    const { trace } = load();
    const r = { ...trace.requests[0], cachedTokens: 16 };
    // one large block holds the whole sequence, as on a hybrid model
    const B = 1072;
    const mid = r.tokenT[9];
    expect(blockFill(r, 0, B, mid, false)).toEqual({ start: 0, end: B, cached: 16, prompt: r.promptTokens, filled: r.promptTokens + 10 });
    expect(blockFill(r, 0, B, r.tEnd, false).filled).toBe(r.promptTokens + r.tokenT.length);
    // a cached block nobody reads still holds all of its slots
    expect(blockFill(r, 0, 16, r.tEnd, true)).toMatchObject({ start: 0, end: 16, filled: 16 });
  });
});
