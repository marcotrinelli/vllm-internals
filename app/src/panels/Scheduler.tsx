import type { MouseEvent } from 'react';

import { useTip, useWidth } from '../components/hooks';
import { Empty, Info, Legend, Row, Stat } from '../components/ui';
import { fmt, int, ms, pct, secs } from '../model/format';
import { histWindow, idxAt, interpAt } from '../model/metrics';
import type { Run } from '../model/run';
import type { MetricsRow } from '../model/types';
import { PALETTE } from '../theme';

const PLOT_H = 150;
const PAD = { l: 38, r: 8, t: 8, b: 20 };
const DOT_R = 4;
const HIT_R = 14;

const STAGES = [
  { key: 'queue', hist: 'request_queue_time_seconds', label: 'queued', color: PALETTE.waiting },
  { key: 'prefill', hist: 'request_prefill_time_seconds', label: 'prefill', color: PALETTE.prefill },
  { key: 'decode', hist: 'request_decode_time_seconds', label: 'decode', color: PALETTE.decode },
] as const;

/* v rounded up to a quarter of its decade: 1,050 -> 1,250 */
const axisMax = (v: number): number => {
  const step = 10 ** Math.floor(Math.log10(Math.max(1, v))) / 4;
  return Math.ceil(v / step) * step;
};

interface Point {
  x: number;
  y: number;
  row: MetricsRow;
}

interface ScatterProps {
  title: string;
  points: Point[];
  current: Point | null;
  xMax: number;
  // max_num_seqs over all engine cores, drawn as the wall the batch cannot grow past
  xLimit: number | null;
  color: string;
}

/* One dot per scrape: how big the batch was against what it produced */
const Scatter = ({ title, points, current, xMax, xLimit, color }: ScatterProps) => {
  const [wrapRef, width] = useWidth<HTMLDivElement>(320);
  const { tip, place, clear, style } = useTip<Point>(wrapRef);
  const yMax = axisMax(Math.max(1, ...points.map((p) => p.y)));
  const X = (v: number) => PAD.l + (v / xMax) * (width - PAD.l - PAD.r);
  const Y = (v: number) => PAD.t + (1 - v / yMax) * (PLOT_H - PAD.t - PAD.b);

  const handleMove = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - box.left;
    const my = e.clientY - box.top;
    let best: Point | null = null;
    let bestD = HIT_R * HIT_R;
    for (const p of points) {
      const d = (X(p.x) - mx) ** 2 + (Y(p.y) - my) ** 2;
      if (d < bestD) { best = p; bestD = d; }
    }
    if (best) { place(e, best); } else { clear(); }
  };

  return (
    <div className="hoverwrap" ref={wrapRef}>
      <h3 className="sub">{title}</h3>
      <svg className="chart" width={width} height={PLOT_H} role="img" aria-label={title} onMouseMove={handleMove} onMouseLeave={clear}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={PAD.l} x2={width - PAD.r} y1={Y(f * yMax)} y2={Y(f * yMax)} stroke={f ? PALETTE.line : PALETTE.axis} />
            <text x={PAD.l - 6} y={Y(f * yMax) + 4} textAnchor="end" className="axis">{int(f * yMax)}</text>
          </g>
        ))}
        {[0, 0.5, 1].map((f) => <text key={f} x={X(f * xMax)} y={PLOT_H - 4} textAnchor={f ? (f === 1 ? 'end' : 'middle') : 'start'} className="axis">{int(f * xMax)}</text>)}
        {xLimit != null && <line x1={X(xLimit)} x2={X(xLimit)} y1={PAD.t} y2={PLOT_H - PAD.b} stroke={PALETTE.text2} strokeDasharray="3 3" />}
        {points.map((p) => <circle key={p.row.t} cx={X(p.x)} cy={Y(p.y)} r={DOT_R} fill={color} opacity={0.45} />)}
        {current && <circle cx={X(current.x)} cy={Y(current.y)} r={DOT_R + 2} fill={color} stroke={PALETTE.text} strokeWidth={2} />}
        {tip && <circle cx={X(tip.x)} cy={Y(tip.y)} r={DOT_R + 3} fill="none" stroke={PALETTE.text} strokeWidth={1.5} />}
      </svg>
      {tip && (
        <div className="tip" style={style}>
          <h4>t = {fmt(tip.row.t, 2)} s</h4>
          <Row k="running" v={int(tip.row.running)} />
          <Row k="decode tok/s" v={int(tip.row.decodePerS)} />
          <Row k="tok/s per request" v={fmt(tip.row.decodePerS / Math.max(1, tip.row.running), 1)} />
          <Row k="ms per step" v={fmt(tip.row.msPerStep, 1)} />
          <Row k="tokens per step" v={fmt(tip.row.tokPerStep, 0)} />
        </div>
      )}
    </div>
  );
};

/* Mean time per stage of the requests that finished in [t0, t]: measured per request when the
 * trace is loaded (queue is the reconstructed batch entry), the server's histograms otherwise */
const Lifecycle = ({ run, t0, t }: { run: Run; t0: number; t: number }) => {
  let means: number[] | null = null;
  if (run.trace) {
    const done = run.requests.filter((r) => r.ok && r.tFirst != null && r.tEnd >= t0 && r.tEnd <= t);
    if (done.length) {
      const mean = (f: (r: (typeof done)[number]) => number) => done.reduce((a, r) => a + f(r), 0) / done.length;
      means = [mean((r) => r.tRunStart - r.tSubmit), mean((r) => (r.tFirst ?? r.tEnd) - r.tRunStart), mean((r) => r.tEnd - (r.tFirst ?? r.tEnd))];
    }
  } else if (run.metrics) {
    const rec = run.metrics;
    const i0 = idxAt(rec, t0);
    const i1 = idxAt(rec, t);
    const ws = STAGES.map((s) => (i0 < i1 ? histWindow(rec, s.hist, i0, i1) : null));
    if (ws.every((w) => w && w.n > 0)) { means = ws.map((w) => w?.mean ?? 0); }
  }
  const total = means?.reduce((a, v) => a + v, 0) ?? 0;
  if (!means || !total) { return <Empty>No request finished in this range{run.trace ? '' : ' (or the server exports no queue, prefill and decode histograms)'}.</Empty>; }
  return (
    <>
      <div className="stack" role="img" aria-label="Request lifecycle">
        {STAGES.map((s, i) => means[i] > 0 && <i key={s.key} style={{ flexGrow: means[i], background: s.color }} />)}
      </div>
      <Legend items={STAGES.map((s, i) => [s.color, `${s.label} ${means[i] < 1 ? ms(means[i]) : secs(means[i])} (${pct(means[i] / total, 0)})`])} />
    </>
  );
};

interface Props {
  run: Run;
  t: number;
  // the shown range, as the engine timeline has it
  t0: number;
  t1: number;
}

/* Continuous batching as it ran: batch size against throughput, the per-step token budget,
 * and where a request's time went */
export function Scheduler({ run, t, t0, t1 }: Props) {
  const rec = run.metrics;
  if (!rec) { return <Empty>Drop the <code>vllm-metrics/1</code> file to see the batch the scheduler built.</Empty>; }
  const cfg = rec.server?.config ?? null;
  const row = rec.rows[idxAt(rec, t)];
  const end = Math.min(t, t1);
  const seen = rec.rows.filter((r, i) => i > 0 && r.t >= t0 && r.t <= end && r.running >= 1 && r.decodePerS > 0);
  const total = seen.map((r) => ({ x: r.running, y: r.decodePerS, row: r }));
  const each = seen.map((r) => ({ x: r.running, y: r.decodePerS / r.running, row: r }));
  const at = seen[seen.length - 1] === row ? seen.length - 1 : -1;
  // max_num_seqs bounds each engine core; the running gauge is summed over them
  const limit = cfg?.max_num_seqs == null ? null : cfg.max_num_seqs * Math.max(1, rec.engines.length);
  const budget = cfg?.max_num_batched_tokens ?? null;
  const xMax = axisMax(Math.max(4, limit ?? 0, ...seen.map((r) => r.running)));

  return (
    <>
      <div className="stats">
        <Stat label="running" value={int(row.running)} unit={limit ? ` / ${int(limit)} seqs` : undefined} />
        <Stat label="tokens per step" value={fmt(row.tokPerStep, 0)} unit={budget ? ` / ${int(budget)}` : undefined} />
        <Stat label="ms per step" value={fmt(row.msPerStep, 1)} />
        <Stat label="engine steps" value={rec.hasSteps ? int(interpAt(rec, 'cSteps', t)) : '—'} unit=" so far" />
      </div>
      {seen.length ? (
        <div className="pair">
          <Scatter title="decode tok/s vs running requests" points={total} current={at >= 0 ? total[at] : null} xMax={xMax} xLimit={limit} color={PALETTE.decode} />
          <Scatter title="tok/s per request vs running requests" points={each} current={at >= 0 ? each[at] : null} xMax={xMax} xLimit={limit} color={PALETTE.running} />
        </div>
      ) : (
        <Empty>No decoding in this range.</Empty>
      )}
      {limit != null && <div className="legend"><span><i className="line" style={{ background: PALETTE.text2 }} />max_num_seqs</span></div>}
      <h3 className="sub">Where a request's time went<Info text={run.trace
        ? 'Mean per stage for the requests that finished in the shown range, from the trace: queued until the reconstructed entry into the batch, prefill until the first token, decode until the last.'
        : 'Mean per stage for the requests that finished in the shown range, from the server histograms (request_queue_time, request_prefill_time, request_decode_time).'} /></h3>
      <Lifecycle run={run} t0={t0} t={end} />
    </>
  );
}
