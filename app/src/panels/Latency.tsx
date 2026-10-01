import { useMemo, useState, type MouseEvent } from 'react';

import { useTip, useWidth } from '../components/hooks';
import { Empty, Info, Row, Stat } from '../components/ui';
import { fmt, int, ms, secs } from '../model/format';
import { histQuantile, histWindow, idxAt, rateWindow, windowAt } from '../model/metrics';
import type { Axis } from '../model/run';
import type { Metrics } from '../model/types';
import { PALETTE } from '../theme';

const H = 160;
const PAD = { l: 52, r: 8, t: 8, b: 18 };
const GRID = 3;
// histogram -> label, in picker order
const LATENCIES: Record<string, string> = {
  time_to_first_token_seconds: 'time to first token',
  inter_token_latency_seconds: 'inter-token latency',
  e2e_request_latency_seconds: 'end-to-end latency',
  request_queue_time_seconds: 'queue time',
  request_prefill_time_seconds: 'prefill time',
  request_decode_time_seconds: 'decode time',
  request_time_per_output_token_seconds: 'time per output token',
};
// p99 is dashed: amber and coral alone are too close to tell apart
const LINES = [
  { q: 0.5, label: 'p50', color: PALETTE.accent, dash: undefined },
  { q: 0.95, label: 'p95', color: PALETTE.waiting, dash: undefined },
  { q: 0.99, label: 'p99', color: PALETTE.critical, dash: '4 3' },
];

const dur = (v: number | null | undefined) => (v == null ? '—' : v < 1 ? ms(v) : secs(v));

/* The next 1, 2 or 5 step at or above v */
const niceMax = (v: number): number => {
  const e = 10 ** Math.floor(Math.log10(v));
  const f = v / e;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * e;
};

interface Point {
  t: number;
  n: number;
  v: Array<number | null>;
}

function series(rec: Metrics, base: string, win: number): Point[] {
  return rec.rows.map((r, i) => {
    const i0 = idxAt(rec, r.t - win);
    const w = i0 < i ? histWindow(rec, base, i0, i) : null;
    return { t: r.t, n: w?.n ?? 0, v: LINES.map((l) => histQuantile(w, l.q)) };
  });
}

interface Props {
  rec: Metrics;
  axis: Axis;
  t: number;
}

/* Latency quantiles over time, from the server's own histogram buckets */
export function Latency({ rec, axis, t }: Props) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const { tip, place, clear, style } = useTip<Point>(wrapRef);
  const [picked, setPicked] = useState('time_to_first_token_seconds');
  const options = Object.keys(LATENCIES).filter((b) => rec.edges[b]);
  const base = options.includes(picked) ? picked : options[0];
  const win = rateWindow(rec);
  const all = useMemo(() => (base ? series(rec, base, win) : []), [rec, base, win]);
  if (!base) { return <Empty>This recording has no latency histograms.</Empty>; }

  const points = all.filter((p) => p.t >= axis.t0 && p.t <= axis.t1);
  const [i0, i1] = windowAt(rec, t, win);
  const now = i0 < i1 ? histWindow(rec, base, i0, i1) : null;
  const yMax = niceMax(Math.max(1e-3, ...points.flatMap((p) => p.v.filter((v): v is number => v != null))));
  const plotW = Math.max(1, width - PAD.l - PAD.r);
  const X = (v: number) => PAD.l + axis.x(v) * plotW;
  const Y = (v: number) => PAD.t + (1 - Math.min(v, yMax) / yMax) * (H - PAD.t - PAD.b);
  const path = (k: number) => {
    let d = '';
    let pen = 'M';
    for (const p of points) {
      const v = p.v[k];
      if (v == null) { pen = 'M'; continue; }
      d += `${pen}${X(p.t).toFixed(1)} ${Y(v).toFixed(1)}`;
      pen = 'L';
    }
    return d;
  };

  const handleMove = (e: MouseEvent<SVGSVGElement>) => {
    const mx = e.clientX - e.currentTarget.getBoundingClientRect().left;
    let best: Point | null = null;
    for (const p of points) {
      if (!best || Math.abs(X(p.t) - mx) < Math.abs(X(best.t) - mx)) { best = p; }
    }
    if (best) { place(e, best); } else { clear(); }
  };

  const shown = tip ?? points.filter((p) => p.t <= t).pop();
  const px = axis.x(t);

  return (
    <>
      <div className="legend top">
        <select value={base} onChange={(e) => setPicked(e.target.value)} aria-label="Latency">
          {options.map((b) => <option key={b} value={b}>{LATENCIES[b]}</option>)}
        </select>
        <Info text={`Quantiles from the server's own histogram buckets. Each point covers the observations in the ${win.toFixed(1)} s before it (a twentieth of the recording), so a quiet stretch reads as a gap, not as zero. The top bucket is unbounded: a quantile landing in it is reported at its lower edge. Hover the chart for the time and the values.`} />
      </div>
      <div className="stats">
        <Stat label="observed" value={int(now?.n ?? 0)} />
        <Stat label="mean" value={dur(now?.mean)} />
        {LINES.map((l) => <Stat key={l.label} label={l.label} value={dur(histQuantile(now, l.q))} />)}
      </div>
      <div className="hoverwrap" ref={wrapRef}>
        <svg className="chart" width={width} height={H} role="img" aria-label={`${LATENCIES[base]} over time`}
             onMouseMove={handleMove} onMouseLeave={clear}>
          <defs><clipPath id="latency-plot"><rect x={PAD.l} y={0} width={plotW} height={H} /></clipPath></defs>
          {Array.from({ length: GRID + 1 }, (_, g) => {
            const v = (yMax / GRID) * g;
            return (
              <g key={g}>
                <line x1={PAD.l} x2={width - PAD.r} y1={Y(v)} y2={Y(v)} stroke={g ? PALETTE.line : PALETTE.axis} />
                <text x={PAD.l - 6} y={Y(v) + 3} textAnchor="end" className="axis">{dur(v)}</text>
              </g>
            );
          })}
          <text x={PAD.l} y={H - 3} className="axis">{fmt(axis.t0, 2)} s</text>
          <text x={width - PAD.r} y={H - 3} textAnchor="end" className="axis">{fmt(axis.t1, 2)} s</text>
          <g clipPath="url(#latency-plot)">
            {LINES.map((l, k) => <path key={l.label} d={path(k)} fill="none" stroke={l.color} strokeWidth={2} strokeDasharray={l.dash} strokeLinejoin="round" />)}
          </g>
          {px >= 0 && px <= 1 && <line x1={X(t)} x2={X(t)} y1={PAD.t} y2={H - PAD.b} stroke={PALETTE.text} strokeWidth={1.5} />}
          {tip && (
            <g>
              <line x1={X(tip.t)} x2={X(tip.t)} y1={PAD.t} y2={H - PAD.b} stroke={PALETTE.muted} />
              {LINES.map((l, k) => tip.v[k] != null && <circle key={l.label} cx={X(tip.t)} cy={Y(tip.v[k] ?? 0)} r={4} fill={l.color} stroke={PALETTE.panel} strokeWidth={2} />)}
            </g>
          )}
        </svg>
        <div className="chart-legend">
          {LINES.map((l, k) => <span key={l.label}><i style={{ background: l.color }} />{l.label}<b>{dur(shown?.v[k])}</b></span>)}
          <span className="chart-stamp">{shown ? `t = ${fmt(shown.t, 2)} s` : ''}</span>
        </div>
        {tip && (
          <div className="tip" style={style}>
            <h4>t = {fmt(tip.t, 2)} s<span className="sub">last {win.toFixed(1)} s</span></h4>
            {LINES.map((l, k) => <Row key={l.label} k={l.label} v={dur(tip.v[k])} />)}
            <Row k="observed" v={int(tip.n)} />
          </div>
        )}
      </div>
    </>
  );
}
