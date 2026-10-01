import { useMemo, useRef, useState, type MouseEvent, type PointerEvent } from 'react';

import { useTip } from '../components/hooks';
import { Legend, Row } from '../components/ui';
import { fmt, int, pct } from '../model/format';
import { clamp } from '../model/metrics';
import type { Axis } from '../model/run';
import type { Metrics } from '../model/types';
import { alpha, PALETTE, TRACK_LEFT } from '../theme';

const W = 1000;
const H = { kv: 64, gap: 9, tok: 52, batch: 40, pad: 10 };
const TOTAL = H.kv + H.gap + H.tok + H.gap + H.batch + H.pad;
// a press that moves less than this is a click on the playhead, not a range
const DRAG_PX = 4;

interface Props {
  rec: Metrics;
  axis: Axis;
  t: number;
  onSeek: (t: number) => void;
  onRange: (a: number, b: number) => void;
}

interface Drag {
  px: number;
  t0: number;
  t1: number;
  moved: boolean;
}

/* KV utilisation, prefill/decode token rate, batch depth and preemptions per scrape window */
export function EngineTimeline({ rec, axis, t, onSeek, onRange }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const { tip, place, clear, style } = useTip<number>(wrapRef);
  const [drag, setDrag] = useState<Drag | null>(null);
  const R = rec.rows;
  const xv = useMemo(() => R.map((r) => axis.x(r.t) * W), [R, axis]);
  const maxTok = Math.max(1, ...R.map((r) => r.prefillPerS + r.decodePerS));
  const maxSeq = Math.max(1, rec.runningPeak + rec.waitingPeak);
  const tokTop = H.kv + H.gap;
  const batTop = tokTop + H.tok + H.gap;
  const kvPath = useMemo(() => {
    const d = R.map((r, i) => `${i ? 'L' : 'M'} ${xv[i].toFixed(2)} ${(H.kv - clamp(r.kv, 0, 1) * H.kv).toFixed(2)}`).join(' ');
    return { line: d, area: `M ${xv[0]} ${H.kv} ${d.replace(/^M/, 'L')} L ${xv[xv.length - 1]} ${H.kv} Z` };
  }, [R, xv]);

  // a captured drag keeps reporting past the edges, so f is clamped
  const locate = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const f = clamp((e.clientX - box.left) / Math.max(1, box.width), 0, 1);
    let i = 0;
    for (let k = 1; k < xv.length; k++) {
      if (Math.abs(xv[k] - f * W) < Math.abs(xv[i] - f * W)) { i = k; }
    }
    return { f, i };
  };
  const h = tip != null ? R[tip] : null;
  const px = axis.x(t) * W;

  const handleDown = (e: PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) { return; }
    e.currentTarget.setPointerCapture(e.pointerId);
    const at = axis.tAt(locate(e).f);
    setDrag({ px: e.clientX, t0: at, t1: at, moved: false });
  };
  const handleMove = (e: PointerEvent<SVGSVGElement>) => {
    const p = locate(e);
    if (drag) {
      setDrag({ ...drag, t1: axis.tAt(p.f), moved: drag.moved || Math.abs(e.clientX - drag.px) >= DRAG_PX });
      clear();
    } else {
      place(e, p.i);
    }
  };
  const handleUp = (e: PointerEvent<SVGSVGElement>) => {
    if (!drag) { return; }
    const at = axis.tAt(locate(e).f);
    setDrag(null);
    if (drag.moved) { onRange(drag.t0, at); } else { onSeek(at); }
  };
  const xa = drag ? axis.x(Math.min(drag.t0, drag.t1)) * W : 0;
  const xb = drag ? axis.x(Math.max(drag.t0, drag.t1)) * W : 0;

  return (
    <div className="hoverwrap" ref={wrapRef}>
      <Legend top items={[
        [PALETTE.kv, 'KV cache utilisation'], [PALETTE.prefill, 'prefill tok/s'], [PALETTE.decode, 'decode tok/s'],
        [PALETTE.running, 'running'], [alpha(PALETTE.waiting, 0.7), 'waiting'], [PALETTE.preempted, 'preemption'],
      ]} />
      <svg className="chart brushable" viewBox={`0 0 ${W} ${TOTAL}`} preserveAspectRatio="none"
           style={{ marginLeft: TRACK_LEFT, width: `calc(100% - ${TRACK_LEFT}px)` }}
           role="img" aria-label="Engine timeline"
           onPointerDown={handleDown} onPointerMove={handleMove} onPointerUp={handleUp}
           onPointerCancel={() => setDrag(null)} onPointerLeave={clear}>
        <line x1="0" y1={H.kv} x2={W} y2={H.kv} stroke={PALETTE.axis} strokeWidth=".7" />
        <path d={kvPath.area} fill={alpha(PALETTE.kv, 0.18)} />
        <path d={kvPath.line} fill="none" stroke={PALETTE.kv} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        <line x1="0" y1={batTop + H.batch} x2={W} y2={batTop + H.batch} stroke={PALETTE.axis} strokeWidth=".7" />
        {R.map((r, i) => {
          if (!i) { return null; }
          const x0 = xv[i - 1];
          const w = Math.max(0.8, xv[i] - x0);
          const hd = (r.decodePerS / maxTok) * H.tok;
          const hp = (r.prefillPerS / maxTok) * H.tok;
          const hr = (r.running / maxSeq) * H.batch;
          const hq = (r.waiting / maxSeq) * H.batch;
          return (
            <g key={i}>
              <rect x={x0} y={tokTop + H.tok - hd} width={w} height={hd} fill={PALETTE.decode} />
              <rect x={x0} y={tokTop + H.tok - hd - hp} width={w} height={hp} fill={PALETTE.prefill} />
              <rect x={x0} y={batTop + H.batch - hr} width={w} height={hr} fill={PALETTE.running} />
              <rect x={x0} y={batTop + H.batch - hr - hq} width={w} height={hq} fill={alpha(PALETTE.waiting, 0.7)} />
              {r.dPreempt > 0 && <rect x={x0} y={batTop + H.batch + 2} width={Math.max(1.4, w)} height={4} fill={PALETTE.preempted} />}
            </g>
          );
        })}
        <line x1={px} y1="0" x2={px} y2={TOTAL - 4} stroke={PALETTE.text} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        {drag?.moved && (
          <rect x={xa} width={Math.max(0, xb - xa)} y={0} height={TOTAL} fill={alpha(PALETTE.accent, 0.18)}
                stroke={PALETTE.accent} strokeWidth={1} vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      {drag?.moved && (
        <div className="brush-label num" style={{ left: `calc(${TRACK_LEFT}px + (100% - ${TRACK_LEFT}px) * ${(xa + xb) / 2 / W})` }}>
          {fmt(Math.min(drag.t0, drag.t1), 2)} – {fmt(Math.max(drag.t0, drag.t1), 2)} s · <b>{fmt(Math.abs(drag.t1 - drag.t0), 2)} s</b>
        </div>
      )}
      {h && (
        <div className="tip" style={style}>
          <h4>t = {fmt(h.t, 2)} s<span className="sub">{rec.hasSteps ? `step ${int(h.cSteps)}` : 'sample'}</span></h4>
          <Row k="KV utilisation" v={pct(h.kv)} tone={h.kv > 0.95 ? 'bad' : h.kv > 0.8 ? 'warn' : 'plain'} />
          <Row k="running / waiting" v={`${int(h.running)} / ${int(h.waiting)}`} tone={h.waiting ? 'warn' : 'plain'} />
          <Row k="engine steps in window" v={int(h.dSteps)} />
          <Row k="ms per step" v={fmt(h.msPerStep, 2)} />
          <Row k="tokens per step" v={fmt(h.tokPerStep, 1)} />
          <Row k="prefill tok/s" v={int(h.prefillPerS)} />
          <Row k="decode tok/s" v={int(h.decodePerS)} />
          <Row k="finished in window" v={int(h.dDone)} />
          <Row k="preemptions in window" v={int(h.dPreempt)} tone={h.dPreempt ? 'bad' : 'plain'} />
        </div>
      )}
    </div>
  );
}
