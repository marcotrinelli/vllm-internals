import { useRef } from 'react';

import { useTip } from '../components/hooks';
import { Legend, Row } from '../components/ui';
import { int, ms, secs } from '../model/format';
import { clamp } from '../model/metrics';
import { stateAt, type Axis, type Run } from '../model/run';
import { requestStats } from '../model/trace';
import { alpha, PALETTE, TRACK_LEFT } from '../theme';

const CAP = 256;

interface Props {
  run: Run;
  axis: Axis;
  t: number;
  selected: number | null;
  onSelect: (id: number) => void;
}

/* One row per request: queued (reconstructed from the running gauge), prefill, decode, and a
 * tick at every token's arrival */
export function Gantt({ run, axis, t, selected, onSelect }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const { tip, place, clear, style } = useTip<number>(wrapRef);
  const reqs = run.requests.slice(0, CAP);
  const rowH = clamp(Math.round(320 / Math.max(1, reqs.length)), 7, 18);
  const left = (v: number) => `${axis.x(v) * 100}%`;
  const width = (a: number, b: number) => `${Math.max(0.15, (axis.x(b) - axis.x(a)) * 100)}%`;
  const h = tip != null ? run.byId.get(tip) : undefined;
  const hs = h ? requestStats(h) : null;
  const hasQueue = run.metrics != null;

  return (
    <div className="hoverwrap" ref={wrapRef}>
      <Legend top items={[
        ...(hasQueue ? [[alpha(PALETTE.waiting, 0.45), 'queued (reconstructed)'] as [string, string]] : []),
        [PALETTE.prefill, hasQueue ? 'prefill' : 'submit → first token'],
        [PALETTE.decode, 'decode'],
        [alpha(PALETTE.text, 0.6), 'token arrival'],
        [PALETTE.critical, 'failed'],
      ]} />
      <div className="gantt" onMouseLeave={clear}>
        {reqs.map((r) => {
          const first = r.tFirst ?? r.tEnd;
          return (
            <div key={r.id} className={selected === r.id ? 'grow-row sel' : 'grow-row'} style={{ height: rowH }}
                 onClick={() => onSelect(r.id)} onMouseMove={(e) => place(e, r.id)}>
              <div className="glab num" style={{ width: TRACK_LEFT }}>{rowH >= 10 ? `#${r.id}` : ''}</div>
              <div className="gtrack">
                {r.ok ? (<>
                  {r.tRunStart > r.tSubmit && (
                    <div className="gseg" style={{ left: left(r.tSubmit), width: width(r.tSubmit, r.tRunStart), background: alpha(PALETTE.waiting, 0.45) }} />
                  )}
                  <div className="gseg" style={{ left: left(r.tRunStart), width: width(r.tRunStart, first), background: PALETTE.prefill }} />
                  {r.tFirst != null && (
                    <div className="gseg" style={{ left: left(r.tFirst), width: width(r.tFirst, r.tEnd), background: alpha(PALETTE.decode, 0.8) }} />
                  )}
                </>) : (
                  <div className="gseg" style={{ left: left(r.tSubmit), width: width(r.tSubmit, r.tEnd), background: PALETTE.critical }} />
                )}
                {rowH >= 10 && r.tokenT.length <= 1024 && (
                  <svg className="gticks" viewBox="0 0 1000 10" preserveAspectRatio="none" aria-hidden>
                    {r.tokenT.map((v, k) => {
                      const x = axis.x(v) * 1000;
                      return <line key={k} x1={x} x2={x} y1="2" y2="8" stroke={PALETTE.text}
                                   strokeOpacity={v <= t ? 0.6 : 0.15} vectorEffect="non-scaling-stroke" />;
                    })}
                  </svg>
                )}
              </div>
            </div>
          );
        })}
        {axis.x(t) >= 0 && axis.x(t) <= 1 && (
          <div className="ghead" style={{ left: `calc(${TRACK_LEFT}px + (100% - ${TRACK_LEFT}px) * ${axis.x(t)})` }} />
        )}
      </div>
      {run.requests.length > CAP && <Legend items={[]} right={`showing the first ${CAP} of ${int(run.requests.length)} requests`} />}
      {h && hs && (
        <div className="tip" style={style}>
          <h4>request #{h.id}<span className="sub">{h.ok ? stateAt(h, t) : 'failed'}</span></h4>
          {h.error && <div className="note tone-bad">{h.error}</div>}
          <Row k="submitted" v={secs(h.tSubmit)} />
          {hasQueue && <Row k="queued (reconstructed)" v={ms(hs.queue)} tone={hs.queue > 0.05 ? 'warn' : 'plain'} />}
          <Row k="TTFT" v={secs(hs.ttft)} />
          <Row k="TPOT" v={ms(hs.tpot)} />
          <Row k="end to end" v={secs(hs.e2e)} />
          <Row k="tokens in → out" v={`${int(h.promptTokens)} → ${int(h.tokenT.length)}`} />
          {h.cachedTokens > 0 && <Row k="prompt from prefix cache" v={int(h.cachedTokens)} tone="good" />}
        </div>
      )}
    </div>
  );
}
