import { useMemo, useRef } from 'react';

import { useTip } from '../components/hooks';
import { Tokens, type TokenMode } from '../components/Tokens';
import { Empty, Row, Seg } from '../components/ui';
import { int, ms, pct, secs } from '../model/format';
import { countLE } from '../model/metrics';
import { stateAt, type Run } from '../model/run';
import { tokenRange } from '../model/tokens';
import { requestStats } from '../model/trace';
import type { Request } from '../model/types';
import { PALETTE } from '../theme';

const PROMPT_CAP = 1500;
const OUTPUT_CAP = 2000;

interface Props {
  run: Run;
  r: Request | undefined;
  t: number;
  mode: TokenMode;
  onMode: (m: TokenMode) => void;
}

export function RequestDetail({ run, r, t, mode, onMode }: Props) {
  if (!r || !run.trace) { return <Empty>Select a request in the timeline.</Empty>; }
  const s = requestStats(r);
  const st = stateAt(r, t);
  const vocab = run.trace.vocab;
  const arrived = countLE(r.tokenT, t);
  const prompt = tokenRange(r, vocab, 0, r.promptTokens, PROMPT_CAP);
  const output = tokenRange(r, vocab, r.promptTokens, r.promptTokens + arrived, OUTPUT_CAP);
  const table = run.plan?.tables.get(r.id);
  const B = run.blockSize;
  const block = table ? (pos: number) => {
    const lb = Math.floor(pos / B);
    const phys = table.blockIds[lb];
    return `logical ${int(lb)} · slot ${int(pos % B)} → ${phys >= 0 ? `physical #${int(phys)}` : 'not placed'}`;
  } : undefined;
  return (
    <>
      {r.error && <div className="note tone-bad">{r.error}</div>}
      <div className="detail-grid">
        <div>
          <Row k="state at playhead" v={st} tone={st === 'queued' ? 'warn' : st === 'preempted' ? 'bad' : st === 'decode' ? 'good' : 'plain'} />
          {run.metrics && <Row k="queued (reconstructed)" v={ms(s.queue)} tone={s.queue > 0.05 ? 'warn' : 'plain'} />}
          <Row k="TTFT" v={secs(s.ttft)} />
          <Row k="TPOT (mean ITL)" v={ms(s.tpot)} />
          <Row k="ITL p50 / max" v={`${ms(s.itlP50)} / ${ms(s.itlMax)}`} />
          <Row k="end to end" v={secs(s.e2e)} />
        </div>
        <div>
          <Row k="prompt tokens" v={int(r.promptTokens)} />
          <Row k="from prefix cache" v={r.cachedTokens ? `${int(r.cachedTokens)} · ${pct(r.cachedTokens / Math.max(1, r.promptTokens))}` : '0'}
               tone={r.cachedTokens ? 'good' : 'plain'} />
          <Row k="output tokens" v={`${int(arrived)} / ${int(r.tokenT.length)}`} />
          <Row k="finish reason" v={r.finish ?? '—'} />
          {r.preempted.length > 0 && <Row k="preempted (reconstructed)" v={`${int(r.preempted.length)}× (recompute)`} tone="bad" />}
        </div>
      </div>
      <ItlBars r={r} t={t} gaps={s.itl} tpot={s.tpot} />
      <div className="sub2">
        prompt<em>{int(r.promptTokens)} tokens{r.cachedTokens ? ` · ${int(r.cachedTokens)} cached` : ''}</em>
        <Seg label="token view" value={mode} onChange={onMode} options={[['text', 'text'], ['ids', 'ids']]} />
      </div>
      {r.promptIds ? <Tokens items={prompt} mode={mode} block={block} />
        : <Empty>This trace has no prompt ids (the server's <code>/tokenize</code> was not reached).</Empty>}
      <div className="sub2">output<em>one token per engine step · {int(arrived)} of {int(r.tokenT.length)} at the playhead · hover a token</em></div>
      {!r.tokenT.length ? <Empty>No token streamed back.</Empty>
        : arrived ? <Tokens items={output} mode={mode} live={st === 'decode' || st === 'preempted'} block={block}
                            stall={s.tpot != null ? 3 * s.tpot : null} />
        : <div className="toks muted">{st === 'absent' ? '…not submitted yet' : st === 'queued' ? '…queued, not scheduled yet' : '…prefilling, no token sampled yet'}</div>}
      {mode === 'ids' && !r.outputIds && (
        <Empty>No output ids: the server returned no logprobs, so tokens are the streamed text chunks.</Empty>
      )}
    </>
  );
}

/* Inter-token gaps; a bar over 3x the request's mean gap is a stall (a prefill of another
 * request sharing the step, a preemption, a scheduler hiccup) */
function ItlBars({ r, t, gaps, tpot }: { r: Request; t: number; gaps: number[]; tpot: number | null }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const { tip, place, clear, style } = useTip<number>(wrapRef);
  const W = 600;
  const H = 34;
  const max = useMemo(() => Math.max(1e-6, ...gaps), [gaps]);
  const bw = gaps.length ? W / gaps.length : W;
  const stall = (g: number) => tpot != null && g > 3 * tpot;
  return (
    <>
      <div className="sub2">inter-token latency<em>hover a bar</em></div>
      <div className="hoverwrap" ref={wrapRef}>
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onMouseLeave={clear}
             onMouseMove={(e) => {
               const box = e.currentTarget.getBoundingClientRect();
               const i = Math.floor(((e.clientX - box.left) / Math.max(1, box.width)) * gaps.length);
               if (i >= 0 && i < gaps.length) { place(e, i); } else { clear(); }
             }}>
          {gaps.map((g, i) => {
            const hh = Math.max(0.6, (g / max) * H);
            return <rect key={i} x={i * bw} y={H - hh} width={Math.max(0.7, bw - 0.3)} height={hh}
                         fill={stall(g) ? PALETTE.critical : PALETTE.decode}
                         opacity={r.tokenT[i + 1] <= t ? (tip === i ? 1 : 0.8) : 0.18} />;
          })}
        </svg>
        {tip != null && (
          <div className="tip" style={style}>
            <h4>token #{tip + 2}<span className="sub">of {r.tokenT.length}</span></h4>
            <Row k="gap from previous" v={ms(gaps[tip])} tone={stall(gaps[tip]) ? 'bad' : 'plain'} />
            <Row k="arrived at" v={secs(r.tokenT[tip + 1])} />
            <Row k="text" v={<code>{JSON.stringify(r.tokenText[tip + 1])}</code>} />
          </div>
        )}
      </div>
    </>
  );
}
