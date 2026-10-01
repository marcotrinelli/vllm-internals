import { useEffect, useRef } from 'react';

import { fmt, int, ms, secs } from '../model/format';
import { visible, type Elided, type Token } from '../model/tokens';
import { useTip } from './hooks';
import { Row } from './ui';

export type TokenMode = 'text' | 'ids';

const KIND_LABEL = { cached: 'prompt · prefix cache hit', prompt: 'prompt', generated: 'generated' } as const;

interface TokensProps {
  items: Array<Token | Elided>;
  mode: TokenMode;
  // still streaming: the newest token pops in, a cursor follows it and the list keeps it in view
  live?: boolean;
  // where a position's KV sits
  block?: (pos: number) => string;
  // a gap from the previous token above this (s) is flagged as a stall
  stall?: number | null;
}

/* Token content as chips, one per token, coloured by kind. Hover for id, position, logprob
 * and arrival time */
export function Tokens({ items, mode, live, block, stall }: TokensProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const { tip, place, clear, style } = useTip<Token>(wrapRef);
  useEffect(() => {
    const box = boxRef.current;
    if (live && box) { box.scrollTop = box.scrollHeight; }
  }, [live, items.length]);
  const stalled = tip?.gap != null && stall != null && tip.gap > stall;
  return (
    <div className="hoverwrap" ref={wrapRef}>
      <div className="toks" ref={boxRef} onMouseLeave={clear}>
        {items.map((x, i) => {
          if ('elided' in x) { return <span key={`e${i}`} className="more"> … {int(x.elided)} more … </span>; }
          return (
            <span key={x.pos} className={`tk ${x.kind}${live && i === items.length - 1 ? ' new' : ''}`}
                  onMouseMove={(e) => place(e, x)}>
              {mode === 'ids' ? `${x.id ?? '?'} ` : visible(x.text)}
            </span>
          );
        })}
        {live && <span className="cursor">▌</span>}
      </div>
      {tip && (
        <div className="tip" style={style}>
          <h4><i className={`dot ${tip.kind}`} />{tip.kind === 'generated' ? `output token #${int((tip.j ?? 0) + 1)}` : `position ${int(tip.pos)}`}
            <span className="sub">{KIND_LABEL[tip.kind]}</span></h4>
          <Row k="text" v={<code>{JSON.stringify(tip.text)}</code>} />
          <Row k="token id" v={tip.id ?? '—'} />
          {tip.kind === 'generated' && (<>
            <Row k="sequence position" v={int(tip.pos)} />
            <Row k="logprob" v={tip.logprob == null ? '—' : fmt(tip.logprob, 3)} />
            <Row k="arrived at" v={secs(tip.t)} />
            <Row k="since previous token" v={tip.gap == null ? 'first token' : ms(tip.gap)} tone={stalled ? 'warn' : 'plain'} />
          </>)}
          {block && <Row k="KV block" v={block(tip.pos)} />}
          {stalled && <div className="note">Several steps long: other requests' prefill shared those steps, or this one was preempted and recomputed.</div>}
        </div>
      )}
    </div>
  );
}
