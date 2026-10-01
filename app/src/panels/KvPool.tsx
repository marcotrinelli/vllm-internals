import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';

import { useTip, useWidth } from '../components/hooks';
import { Empty, Legend, Row, Seg, Stat } from '../components/ui';
import { int, pct, secs } from '../model/format';
import { blockFill, kvFrame, type Frame } from '../model/kv';
import { idxAt, interpAt } from '../model/metrics';
import type { Run } from '../model/run';
import { tokenAt } from '../model/tokens';
import { alpha, PALETTE, reqColor } from '../theme';

const FILL = [PALETTE.free, alpha(PALETTE.cached, 0.42), alpha(PALETTE.accent, 0.85), alpha(PALETTE.waiting, 0.92)];
const NAME = ['free', 'cached, free', 'allocated', 'shared'];
const LEGEND: Array<[string, string]> = [
  [FILL[2], 'allocated (ref > 0)'], [FILL[3], 'shared (ref > 1)'], [FILL[1], 'cached, free: evictable (LRU)'], [FILL[0], 'free'],
];
// one square per block, drawn at this size and scaled to the panel's width
const CELL = 10;
// zoomed in, a block is framed and tinted by its state and its slots coloured by token kind
const TINT = [PALETTE.free, alpha(PALETTE.cached, 0.16), alpha(PALETTE.accent, 0.16), alpha(PALETTE.waiting, 0.2)];
const SLOT = [PALETTE.cached, PALETTE.prompt, PALETTE.generated];
const SLOT_LEGEND: Array<[string, string]> = [
  [SLOT[0], 'prefix cache hit'], [SLOT[1], 'prompt token'], [SLOT[2], 'generated token'], [TINT[2], 'reserved slot'],
];
// blocks per row; 0 is the whole pool
const ZOOMS: Array<[number, string]> = [[0, 'pool'], [16, '16'], [8, '8'], [4, '4'], [2, '2'], [1, '1']];
// rows scroll inside this height, in either view
const VIEW_H = 360;
// slots narrower than this read as one fill level per token kind; from TEXT_PX on they carry their token
const SLOT_PX = 2.5;
const TEXT_PX = 16;
const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

interface Props {
  run: Run;
  t: number;
  onSelect: (id: number) => void;
}

/* The block pool as reconstructed from the trace. /metrics reports one utilisation number and
 * the trace reports token counts and timings: which physical block holds what is inferred */
export function KvPool({ run, t, onSelect }: Props) {
  const rec = run.metrics;
  if (!rec) { return <Empty>Drop the <code>vllm-metrics/1</code> file of the same run: the pool size comes from its <code>cache_config_info</code>.</Empty>; }
  if (!run.nBlocks) { return <Empty>This recording has no <code>cache_config_info</code>, so the pool size is unknown.</Empty>; }
  const kv = rec.hasKv ? interpAt(rec, 'kv', t) : null;
  if (!run.plan) {
    return (
      <>
        <Meter server={kv} attributed={null} />
        <Empty>Drop the trace of the same run to attribute blocks to requests and see the tokens they hold.</Empty>
      </>
    );
  }
  return <Pool run={run} t={t} kv={kv} onSelect={onSelect} />;
}

const Meter = ({ server, attributed }: { server: number | null; attributed: number | null }) => (
  <div className="meter" title="server-reported utilisation, and the part the trace can attribute">
    <i style={{ width: pct(server ?? 0, 3), background: alpha(PALETTE.kv, 0.5) }} />
    {attributed != null && <i className="over" style={{ width: pct(attributed, 3), background: PALETTE.running }} />}
  </div>
);

// a token's text on a chip, whitespace made visible
const chipText = (text: string | null) => (text == null ? '?' : text.replace(/\n/g, '↵').replace(/\t/g, '→'));
// in a slot, leading space dropped (the slot's edge already separates it), cut to what fits
const slotText = (text: string | null, chars: number) => {
  const v = chipText(text).trim() || '·';
  if (v.length <= chars) { return v; }
  return chars <= 3 ? v.slice(0, chars) : `${v.slice(0, chars - 1)}…`;
};

interface Hover {
  b: number;
  slot: number; // -1 in the pool view, or off the slots
}

function Pool({ run, t, kv, onSelect }: Props & { kv: number | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const { tip, place, clear, style } = useTip<Hover>(wrapRef);
  const [zoom, setZoom] = useState(0);
  const [top, setTop] = useState(0);
  const [scrollRef, width] = useWidth<HTMLDivElement>();
  const plan = run.plan!;
  const n = run.nBlocks;
  // blocks are handed out from the bottom, so past maxHead the pool is free for the whole run
  const shown = plan.maxHead;
  const B = run.blockSize;
  const zoomed = zoom > 0;
  const cols = zoomed ? zoom : shown > 4096 ? 128 : 64;
  const rows = Math.ceil(shown / cols);
  const cell = zoomed ? Math.max(4, Math.floor(width / cols)) : CELL;
  const gap = cell >= 40 ? 2 : 1;
  const pad = cell >= 40 ? 3 : 1;
  // the slots of a block, row-major in a near-square grid
  const sc = Math.ceil(Math.sqrt(B));
  const s = (cell - gap - 2 * pad) / sc;
  const ratio = zoomed && typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  // zoomed, only the rows in view are drawn: a tall canvas would outgrow the browser's limits
  const viewH = zoomed ? Math.min(rows * cell, VIEW_H) : rows * CELL;
  const r0 = zoomed ? Math.floor(top / cell) : 0;
  const r1 = zoomed ? Math.min(rows, Math.ceil((top + viewH) / cell)) : rows;
  const frame: Frame = useMemo(() => kvFrame(run.requests, plan, n, run.apc, t), [run, plan, n, t]);
  const vocab = run.trace?.vocab;

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) { return; }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cols * cell, viewH);
    if (!zoomed) {
      let last = -1;
      for (let i = 0; i < shown; i++) {
        const st = frame.states[i];
        if (st !== last) { ctx.fillStyle = FILL[st]; last = st; }
        ctx.fillRect((i % cols) * CELL, Math.floor(i / cols) * CELL, CELL - 1, CELL - 1);
      }
    } else {
      const fs = Math.min(12, Math.floor(s * 0.5));
      const chars = Math.max(1, Math.floor((s - 3) / (fs * 0.6)));
      ctx.font = `${fs}px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let row = r0; row < r1; row++) {
        for (let c = 0; c < cols && row * cols + c < shown; c++) {
          const i = row * cols + c;
          const x = c * cell;
          const y = row * cell - top;
          const w = cell - gap;
          const st = frame.states[i];
          const r = st ? run.byId.get(frame.src[i]) : undefined;
          if (!r) {
            ctx.fillStyle = FILL[st];
            ctx.fillRect(x, y, w, w);
            continue;
          }
          ctx.fillStyle = TINT[st];
          ctx.fillRect(x, y, w, w);
          ctx.strokeStyle = FILL[st];
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, w - 1);
          const f = blockFill(r, frame.logical[i], B, t, st === 1);
          const cuts = [f.cached, f.prompt, f.filled];
          // a cached block nobody reads is resident but idle: its slots dimmed
          ctx.globalAlpha = st === 1 ? 0.55 : 1;
          for (let sr = 0; f.start + sr * sc < f.filled; sr++) {
            const a = f.start + sr * sc;
            const b = Math.min(a + sc, f.filled);
            const sy = y + pad + sr * s;
            if (s >= SLOT_PX) {
              for (let p = a; p < b; p++) {
                const sx = x + pad + (p - a) * s;
                ctx.fillStyle = SLOT[p < f.cached ? 0 : p < f.prompt ? 1 : 2];
                ctx.fillRect(sx, sy, s - 1, s - 1);
                if (s >= TEXT_PX && vocab) {
                  ctx.fillStyle = PALETTE.text;
                  ctx.fillText(slotText(tokenAt(r, vocab, p).text, chars), sx + (s - 1) / 2, sy + (s - 1) / 2, s - 3);
                }
              }
            } else {
              let from = a;
              for (let kind = 0; kind < 3 && from < b; kind++) {
                const to = Math.min(b, cuts[kind]);
                if (to > from) {
                  ctx.fillStyle = SLOT[kind];
                  ctx.fillRect(x + pad + (from - a) * s, sy, (to - from) * s, Math.max(0.5, s));
                  from = to;
                }
              }
            }
          }
          ctx.globalAlpha = 1;
        }
      }
    }
    if (tip != null) {
      const size = zoomed ? cell : CELL;
      const x = (tip.b % cols) * size;
      const y = Math.floor(tip.b / cols) * size - (zoomed ? top : 0);
      ctx.strokeStyle = PALETTE.text;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 1.5, y - 1.5, size + 2 - (zoomed ? gap : 0), size + 2 - (zoomed ? gap : 0));
      if (tip.slot >= 0 && s >= SLOT_PX) {
        ctx.lineWidth = 1;
        ctx.strokeRect(x + pad + (tip.slot % sc) * s - 0.5, y + pad + Math.floor(tip.slot / sc) * s - 0.5, s, s);
      }
    }
  }, [frame, run, vocab, t, shown, B, zoomed, cols, cell, gap, pad, sc, s, ratio, viewH, r0, r1, top, tip]);

  const locate = (e: MouseEvent<HTMLCanvasElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    if (!box.width || !box.height) { return; }
    let cx: number;
    let cy: number;
    let slot = -1;
    if (zoomed) {
      const px = e.clientX - box.left;
      const py = e.clientY - box.top + top;
      cx = Math.floor(px / cell);
      cy = Math.floor(py / cell);
      const qx = Math.floor((px - cx * cell - pad) / s);
      const qy = Math.floor((py - cy * cell - pad) / s);
      if (qx >= 0 && qx < sc && qy >= 0 && qy * sc + qx < B) { slot = qy * sc + qx; }
    } else {
      cx = Math.floor(((e.clientX - box.left) / box.width) * cols);
      cy = Math.floor(((e.clientY - box.top) / box.height) * rows);
    }
    const b = cy * cols + cx;
    if (cx < 0 || cx >= cols || b < 0 || b >= shown) { clear(); return; }
    place(e, { b, slot });
  };

  const pickZoom = (z: number) => {
    if (scrollRef.current) { scrollRef.current.scrollTop = 0; }
    setTop(0);
    setZoom(z);
    clear();
  };

  const hb = tip?.b ?? null;
  const owner = hb != null && frame.owner[hb] >= 0 ? run.byId.get(frame.owner[hb]) : undefined;
  const src = hb != null && frame.src[hb] >= 0 ? run.byId.get(frame.src[hb]) : undefined;
  const k = hb != null ? frame.logical[hb] : -1;
  const fill = src && k >= 0 ? blockFill(src, k, B, t, frame.states[hb!] === 1) : null;
  // the slots the sequence ever reaches; those past the playhead are reserved, not yet written
  const reach = src && fill ? Math.min(fill.end, src.promptTokens + src.tokenT.length) : 0;
  const chips = src && fill && vocab
    ? Array.from({ length: Math.max(0, reach - fill.start) }, (_, j) => tokenAt(src, vocab, fill.start + j)) : [];
  const holds = !src || !fill ? null
    : fill.start < fill.cached ? 'prompt, from the prefix cache'
    : fill.start >= src.promptTokens ? 'generated tokens (decode)'
    : fill.filled > src.promptTokens ? 'prompt and generated tokens'
    : 'prompt tokens (prefill)';
  // the hovered slot's token, marked in the chips
  const pos = fill && tip && tip.slot >= 0 ? fill.start + tip.slot : -1;
  const tok = src && vocab && fill && pos >= 0 && pos < reach ? tokenAt(src, vocab, pos) : null;
  const hitRate = run.metrics ? run.metrics.rows[idxAt(run.metrics, t)].hitRateRun : null;

  useEffect(() => {
    const list = listRef.current;
    const el = list?.querySelector<HTMLElement>('.mark');
    // both offsets are from the tooltip (the nearest positioned ancestor)
    if (list && el) { list.scrollTop = el.offsetTop - list.offsetTop - (list.clientHeight - el.offsetHeight) / 2; }
  }, [hb, pos]);

  return (
    <div className="hoverwrap" ref={wrapRef}>
      <Meter server={kv} attributed={frame.used / n} />
      <div className="statrow">
        <Stat label="server util" value={pct(kv)} tone={kv != null && kv > 0.95 ? 'bad' : 'plain'} />
        <Stat label="held by the trace" value={int(frame.used)} unit={` / ${int(n)}`} />
        <Stat label="cached, free" value={int(frame.cachedFree)} tone={frame.cachedFree ? 'good' : 'plain'} />
        <Stat label="prefix hit (run)" value={pct(hitRate)} />
      </div>
      <div className="legend top">
        <span>blocks per row</span>
        <Seg label="zoom" value={zoom} onChange={pickZoom} options={ZOOMS} />
        <span className="legend-right">
          {int(B)} token slots per block{zoomed ? '' : ' · zoom in to see them'}
          {shown < n && <> · #{int(shown)}–#{int(n - 1)} never allocated, not drawn</>}
        </span>
      </div>
      <div className="pool-scroll" ref={scrollRef} style={{ maxHeight: VIEW_H }}
           onScroll={(e) => setTop(e.currentTarget.scrollTop)}>
        <div style={{ height: zoomed ? rows * cell : undefined }}>
          <canvas ref={canvasRef} className={zoomed ? 'pool zoomed' : 'pool'}
                  width={Math.round(cols * (zoomed ? cell : CELL) * ratio)} height={Math.round(viewH * ratio)}
                  style={zoomed ? { width: cols * cell, height: viewH } : undefined}
                  onMouseLeave={clear} onClick={() => { if (owner) { onSelect(owner.id); } }}
                  onMouseMove={locate} />
        </div>
      </div>
      <Legend items={zoomed ? [...LEGEND, ...SLOT_LEGEND] : LEGEND} right={zoomed ? 'hover a slot for its token' : 'hover any block for its contents'} />
      {plan.spilled > 0 && <div className="note tone-warn">{int(plan.spilled)} block allocations did not fit the reconstructed pool.</div>}
      {hb != null && (
        <div className="tip wide" style={style}>
          <h4><i className="dot" style={{ background: FILL[frame.states[hb]] }} />physical block #{int(hb)}<span className="sub">{NAME[frame.states[hb]]}</span></h4>
          {frame.states[hb] === 0 ? <div className="note">In the free queue, no live KV.</div> : (<>
            {holds && <Row k="holds" v={holds} />}
            <Row k="request" v={owner ? <span style={{ color: reqColor(owner.id) }}>#{owner.id}</span>
              : `none live${src ? ` (last #${src.id})` : ''}`} />
            <Row k="logical block" v={`${int(k)} of that sequence`} />
            <Row k="token positions" v={`${int(k * B)}–${int((k + 1) * B - 1)}`} />
            <Row k="ref count" v={frame.ref[hb] > 1 ? `${int(frame.ref[hb])} sequences share it` : int(frame.ref[hb])}
                 tone={frame.ref[hb] > 1 ? 'warn' : 'plain'} />
            <Row k="resident since" v={secs(frame.allocT[hb])} />
            {tok && <Row k={`slot ${int(tip!.slot)}`} v={<code>{JSON.stringify(tok.text)}</code>} />}
            {!tok && tip!.slot >= 0 && <Row k={`slot ${int(tip!.slot)}`} v="empty" />}
            {chips.length > 0 && (
              <div className="toklist" ref={listRef}>
                {chips.map((c) => (
                  <span key={c.pos} className={`${c.kind}${c.pos < fill!.filled ? '' : ' pending'}${c.pos === pos ? ' mark' : ''}`}>{chipText(c.text)}</span>
                ))}
              </div>
            )}
            {fill && fill.filled < fill.end && (
              <div className="note">{int(fill.filled - fill.start)}/{int(B)} slots written, the rest reserved but not yet computed.</div>
            )}
          </>)}
        </div>
      )}
    </div>
  );
}
