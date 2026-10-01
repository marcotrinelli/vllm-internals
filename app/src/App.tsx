import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { KpiStrip } from './components/KpiStrip';
import { Splash } from './components/Splash';
import type { TokenMode } from './components/Tokens';
import { TopBar } from './components/TopBar';
import { Card, Dialog, Empty } from './components/ui';
import { int } from './model/format';
import { clamp, parseMetrics } from './model/metrics';
import { combine, makeAxis, type AxisMode } from './model/run';
import { parseTrace } from './model/trace';
import type { Metrics, Trace } from './model/types';
import { validate } from './model/validate';
import { Batch } from './panels/Batch';
import { Deployment } from './panels/Deployment';
import { EngineTimeline } from './panels/EngineTimeline';
import { Gantt } from './panels/Gantt';
import { KvPool } from './panels/KvPool';
import { Latency } from './panels/Latency';
import { RequestDetail } from './panels/RequestDetail';
import { RunSummary } from './panels/RunSummary';
import { Scheduler } from './panels/Scheduler';

// a picked range narrower than this share of the run is a slip of the mouse
const MIN_RANGE = 1 / 200;

export default function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [hot, setHot] = useState(false);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState<AxisMode>('time');
  const [sel, setSel] = useState<number | null>(null);
  const [tokMode, setTokMode] = useState<TokenMode>('text');
  const [deployOpen, setDeployOpen] = useState(false);
  // a span picked on the engine timeline; every chart zooms to it, the playhead goes to its end
  // so the panels show all of it, and Play replays it from its start
  const [range, setRange] = useState<[number, number] | null>(null);
  const pickRef = useRef<HTMLInputElement>(null);

  const run = useMemo(() => (metrics || trace ? combine(metrics, trace) : null), [metrics, trace]);
  const axis = useMemo(() => (run ? makeAxis(run, mode, range) : null), [run, mode, range]);

  /* Any number of files at once; each is recognised by its `schema` and replaces the loaded
   * file of the same kind */
  const load = useCallback(async (files: FileList | File[]) => {
    const errs: string[] = [];
    for (const f of Array.from(files)) {
      try {
        const got = validate(JSON.parse(await f.text()), f.name);
        if (got.kind === 'metrics') { setMetrics(parseMetrics(got.file, f.name)); } else {
          const tr = parseTrace(got.file, f.name);
          setTrace(tr);
          setSel(tr.requests[0]?.id ?? null);
        }
      } catch (e) {
        errs.push(e instanceof SyntaxError ? `${f.name}: not JSON (${e.message})` : String((e as Error).message ?? e));
      }
    }
    setErrors(errs);
    setRange(null);
    setT(0);
    setPlaying(errs.length === 0);
  }, []);

  useEffect(() => {
    const over = (e: DragEvent) => { e.preventDefault(); setHot(true); };
    const leave = (e: DragEvent) => { if (e.relatedTarget == null) { setHot(false); } };
    const drop = (e: DragEvent) => {
      e.preventDefault();
      setHot(false);
      if (e.dataTransfer?.files.length) { void load(e.dataTransfer.files); }
    };
    window.addEventListener('dragover', over);
    window.addEventListener('dragleave', leave);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragover', over);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('drop', drop);
    };
  }, [load]);

  const start = range ? range[0] : run?.tStart ?? 0;
  const end = range ? range[1] : run?.tEnd ?? 0;

  useEffect(() => {
    if (!run || !playing) { return; }
    let last = performance.now();
    let raf = 0;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const dt = Math.min(0.25, (now - last) / 1000) * speed;
      last = now;
      setT((prev) => {
        if (prev + dt < end) { return prev + dt; }
        setPlaying(false);
        return end;
      });
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [run, playing, speed, end]);

  const seek = useCallback((v: number) => {
    if (!run) { return; }
    setPlaying(false);
    setT(clamp(v, run.tStart, run.tEnd));
  }, [run]);
  const togglePlay = useCallback(() => {
    if (!run) { return; }
    if (t >= end - 1e-6 || t < start) { setT(start); setPlaying(true); } else { setPlaying((p) => !p); }
  }, [run, t, start, end]);
  const pickRange = useCallback((a: number, b: number) => {
    if (!run) { return; }
    const lo = clamp(Math.min(a, b), run.tStart, run.tEnd);
    const hi = clamp(Math.max(a, b), run.tStart, run.tEnd);
    if (hi - lo < (run.tEnd - run.tStart) * MIN_RANGE) { return; }
    setRange([lo, hi]);
    setPlaying(false);
    setT(hi);
  }, [run]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!run || (e.target as HTMLElement).tagName === 'INPUT') { return; }
      if (e.code === 'Escape') { setRange(null); }
      if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
      if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
        e.preventDefault();
        seek(t + (e.shiftKey ? 1 : 0.1) * (e.code === 'ArrowLeft' ? -1 : 1));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [run, t, togglePlay, seek]);

  const picker = (
    <input ref={pickRef} type="file" accept=".json,application/json" multiple hidden
           onChange={(e) => { if (e.target.files?.length) { void load(e.target.files); } e.target.value = ''; }} />
  );
  const pick = () => pickRef.current?.click();

  if (!run || !axis) {
    return <>{picker}<Splash hot={hot} errors={errors} onPick={pick} /></>;
  }
  const selected = sel != null ? run.byId.get(sel) : undefined;
  return (
    <>
      {picker}
      <TopBar run={run} t={t} playing={playing} speed={speed} mode={mode} range={range} onClearRange={() => setRange(null)}
              onPlay={togglePlay} onSeek={seek} onSpeed={setSpeed} onMode={setMode}
              onDeployment={() => setDeployOpen(true)} onPick={pick}
              onDetach={(kind) => (kind === 'trace' ? setTrace(null) : setMetrics(null))} />
      <KpiStrip run={run} t={t} />
      {errors.map((e) => <div key={e} className="err banner">{e}</div>)}
      <main>
        <Card title="Engine" count={metrics ? `${int(metrics.rows.length)} scrapes · x = ${mode}` : undefined}
              note="One column per scrape window. Top: KV cache utilisation. Middle: tokens through the forward pass, prefill stacked on decode. Bottom: running requests with waiting ones on top, red ticks for preemptions. Click to move the playhead; press and drag left or right to zoom every chart to a range (Esc or × to go back). Hover for the values of that window.">
          {metrics ? <EngineTimeline rec={metrics} axis={axis} t={t} onSeek={seek} onRange={pickRange} />
            : <Empty>Drop the <code>vllm-metrics/1</code> file to see KV cache and scheduler metrics over time.</Empty>}
        </Card>
        <Card title="Continuous batching" count="FCFS · newest preempted first"
              note="Requests at the playhead. Waiting: submitted and not scheduled yet, or preempted and requeued (its blocks freed, its KV recomputed when it is rescheduled). Running: the batch, prefill then decode, one token per engine step. Finished: newest first. Entry into the batch is reconstructed from the running and waiting gauges, a preemption from a decode gap that spans a scrape window where the server counted one. Click a card to inspect the request.">
          <Batch run={run} t={t} selected={sel} onSelect={setSel} />
        </Card>
        <Card title="Requests" count={trace ? `${int(run.requests.length)} requests` : undefined}
              note="One row per request: queued (reconstructed from the running gauge), prefill until the first token, decode until the last, a tick at every token's arrival. Click a row for its tokens.">
          {trace ? <Gantt run={run} axis={axis} t={t} selected={sel} onSelect={setSel} />
            : <Empty>Drop the <code>vllm-traces/1</code> file to see each request and its tokens.</Empty>}
        </Card>
        <div className="grid g-7-5">
          <Card title={selected ? `Request #${selected.id}` : 'Request'}
                count={selected ? `${int(selected.promptTokens)} → ${int(selected.tokenT.length)} tok` : undefined}>
            <RequestDetail run={run} r={selected} t={t} mode={tokMode} onMode={setTokMode} />
          </Card>
          <Card title="KV cache blocks" count={run.nBlocks ? `${int(run.nBlocks)} × ${run.blockSize} tok` : undefined}
                note="One square is one physical KV block, from block 0 up to the highest one the run ever allocated (the blocks past it stay free the whole run and are not drawn). The pool as reconstructed from the trace: /metrics reports one utilisation number and the trace token counts and timings, so which physical block holds what is inferred. Zoom in (blocks per row) to see inside each block: its token slots coloured by prefix cache hit, prompt and generated, the reserved ones left empty, and from 1 or 2 per row each slot's token. Hover a block for its request, its token positions, its ref count and every token it holds (slots past the playhead dimmed, the hovered slot outlined); click it to select the request.">
            <KvPool run={run} t={t} onSelect={setSel} />
          </Card>
        </div>
        <div className="grid g-1-1">
          <Card title="Scheduler"
                note="Continuous batching as it ran: one dot per scrape, batch size against throughput. Total decode tok/s grows with the batch while each request's own tok/s falls; the dashed line is max_num_seqs (times the engine cores) when the recording has the server config. Tokens per step are against max_num_batched_tokens, the per-step budget chunked prefill fills.">
            <Scheduler run={run} t={t} t0={axis.t0} t1={axis.t1} />
          </Card>
          <Card title="Latency">
            {metrics ? <Latency rec={metrics} axis={axis} t={t} /> : <Empty>Drop the <code>vllm-metrics/1</code> file for the server's latency histograms over time.</Empty>}
          </Card>
        </div>
        <Card title="Run summary"><RunSummary run={run} /></Card>
      </main>
      {deployOpen && (
        <Dialog title="Deployment" count={metrics?.server?.version ? `vLLM ${metrics.server.version}` : undefined}
                note="Detected from the series the server exported; TP, PP and the scheduler limits come from /server_info, which the recorder reads when it stops and which only a server started with VLLM_SERVER_DEV_MODE=1 serves (dev endpoints, not for production)."
                onClose={() => setDeployOpen(false)}>
          <Deployment rec={metrics} t={t} />
        </Dialog>
      )}
      {hot && <div className="overlay">drop to load</div>}
    </>
  );
}
