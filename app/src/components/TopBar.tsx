import { fmt, int } from '../model/format';
import { interpAt } from '../model/metrics';
import type { AxisMode, Run } from '../model/run';
import { PauseIcon, PlayIcon, ReplayIcon, Seg, StartIcon } from './ui';

const SPEEDS: Array<[number, string]> = [[0.25, '0.25×'], [0.5, '0.5×'], [1, '1×'], [2, '2×'], [4, '4×']];
// a scrape this slow blurs metrics against the trace by a visible fraction of a short run
const SLOW_SCRAPE_S = 0.2;

interface TopBarProps {
  run: Run;
  t: number;
  playing: boolean;
  speed: number;
  mode: AxisMode;
  // a span picked on the engine timeline, zooming every chart
  range: [number, number] | null;
  onClearRange: () => void;
  onPlay: () => void;
  onSeek: (t: number) => void;
  onSpeed: (v: number) => void;
  onMode: (m: AxisMode) => void;
  onDeployment: () => void;
  onPick: () => void;
  onDetach: (kind: 'trace' | 'metrics') => void;
}

export function TopBar({ run, t, playing, speed, mode, range, onClearRange, onPlay, onSeek, onSpeed, onMode, onDeployment, onPick, onDetach }: TopBarProps) {
  const rec = run.metrics;
  const atEnd = t >= run.tEnd - 1e-6;
  const playLabel = playing ? 'Pause' : atEnd ? 'Replay' : 'Play';
  const model = run.trace?.model ?? rec?.server?.models.find((m) => !m.parent)?.id;
  const version = rec?.server?.version;

  return (
    <header className="topbar">
      <div className="topbar-row">
        <span className="brand">vLLM internals</span>
        {run.trace && (
          <span className="chip" title={run.trace.name}>
            trace <b>{int(run.requests.length)}</b> req
            <button type="button" className="x" aria-label="detach trace" onClick={() => onDetach('trace')}>×</button>
          </span>
        )}
        {rec && (
          <span className="chip" title={rec.source ?? rec.name}>
            metrics <b>{int(rec.rows.length)}</b> scrapes
            <button type="button" className="x" aria-label="detach metrics" onClick={() => onDetach('metrics')}>×</button>
          </span>
        )}
        {model && <span className="chip">{model}</span>}
        {version && <span className="chip">v{version}</span>}
        {rec && rec.engines.length > 1 && <span className="chip">{rec.engines.length} engine cores</span>}
        {run.trace && rec && Math.abs(run.offset) > 5 && (
          <span className="chip warn" title="the two files' t=0 are far apart: they may not be the same run">
            clocks <b>{fmt(run.offset, 1)} s</b> apart
          </span>
        )}
        {run.trace && rec && rec.rtt > SLOW_SCRAPE_S && (
          <span className="chip warn" title="the server read its counters somewhere inside each scrape: against the trace, metrics are placed to within half a scrape">
            scrapes take <b>{fmt(rec.rtt, 2)} s</b> · metrics ±{fmt(rec.rtt / 2, 2)} s
          </span>
        )}
        <span className="grow" />
        <button type="button" onClick={onDeployment}>Deployment</button>
        <button type="button" onClick={onPick}>Open files…</button>
      </div>
      <div className="topbar-row">
        <button type="button" className="primary icon" aria-label={playLabel} title={playLabel} onClick={onPlay}>
          {playing ? <PauseIcon /> : atEnd ? <ReplayIcon /> : <PlayIcon />}
        </button>
        <button type="button" className="icon" aria-label="Back to start" title="Back to start" onClick={() => onSeek(run.tStart)}><StartIcon /></button>
        <Seg label="speed" value={speed} options={SPEEDS} onChange={onSpeed} />
        <input className="scrub" type="range" aria-label="playhead" min={run.tStart} max={run.tEnd}
               step={(run.tEnd - run.tStart) / 2000} value={t} onChange={(e) => onSeek(+e.target.value)} />
        {range && (
          <span className="chip" title={`${fmt(range[0], 2)} s – ${fmt(range[1], 2)} s`}>
            <b>{fmt(range[1] - range[0], 2)} s</b> selected
            <button type="button" aria-label="Clear selection" onClick={onClearRange}>×</button>
          </span>
        )}
        <span className="clock num"><b>{fmt(t, 2)}</b> / {fmt(run.tEnd, 2)} s
          {rec?.hasSteps && <> · step <b>{int(interpAt(rec, 'cSteps', t))}</b></>}</span>
        <Seg label="x axis" value={mode} onChange={onMode}
             options={[['time', 'time'], ['step', 'engine step', !rec?.hasSteps]]} />
      </div>
    </header>
  );
}
