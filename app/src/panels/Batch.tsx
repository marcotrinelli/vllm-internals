import type { ReactNode } from 'react';

import { Empty } from '../components/ui';
import { int } from '../model/format';
import { countLE } from '../model/metrics';
import { stateAt, type RequestState, type Run } from '../model/run';
import type { Request } from '../model/types';
import { reqColor } from '../theme';

const LANE_CAP = 48;
const DONE_CAP = 24;

interface Props {
  run: Run;
  t: number;
  selected: number | null;
  onSelect: (id: number) => void;
}

/* Requests at the playhead in the scheduler's three places: the waiting queue (preempted ones
 * go back to it), the running batch, finished */
export function Batch({ run, t, selected, onSelect }: Props) {
  if (!run.trace) { return <Empty>Drop the <code>vllm-traces/1</code> file to see requests move through the scheduler.</Empty>; }
  const waiting: Array<[Request, RequestState]> = [];
  const running: Array<[Request, RequestState]> = [];
  const done: Array<[Request, RequestState]> = [];
  for (const r of run.requests) {
    const st = stateAt(r, t);
    if (st === 'absent') { continue; }
    (st === 'finished' ? done : st === 'queued' || st === 'preempted' ? waiting : running).push([r, st]);
  }
  done.sort((a, b) => b[0].tEnd - a[0].tEnd);
  const rec = run.metrics;
  const cfg = rec?.server?.config;
  const limit = cfg?.max_num_seqs ? cfg.max_num_seqs * Math.max(1, rec!.engines.length) : null;
  const card = ([r, st]: [Request, RequestState], lane: string) => (
    <Card key={`${lane}-${r.id}`} r={r} st={st} t={t} focused={r.id === selected} onClick={() => onSelect(r.id)} />
  );
  return (
    <div className="pipe">
      <Lane title="WAITING" count={int(waiting.length)} note={rec ? `peak ${int(rec.waitingPeak)}` : undefined}
            more={waiting.length > LANE_CAP ? `+${int(waiting.length - LANE_CAP)}` : null}>
        {waiting.slice(0, LANE_CAP).map((x) => card(x, 'wait'))}
      </Lane>
      <Lane title="RUNNING" count={limit ? `${int(running.length)}/${int(limit)}` : int(running.length)}
            note={rec ? `peak ${int(rec.runningPeak)}` : undefined}
            more={running.length > LANE_CAP ? `+${int(running.length - LANE_CAP)}` : null}>
        {running.slice(0, LANE_CAP).map((x) => card(x, 'run'))}
      </Lane>
      <Lane title="FINISHED" count={`${int(done.length)}/${int(run.requests.length)}`} note="newest first"
            more={done.length > DONE_CAP ? `+${int(done.length - DONE_CAP)} earlier` : null}>
        {done.slice(0, DONE_CAP).map((x) => card(x, 'done'))}
      </Lane>
    </div>
  );
}

function Lane({ title, count, note, more, children }: { title: string; count: string; note?: string; more: string | null; children: ReactNode }) {
  return (
    <div className="lane">
      <div className="lh"><b>{title}</b><span>{count}{note ? ` · ${note}` : ''}</span></div>
      <div className="cards">
        {children}
        {more && <div className="more">{more}</div>}
      </div>
    </div>
  );
}

function Card({ r, st, t, focused, onClick }: { r: Request; st: RequestState; t: number; focused: boolean; onClick: () => void }) {
  const out = r.tokenT.length;
  const arrived = countLE(r.tokenT, t);
  const preempts = r.preempted.filter(([a]) => a < t).length;
  let label: string;
  let frac = 0;
  if (st === 'queued') {
    label = 'queued';
  } else if (st === 'preempted') {
    label = 'preempted · requeued';
  } else if (st === 'prefill') {
    // prefill progress is not reported: the share of the wait for the first token elapsed
    frac = r.tFirst == null ? 0 : (t - r.tRunStart) / Math.max(1e-9, r.tFirst - r.tRunStart);
    label = `prefill ${Math.round(Math.min(1, frac) * 100)}%`;
  } else if (st === 'decode') {
    frac = arrived / Math.max(1, out);
    label = `decode ${int(arrived)}/${int(out)}`;
  } else {
    label = r.ok ? `done · ${int(out)} tok` : 'failed';
  }
  const phase = st === 'finished' ? 'done' : st;
  return (
    <div className={`rcard ${phase}${focused ? ' focus' : ''}${preempts && st !== 'finished' ? ' was-preempted' : ''}`}
         style={{ borderLeftColor: reqColor(r.id) }} onClick={onClick}
         title={`request #${r.id} · ${int(r.promptTokens)} prompt tokens${r.cachedTokens ? ` (${int(r.cachedTokens)} cached)` : ''} · ${int(out)} output tokens${preempts ? ` · preempted ${preempts}×` : ''}`}>
      <div className="id">#{r.id}</div>
      <div className="ph">{label}</div>
      {frac > 0 && <div className="bar" style={{ width: `${Math.min(100, frac * 100)}%` }} />}
    </div>
  );
}
