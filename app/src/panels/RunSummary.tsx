import { Empty } from '../components/ui';
import { fmt, int, ms, pct, secs } from '../model/format';
import type { Run } from '../model/run';

/* Whole-run latency from the trace (the same numbers `summary()` prints in the notebook) and
 * the scheduler's side of the story from /metrics */
export function RunSummary({ run }: { run: Run }) {
  const s = run.summary;
  const rec = run.metrics;
  const last = rec?.rows[rec.rows.length - 1];
  const rows: Array<[string, string, string?]> = [];
  if (run.trace) {
    rows.push(
      ['requests', `${int(s.n)}${s.errors ? ` · ${int(s.errors)} failed` : ''}`, s.errors ? 'tone-bad' : undefined],
      ['wall time', secs(s.wall_s)],
      ['output tok/s', int(s.out_tok_per_s)],
      ['requests/s', fmt(s.req_per_s, 2)],
      ['TTFT p50 / p95 / p99', `${secs(s.ttft_p50)} / ${secs(s.ttft_p95)} / ${secs(s.ttft_p99)}`],
      ['ITL p50 / p95 / p99', `${ms(s.itl_p50)} / ${ms(s.itl_p95)} / ${ms(s.itl_p99)}`],
      ['TPOT p50 / p95', `${ms(s.tpot_p50)} / ${ms(s.tpot_p95)}`],
      ['E2E p50 / p95', `${secs(s.e2e_p50)} / ${secs(s.e2e_p95)}`],
    );
  }
  if (rec && last) {
    rows.push(
      ['prefix cache hit rate', pct(last.hitRateRun)],
      ['prompt tokens from cache', pct(last.cachedFrac)],
      ['preemptions', int(last.cPreempt), last.cPreempt ? 'tone-bad' : undefined],
      ['KV utilisation peak', pct(rec.kvPeak), rec.kvPeak > 0.95 ? 'tone-bad' : undefined],
      ['running / waiting peak', `${int(rec.runningPeak)} / ${int(rec.waitingPeak)}`],
      ['engine steps', rec.hasSteps ? int(last.cSteps) : '—'],
    );
  }
  if (!rows.length) { return <Empty>Nothing loaded.</Empty>; }
  return (
    <table className="tbl"><tbody>
      {rows.map(([k, v, cls]) => <tr key={k}><td className="l">{k}</td><td className={cls}>{v}</td></tr>)}
    </tbody></table>
  );
}
