import { int, ms, pct, secs } from '../model/format';
import { countLE, histQuantile, histWindow, rate, rateWindow, windowAt } from '../model/metrics';
import { stateAt, type Run } from '../model/run';
import { quantile } from '../model/trace';
import type { MetricsRow } from '../model/types';
import { PALETTE } from '../theme';
import type { Tone } from './ui';

const SPARK_SAMPLES = 90;
const SPARK_W = 100;
const SPARK_BOX = 10;
const KV_PRESSURE = 0.92;
// output rate from the trace alone: tokens that arrived in the last half second
const TRACE_RATE_S = 0.5;

const Spark = ({ values, color }: { values: number[]; color: string }) => {
  const max = Math.max(...values, 0);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const step = SPARK_W / Math.max(1, values.length - 1);
  const path = values.map((v, i) => `${i ? 'L' : 'M'}${(i * step).toFixed(2)} ${(SPARK_BOX - ((v - min) / span) * SPARK_BOX).toFixed(2)}`).join('');
  return (
    <svg className="spark" viewBox={`0 0 ${SPARK_W} ${SPARK_BOX}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={path} fill="none" stroke={color} strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
};

interface KpiProps {
  label: string;
  value: string;
  sub: string;
  color: string;
  spark?: number[];
  tone?: Tone;
}

const Kpi = ({ label, value, sub, color, spark, tone = 'plain' }: KpiProps) => (
  <div className="kpi" data-tone={tone}>
    <span className="kpi-label">{label}</span>
    <strong style={{ color }}>{value}</strong>
    <span className="kpi-sub">{sub}</span>
    {spark && spark.length > 1 && <Spark values={spark} color={color} />}
  </div>
);

/* Latency so far, from the requests the trace measured: what a client saw */
function traceLatency(run: Run, t: number) {
  const ttft: number[] = [];
  const itl: number[] = [];
  for (const r of run.requests) {
    if (r.tFirst != null && r.tFirst <= t) { ttft.push(r.tFirst - r.tSubmit); }
    const n = countLE(r.tokenT, t);
    for (let k = 1; k < n; k++) { itl.push(r.tokenT[k] - r.tokenT[k - 1]); }
  }
  return { ttft, itl };
}

/* The run at the playhead: gauges as sampled, counters as rates over the window before it */
export function KpiStrip({ run, t }: { run: Run; t: number }) {
  const rec = run.metrics;
  const lat = run.trace ? traceLatency(run, t) : null;
  const ttftP = (q: number) => (lat ? quantile(lat.ttft, q) : null);
  const itlP = (q: number) => (lat ? quantile(lat.itl, q) : null);

  if (!rec) {
    const counts = { queued: 0, prefill: 0, decode: 0, preempted: 0, finished: 0 };
    let recent = 0;
    for (const r of run.requests) {
      const s = stateAt(r, t);
      if (s !== 'absent') { counts[s]++; }
      recent += countLE(r.tokenT, t) - countLE(r.tokenT, t - TRACE_RATE_S);
    }
    return (
      <div className="kpis">
        <Kpi label="Running" value={int(counts.prefill + counts.decode)} sub="from the trace" color={PALETTE.running} />
        <Kpi label="Waiting" value={int(counts.queued)} sub="submitted, no token yet" color={PALETTE.waiting} tone={counts.queued ? 'warn' : 'plain'} />
        <Kpi label="Output" value={`${int(recent / TRACE_RATE_S)}/s`} sub={`tokens, last ${TRACE_RATE_S} s`} color={PALETTE.decode} />
        <Kpi label="TTFT p95" value={secs(ttftP(0.95))} sub={`p50 ${secs(ttftP(0.5))} · so far`} color={PALETTE.accent} />
        <Kpi label="Inter-token p95" value={ms(itlP(0.95))} sub={`p50 ${ms(itlP(0.5))} · so far`} color={PALETTE.accent} />
        <Kpi label="Finished" value={`${counts.finished}/${run.requests.length}`} sub="requests" color={PALETTE.running} />
      </div>
    );
  }

  const [i0, i1] = windowAt(rec, t, rateWindow(rec));
  const row = rec.rows[i1];
  const recent = rec.rows.slice(Math.max(0, i1 - SPARK_SAMPLES + 1), i1 + 1);
  const spark = (pick: (r: MetricsRow) => number) => recent.map(pick);
  const perS = (v: number | null, d = 0) => (v == null ? '—' : `${d ? v.toFixed(d) : int(v)}/s`);
  const r = (metric: string) => rate(rec, metric, i0, i1);
  const decode = r('generation_tokens_total');
  const stepTokens = r('iteration_tokens_total_sum');
  const steps = r('iteration_tokens_total_count');
  const queries = r('prefix_cache_queries_total');
  const preempt = r('num_preemptions_total') ?? 0;
  // without a trace, latency comes from the server's histograms since the recording started
  const ttftH = i1 > 0 ? histWindow(rec, 'time_to_first_token_seconds', 0, i1) : null;
  const itlH = i1 > 0 ? histWindow(rec, 'inter_token_latency_seconds', 0, i1) : null;
  const ttft = (q: number) => (lat ? ttftP(q) : histQuantile(ttftH, q));
  const itl = (q: number) => (lat ? itlP(q) : histQuantile(itlH, q));
  const reasons = Object.entries(row.l.num_requests_waiting_by_reason ?? {})
    .filter(([, v]) => v > 0)
    .map(([k, v]) => `${k.replace('reason=', '')} ${int(v)}`)
    .join(' · ');
  const cores = rec.engines.length;
  const win = `last ${rateWindow(rec).toFixed(1)} s`;

  return (
    <div className="kpis">
      <Kpi label="Running" value={int(row.running)} sub={cores > 1 ? `across ${cores} engine cores` : 'in the batch'}
        color={PALETTE.running} spark={spark((x) => x.running)} />
      <Kpi label="Waiting" value={int(row.waiting)} sub={reasons || 'queue empty'}
        color={PALETTE.waiting} tone={row.waiting ? 'warn' : 'plain'} spark={spark((x) => x.waiting)} />
      <Kpi label="KV cache" value={rec.hasKv ? pct(row.kv) : '—'}
        sub={rec.nBlocks ? `${int(rec.nBlocks)} blocks${cores > 1 ? ' per core' : ''}` : 'pool size unknown'}
        color={PALETTE.kv} tone={row.kv > KV_PRESSURE ? 'bad' : 'plain'} spark={spark((x) => x.kv)} />
      <Kpi label="Output" value={perS(decode)} sub={`generated tokens, ${win}`}
        color={PALETTE.decode} spark={spark((x) => x.decodePerS)} />
      <Kpi label="Prefill" value={perS(stepTokens == null || decode == null ? null : Math.max(0, stepTokens - decode))} sub="prompt tokens"
        color={PALETTE.prefill} spark={spark((x) => x.prefillPerS)} />
      <Kpi label="TTFT p95" value={secs(ttft(0.95))} sub={`p50 ${secs(ttft(0.5))} · so far`} color={PALETTE.accent} />
      <Kpi label="Inter-token p95" value={ms(itl(0.95))} sub={`p50 ${ms(itl(0.5))} · so far`} color={PALETTE.accent} />
      <Kpi label="Prefix hits" value={queries ? pct((r('prefix_cache_hits_total') ?? 0) / queries) : '—'}
        sub={`${pct(row.hitRateRun)} since start`} color={PALETTE.hit} />
      <Kpi label="Finished" value={perS(r('request_success_total'), 1)} sub={`${int(row.cDone)} since start`}
        color={PALETTE.running} spark={spark((x) => x.dDone)} />
      <Kpi label="Preemptions" value={perS(preempt, 1)} sub={preempt > 0 ? 'KV pressure, sequences recomputed' : `${int(row.cPreempt)} since start`}
        color={preempt > 0 ? PALETTE.preempted : PALETTE.muted} tone={preempt > 0 ? 'bad' : 'plain'} spark={spark((x) => x.dPreempt)} />
      <Kpi label="Engine steps" value={perS(steps, 1)} sub={steps && stepTokens ? `${int(stepTokens / steps)} tokens/step` : 'idle'}
        color={PALETTE.accent} spark={spark((x) => x.dSteps)} />
    </div>
  );
}
