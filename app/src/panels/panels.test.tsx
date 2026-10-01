import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { renderToString } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { KpiStrip } from '../components/KpiStrip';
import { Splash } from '../components/Splash';
import { TopBar } from '../components/TopBar';
import { parseMetrics } from '../model/metrics';
import { combine, makeAxis } from '../model/run';
import { parseTrace } from '../model/trace';
import type { MetricsFile, TraceFile } from '../model/types';
import { Batch } from './Batch';
import { Deployment } from './Deployment';
import { EngineTimeline } from './EngineTimeline';
import { Gantt } from './Gantt';
import { KvPool } from './KvPool';
import { Latency } from './Latency';
import { RequestDetail } from './RequestDetail';
import { RunSummary } from './RunSummary';
import { Scheduler } from './Scheduler';

const EXAMPLES = resolve(__dirname, '../../../examples');
const read = <T,>(name: string) => JSON.parse(readFileSync(resolve(EXAMPLES, name), 'utf-8')) as T;
const metrics = parseMetrics(read<MetricsFile>('batch.metrics.json'), 'm');
const trace = parseTrace(read<TraceFile>('batch.trace.json'), 't');
const noop = () => undefined;

/* A server render of every panel on the example run: catches a panel that throws on real
 * data without a browser (effects, and so the canvas, do not run here) */
describe('panels render the example run', () => {
  const run = combine(metrics, trace);
  const axis = makeAxis(run, 'time');
  const t = run.tEnd / 2;
  const r = run.requests[3];

  it('with both files', () => {
    const html = [
      renderToString(<TopBar run={run} t={t} playing={false} speed={1} mode="time" range={[0, t]} onClearRange={noop}
                             onPlay={noop} onSeek={noop} onSpeed={noop} onMode={noop} onDeployment={noop}
                             onPick={noop} onDetach={noop} />),
      renderToString(<KpiStrip run={run} t={t} />),
      renderToString(<EngineTimeline rec={metrics} axis={axis} t={t} onSeek={noop} onRange={noop} />),
      renderToString(<Gantt run={run} axis={axis} t={t} selected={r.id} onSelect={noop} />),
      renderToString(<Batch run={run} t={t} selected={r.id} onSelect={noop} />),
      renderToString(<RequestDetail run={run} r={r} t={t} mode="text" onMode={noop} />),
      renderToString(<KvPool run={run} t={t} onSelect={noop} />),
      renderToString(<RunSummary run={run} />),
      renderToString(<Scheduler run={run} t={t} t0={axis.t0} t1={axis.t1} />),
      renderToString(<Latency rec={metrics} axis={axis} t={t} />),
      renderToString(<Deployment rec={metrics} t={t} />),
    ].join('');
    expect(html).toContain('vLLM internals');
    expect(html).toContain(`#${r.id}`);
    // the token content of the selected request is on the page
    expect(html).toContain('tk generated');
    expect(html).toContain(r.tokenText[0].replace(/&/g, '&amp;'));
    expect(html).toContain('TTFT p50 / p95 / p99');
    // the range chip, the KPI strip and the panels
    expect(html).toContain('selected');
    expect(html).toContain('Inter-token p95');
    expect(html).toContain('decode tok/s vs running requests');
    expect(html).toContain('time to first token over time');
    expect(html).toContain('RUNNING');
    // the example has no `server`
    expect(html).toContain('needs /server_info (VLLM_SERVER_DEV_MODE=1)');
  });

  it('with only one of the two files', () => {
    const traceOnly = combine(null, trace);
    const metricsOnly = combine(metrics, null);
    expect(renderToString(<KvPool run={traceOnly} t={0} onSelect={noop} />)).toContain('cache_config_info');
    expect(renderToString(<KpiStrip run={traceOnly} t={trace.requests[0].tEnd} />)).toContain('from the trace');
    expect(renderToString(<Scheduler run={traceOnly} t={0} t0={0} t1={1} />)).toContain('Drop the');
    expect(renderToString(<Deployment rec={null} t={0} />)).toContain('Drop the');
    expect(renderToString(<KvPool run={metricsOnly} t={0} onSelect={noop} />)).toContain('Drop the trace');
    expect(renderToString(<RequestDetail run={metricsOnly} r={undefined} t={0} mode="ids" onMode={noop} />))
      .toContain('Select a request');
    expect(renderToString(<Gantt run={traceOnly} axis={makeAxis(traceOnly, 'step')} t={0} selected={null} onSelect={noop} />))
      .toContain('submit → first token');
  });

  it('splash lists load errors', () => {
    expect(renderToString(<Splash hot={false} errors={['x.json: unsupported schema']} onPick={noop} />))
      .toContain('x.json: unsupported schema');
  });
});
