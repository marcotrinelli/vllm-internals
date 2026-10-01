import { Empty, Info } from '../components/ui';
import { fmt, int, pct } from '../model/format';
import { rate, rateWindow, windowAt } from '../model/metrics';
import type { Metrics } from '../model/types';

const NEEDS_INFO = 'needs /server_info (VLLM_SERVER_DEV_MODE=1)';

type State = 'on' | 'off' | 'unknown';

interface Feature {
  label: string;
  state: State;
  detail: string;
}

const isSet = (v: string | undefined): v is string => v != null && v !== '' && v !== 'None';
const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;

/* What the deployment was, from the series the server exported and, when the recorder could
 * read it, its config */
function features(rec: Metrics): Feature[] {
  const cfg = rec.server?.config ?? null;
  const c = rec.info.cache_config_info ?? {};
  const last = rec.rows[rec.rows.length - 1];
  const cores = Math.max(1, rec.engines.length);
  const drafts = last.m.spec_decode_num_draft_tokens_total ?? 0;
  const accepted = last.m.spec_decode_num_accepted_tokens_total ?? 0;
  const external = last.l.prompt_tokens_by_source_total?.['source=external_kv_transfer'] ?? 0;
  const lora = rec.info.lora_requests_info;
  const mm = last.m.mm_cache_queries_total ?? 0;
  const parallel = (size: number | undefined): State => (size == null ? 'unknown' : size > 1 ? 'on' : 'off');
  const spec = cfg?.spec_method ?? null;
  return [
    { label: 'Tensor parallel', state: parallel(cfg?.tp), detail: cfg ? `${plural(cfg.tp, 'GPU')} per engine core` : NEEDS_INFO },
    { label: 'Pipeline parallel', state: parallel(cfg?.pp), detail: cfg ? plural(cfg.pp, 'stage') : NEEDS_INFO },
    { label: 'Data parallel', state: cores > 1 ? 'on' : 'off', detail: plural(cores, 'engine core') },
    { label: 'Prefix caching', state: rec.apc ? 'on' : 'off', detail: rec.apc ? `${isSet(c.prefix_caching_hash_algo) ? `${c.prefix_caching_hash_algo} hashing, ` : ''}block ${rec.blockSize}` : 'disabled in the cache config' },
    {
      label: 'Speculative decoding',
      state: spec || rec.metricNames.includes('spec_decode_num_drafts_total') ? 'on' : 'off',
      detail: spec ? `${spec}, ${cfg?.spec_tokens ?? '—'} tokens${drafts ? `, ${pct(accepted / drafts, 0)} accepted` : ''}`
        : drafts ? `${pct(accepted / drafts, 0)} of draft tokens accepted` : 'no draft series exported',
    },
    {
      label: 'KV connector / disagg',
      state: cfg?.kv_connector || external > 0 ? 'on' : 'off',
      detail: cfg?.kv_connector ? `${cfg.kv_connector}, ${cfg.kv_role ?? 'no role'}` : external > 0 ? `${int(external)} prompt tokens pulled` : 'no external KV transfer',
    },
    { label: 'KV offloading', state: isSet(c.kv_offloading_size) ? 'on' : 'off', detail: isSet(c.kv_offloading_size) ? `${c.kv_offloading_backend}, ${c.kv_offloading_size} GiB` : 'KV lives on GPU only' },
    { label: 'LoRA adapters', state: lora ? 'on' : 'off', detail: lora ? `${lora.running_lora_adapters || 'none'} running, max ${lora.max_lora ?? '—'}` : 'no adapter series exported' },
    { label: 'Multimodal', state: mm > 0 ? 'on' : 'off', detail: mm > 0 ? `${int(mm)} cache queries` : 'no multimodal cache queries' },
  ];
}

const EngineRows = ({ rec, t }: { rec: Metrics; t: number }) => {
  const [i0, i1] = windowAt(rec, t, rateWindow(rec));
  const row = rec.rows[i1];
  const first = rec.rows[0];
  const multi = rec.engines.length > 0;
  // a single core's values are the totals, and the recorder writes no per-engine copy of them
  const engines = multi ? rec.engines : ['0'];
  return (
    <div className="scroll">
      <table className="rows">
        <thead>
          <tr>
            <th>engine</th>
            <td>running</td>
            <td>waiting</td>
            <td>KV</td>
            <td>gen tok/s</td>
            <td>prefix hits</td>
            <td>preempted</td>
          </tr>
        </thead>
        <tbody>
          {engines.map((e) => {
            const at = (r: typeof row, name: string) => (multi ? r.e[e]?.[name] : r.m[name]) ?? 0;
            const r = (name: string) => rate(rec, name, i0, i1, multi ? e : undefined);
            const queries = r('prefix_cache_queries_total');
            const since = at(row, 'num_preemptions_total') - at(first, 'num_preemptions_total');
            return (
              <tr key={e}>
                <th>{e}</th>
                <td>{int(at(row, 'num_requests_running'))}</td>
                <td>{int(at(row, 'num_requests_waiting'))}</td>
                <td>{pct(at(row, 'kv_cache_usage_perc'), 0)}</td>
                <td>{int(r('generation_tokens_total'))}</td>
                <td>{queries ? pct((r('prefix_cache_hits_total') ?? 0) / queries, 0) : '—'}</td>
                <td className={since > 0 ? 'tone-bad' : undefined}>{int(since)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

interface Props {
  rec: Metrics | null;
  t: number;
}

export function Deployment({ rec, t }: Props) {
  if (!rec) { return <Empty>Drop the <code>vllm-metrics/1</code> file: the deployment is read from what the server exported.</Empty>; }
  const cfg = rec.server?.config ?? null;
  const c = rec.info.cache_config_info ?? {};
  const model = rec.server?.models.find((m) => !m.parent);
  const perCore = rec.engines.length > 1 ? ' per core' : '';
  const onOff = (v: boolean | null | undefined) => (v == null ? undefined : v ? 'on' : 'off');
  const config: Array<[string, string | undefined]> = [
    ['model', model?.id],
    ['max_model_len', model?.max_model_len == null ? undefined : int(model.max_model_len)],
    ['GPUs', cfg?.gpus.length ? `${cfg.gpus.length} × ${cfg.gpus[0].replace(/^NVIDIA\s+/, '')}` : undefined],
    ['weights dtype', cfg?.dtype ?? undefined],
    ['quantization', cfg ? cfg.quantization ?? 'none' : undefined],
    ['KV dtype', c.cache_dtype],
    ['block size', isSet(c.block_size) ? `${c.block_size} tokens` : undefined],
    ['GPU blocks', isSet(c.num_gpu_blocks) ? `${int(+c.num_gpu_blocks)}${perCore}` : undefined],
    ['KV tokens', isSet(c.kv_cache_size_tokens) ? int(+c.kv_cache_size_tokens) : undefined],
    ['max concurrency', isSet(c.kv_cache_max_concurrency) ? `${fmt(+c.kv_cache_max_concurrency, 1)} × max_model_len` : undefined],
    ['GPU memory fraction', isSet(c.gpu_memory_utilization) ? c.gpu_memory_utilization : undefined],
    ['max_num_seqs', cfg?.max_num_seqs == null ? undefined : `${int(cfg.max_num_seqs)}${perCore}`],
    ['max_num_batched_tokens', cfg?.max_num_batched_tokens == null ? undefined : `${int(cfg.max_num_batched_tokens)}${perCore}`],
    ['chunked prefill', onOff(cfg?.chunked_prefill)],
    ['async scheduling', onOff(cfg?.async_scheduling)],
    ['CUDA graphs', cfg?.enforce_eager == null ? undefined : cfg.enforce_eager ? 'off (eager)' : 'on'],
  ];

  return (
    <>
      <div className="features">
        {features(rec).map((f) => (
          <div key={f.label} className="feature" data-state={f.state}>
            <strong>{f.label}</strong>
            <span title={f.detail}>{f.detail}</span>
          </div>
        ))}
      </div>
      <div className="deploy-grid">
        <div>
          <h3 className="sub">Engine cores<Info text={`One row per data-parallel rank (the engine label in /metrics); tensor and pipeline parallel ranks live inside a core and do not appear separately. Rates over the ${rateWindow(rec).toFixed(1)} s before the playhead, preemptions since the recording started.`} /></h3>
          <EngineRows rec={rec} t={t} />
        </div>
        <div>
          <h3 className="sub">Config<Info text="Cache settings from cache_config_info in /metrics; the model, the GPUs and the scheduler limits from what the recorder read when it stopped (/v1/models, and /server_info only from a dev-mode server). Block and token counts are per engine core." /></h3>
          <div className="config">
            {config.filter(([, v]) => v != null).map(([k, v]) => <div key={k}><span>{k}</span><b title={v}>{v}</b></div>)}
          </div>
        </div>
      </div>
    </>
  );
}
