/* Wire shapes of the two files the notebook writes (see docs/schemas.md), and the parsed
 * models the panels read. Field names on the wire are snake_case, as Python writes them */

export const TRACE_SCHEMA = 'vllm-traces/1';
export const METRICS_SCHEMA = 'vllm-metrics/1';

export interface MetricsFile {
  schema: typeof METRICS_SCHEMA;
  source?: string;
  recorded_at?: string;
  t0_unix?: number;
  interval_s?: number;
  info?: Record<string, Record<string, string>>;
  server?: ServerWire;
  buckets?: Record<string, Array<number | null>>;
  samples: Array<{
    // the midpoint of the scrape, and how long it took; without `rtt`, `t` is when the
    // response arrived
    t: number;
    rtt?: number;
    m: Record<string, number>;
    h?: Record<string, number[]>;
    l?: Record<string, Record<string, number>>;
    // per engine core, data parallel only
    e?: Record<string, Record<string, number>>;
  }>;
}

/* What the recorder read from `/version`, `/v1/models` and `/server_info` when it stopped.
 * `config` is null unless the server ran with VLLM_SERVER_DEV_MODE=1 */
export interface ServerWire {
  version: string | null;
  models: Array<{ id: string; max_model_len: number | null; parent: string | null }>;
  config: ServerConfig | null;
}

export interface ServerConfig {
  tp: number;
  pp: number;
  dp: number;
  dp_local: number;
  ep: boolean;
  nnodes: number;
  executor: string | null;
  dtype: string | null;
  quantization: string | null;
  kv_dtype: string | null;
  max_model_len: number | null;
  // per engine core
  max_num_seqs: number | null;
  max_num_batched_tokens: number | null;
  chunked_prefill: boolean | null;
  async_scheduling: boolean | null;
  enforce_eager: boolean | null;
  spec_method: string | null;
  spec_tokens: number | null;
  max_loras: number | null;
  kv_connector: string | null;
  kv_role: string | null;
  gpus: string[];
}

export interface TraceRequestWire {
  idx: number;
  ok: boolean;
  error: string | null;
  messages: Array<{ role: string; content: string }>;
  t_submit: number;
  t_first: number | null;
  t_end: number | null;
  token_t: number[];
  token_text: string[];
  output_ids: Array<number | null> | null;
  token_logprob: Array<number | null> | null;
  prompt_ids: number[] | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cached_tokens: number | null;
  finish_reason: string | null;
}

export interface TraceFile {
  schema: typeof TRACE_SCHEMA;
  created_at?: string;
  t0_unix?: number;
  model?: string;
  run?: Record<string, unknown>;
  requests: TraceRequestWire[];
  vocab: Record<string, string>;
}

/* One /metrics scrape with the derived per-window and since-start values */
export interface MetricsRow {
  t: number;
  m: Record<string, number>;
  // cumulative bucket counts aligned to `Metrics.edges`
  h: Record<string, number[]>;
  l: Record<string, Record<string, number>>;
  e: Record<string, Record<string, number>>;
  running: number;
  waiting: number;
  kv: number;
  dSteps: number;
  dDecode: number;
  dPrefill: number;
  dPreempt: number;
  dDone: number;
  msPerStep: number | null;
  tokPerStep: number | null;
  decodePerS: number;
  prefillPerS: number;
  cSteps: number;
  cPreempt: number;
  cDone: number;
  hitRateRun: number | null;
  cachedFrac: number | null;
}

export interface Metrics {
  name: string;
  source: string | null;
  info: Record<string, Record<string, string>>;
  server: ServerWire | null;
  // `engine` label values, one per data-parallel core; empty when the file has no `e`
  engines: string[];
  // upper bucket bounds per histogram, Infinity for the last
  edges: Record<string, number[]>;
  rows: MetricsRow[];
  times: number[];
  metricNames: string[];
  t0Unix: number | null;
  tEnd: number;
  // median scrape round trip: a sample is placed to within half of it against the trace
  rtt: number;
  hasSteps: boolean;
  hasKv: boolean;
  blockSize: number;
  nBlocks: number;
  apc: boolean | null;
  kvPeak: number;
  runningPeak: number;
  waitingPeak: number;
}

export interface Request {
  id: number;
  ok: boolean;
  error: string | null;
  tSubmit: number;
  tFirst: number | null;
  tEnd: number;
  tokenT: number[];
  tokenText: string[];
  outputIds: Array<number | null> | null;
  logprobs: Array<number | null> | null;
  promptIds: number[] | null;
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number | null;
  finish: string | null;
  // filled in by `combine`, once the trace is placed on the metrics clock
  tRunStart: number;
  // decode gaps read as a preemption and recompute (reconstructed, see `assignPreemptions`)
  preempted: Array<[number, number]>;
}

export interface Trace {
  name: string;
  model: string | null;
  run: Record<string, unknown>;
  t0Unix: number | null;
  requests: Request[];
  vocab: Record<string, string>;
}

export interface Summary {
  n: number;
  errors: number;
  wall_s: number;
  out_tokens: number;
  out_tok_per_s: number | null;
  req_per_s: number | null;
  [quantile: `${'ttft' | 'itl' | 'tpot' | 'e2e'}_p${number}`]: number | null;
}
