import { METRICS_SCHEMA, TRACE_SCHEMA, type MetricsFile, type TraceFile } from './types';

export type Loaded =
  | { kind: 'metrics'; file: MetricsFile }
  | { kind: 'trace'; file: TraceFile };

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isNumOrNull = (v: unknown) => v === null || isNum(v);
const isNumArray = (v: unknown) => Array.isArray(v) && v.every(isNum);

/* Throws with the file name, the path of the first offending field and what was expected,
 * so a wrong or hand-edited file says what to fix instead of rendering nonsense */
function check(ok: boolean, name: string, path: string, expected: string): void {
  if (!ok) { throw new Error(`${name}: \`${path}\` must be ${expected}`); }
}

function validateMetrics(raw: Obj, name: string): MetricsFile {
  const samples = raw.samples;
  check(Array.isArray(samples) && samples.length > 0, name, 'samples', 'a non-empty array');
  (samples as unknown[]).forEach((s, i) => {
    check(isObj(s), name, `samples[${i}]`, 'an object');
    const o = s as Obj;
    check(isNum(o.t), name, `samples[${i}].t`, 'a number (seconds)');
    check(o.rtt === undefined || isNum(o.rtt), name, `samples[${i}].rtt`, 'a number (seconds)');
    check(isObj(o.m), name, `samples[${i}].m`, 'an object of metric values');
  });
  check(raw.t0_unix === undefined || isNum(raw.t0_unix), name, 't0_unix', 'a unix timestamp');
  check(raw.info === undefined || isObj(raw.info), name, 'info', 'an object');
  check(raw.server === undefined || raw.server === null || (isObj(raw.server) && Array.isArray(raw.server.models)),
    name, 'server', 'an object with a `models` array');
  check(raw.buckets === undefined || isObj(raw.buckets), name, 'buckets', 'an object {histogram: [upper bounds]}');
  return raw as unknown as MetricsFile;
}

function validateTrace(raw: Obj, name: string): TraceFile {
  check(Array.isArray(raw.requests), name, 'requests', 'an array');
  check(raw.vocab === undefined || isObj(raw.vocab), name, 'vocab', 'an object {id: text}');
  check(raw.t0_unix === undefined || isNum(raw.t0_unix), name, 't0_unix', 'a unix timestamp');
  (raw.requests as unknown[]).forEach((r, i) => {
    const p = `requests[${i}]`;
    check(isObj(r), name, p, 'an object');
    const o = r as Obj;
    check(isNum(o.idx), name, `${p}.idx`, 'a number');
    check(typeof o.ok === 'boolean', name, `${p}.ok`, 'a boolean');
    check(isNum(o.t_submit), name, `${p}.t_submit`, 'a number (seconds)');
    check(isNumOrNull(o.t_first), name, `${p}.t_first`, 'a number or null');
    check(isNumOrNull(o.t_end), name, `${p}.t_end`, 'a number or null');
    check(isNumArray(o.token_t), name, `${p}.token_t`, 'an array of numbers');
    check(Array.isArray(o.token_text) && o.token_text.length === (o.token_t as unknown[]).length,
      name, `${p}.token_text`, 'an array as long as `token_t`');
    for (const k of ['output_ids', 'token_logprob'] as const) {
      const v = o[k];
      check(v === null || (Array.isArray(v) && v.length === (o.token_t as unknown[]).length && v.every(isNumOrNull)),
        name, `${p}.${k}`, 'null or an array of numbers/nulls as long as `token_t`');
    }
    check(o.prompt_ids === null || isNumArray(o.prompt_ids), name, `${p}.prompt_ids`, 'null or an array of ids');
    for (const k of ['prompt_tokens', 'completion_tokens', 'cached_tokens'] as const) {
      check(o[k] === undefined || isNumOrNull(o[k]), name, `${p}.${k}`, 'a number or null');
    }
  });
  return { vocab: {}, ...raw } as unknown as TraceFile;
}

export function validate(raw: unknown, name: string): Loaded {
  const schema = isObj(raw) ? raw.schema : undefined;
  if (schema === METRICS_SCHEMA) { return { kind: 'metrics', file: validateMetrics(raw as Obj, name) }; }
  if (schema === TRACE_SCHEMA) { return { kind: 'trace', file: validateTrace(raw as Obj, name) }; }
  throw new Error(`${name}: unsupported schema ${schema === undefined ? '(none)' : JSON.stringify(schema)}, `
    + `expected "${TRACE_SCHEMA}" or "${METRICS_SCHEMA}"`);
}
