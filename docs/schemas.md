# File schemas

The notebook writes two JSON files per run and the viewer reads them. Each carries a
`schema` field (`<name>/<major>`); the viewer refuses any other value, so a breaking change
to either shape bumps the major. Writer and reader are pinned to each other by tests on both
sides: `tests/test_trace.py` and `app/src/model/model.test.ts` compute the run summary of
`examples/batch.trace.json` and compare it with `examples/batch.summary.json`.

All times are seconds, rounded to 0.1 ms, relative to the file's own t=0. Each file also
records the wall-clock instant of that t=0 (`t0_unix`), which is how the viewer puts the two
files on one clock: trace times are shifted by `trace.t0_unix - metrics.t0_unix`.

## `vllm-traces/1`: per-request timings and token content

Written by `vllm_internals.build_trace` (Python) and parsed by `app/src/model/trace.ts`,
validated by `app/src/model/validate.ts`.

```jsonc
{
  "schema": "vllm-traces/1",
  "created_at": "2026-01-01T12:00:00+00:00",
  "t0_unix": 1767268800.123456,        // wall clock at t=0 (Clock.t0_unix)
  "model": "Qwen/Qwen3-8B",
  "run": {"label": "batch", "concurrency": 8, "max_tokens": 128},  // free-form
  "requests": [{
    "idx": 0,
    "ok": true,                         // no error and at least one token streamed
    "error": null,                      // "HTTP 400: ...", "ConnectError: ...", ...
    "messages": [{"role": "user", "content": "..."}],
    "t_submit": 0.0012,                 // before the request was sent
    "t_first": 0.2031,                  // first token (TTFT = t_first - t_submit)
    "t_end": 1.9876,                    // stream closed
    "token_t": [0.2031, 0.2242, ...],   // arrival of every output token
    "token_text": ["The", " KV", ...],  // text of each, same length as token_t
    "output_ids": [791, 85437, ...],    // or null when the server returned no logprobs
    "token_logprob": [-0.01, -1.2, ...],// or null, same length as token_t
    "prompt_ids": [151644, 872, ...],   // from /tokenize, chat template applied; or null
    "prompt_tokens": 75,                // server-reported (usage), may be null
    "completion_tokens": 46,
    "cached_tokens": 64,                // prompt tokens served from the prefix cache; null
                                        // unless vLLM runs with --enable-prompt-tokens-details
    "finish_reason": "stop"
  }],
  "vocab": {"151644": "<|im_start|>", "872": "user", ...}   // prompt id -> decoded text
}
```

Definitions (same in Python and TypeScript, over `ok` requests):

- TTFT: `t_first - t_submit`
- ITL: every gap between consecutive `token_t` of a request, pooled across requests
- TPOT: per request, `(token_t[-1] - token_t[0]) / (len(token_t) - 1)`
- E2E: `t_end - t_submit`
- throughput: `sum(completion_tokens or len(token_t)) / (max(t_end) - min(t_submit))`
- percentiles: linear interpolation between closest ranks (numpy's default)

Where the token content comes from:

- output: the streamed `logprobs.content[]`, requested with `logprobs: true` and vLLM's
  `return_tokens_as_token_ids: true`. Each entry carries the id (`"token_id:791"`) and the
  UTF-8 `bytes` of the token's text. Several tokens can arrive in one chunk (speculative
  decoding, or a partial multi-byte character held back); they share the chunk's arrival
  time, so their ITL is 0. Without logprobs, one chunk counts as one token and
  `output_ids`/`token_logprob` are null
- prompt: `POST /tokenize` with the request's `messages` and `add_generation_prompt: true`,
  so the ids include the chat template. `vocab` is `POST /detokenize` of each distinct id on
  its own, which is how a token reads in isolation: a byte-level piece of a multi-byte
  character decodes to U+FFFD

## `vllm-metrics/1`: `/metrics` as a replayable time series

Written by `vllm_internals.Recorder` (Python) and parsed by `app/src/model/metrics.ts`. The
recorder scrapes on a background thread; nothing in it knows the workload, so it records any
vLLM server whoever drives it.

```jsonc
{
  "schema": "vllm-metrics/1",
  "source": "http://localhost:8000/metrics",   // credentials stripped
  "recorded_at": "2026-01-01T12:00:05+00:00",
  "t0_unix": 1767268800.101234,               // wall clock at the recorder's t=0
  "interval_s": 0.1,
  "duration_s": 2.43,
  "scrapes": 25,
  "errors": 0,                                 // failed scrapes (connection, timeout)
  "info": {"cache_config_info": {"block_size": "16", "num_gpu_blocks": "96",
                                 "enable_prefix_caching": "True", ...}},
  "server": {"version": "0.30.0",                // read once, when the recording stops
             "models": [{"id": "Qwen/Qwen3-8B", "max_model_len": 32768, "parent": null}],
             "config": {"tp": 1, "pp": 1, "dp": 1, "max_num_seqs": 256, ...}},  // null without dev mode
  "buckets": {"iteration_tokens_total": [1, 8, 16, ..., null]},   // null = +Inf
  "samples": [{
    "t": 0.42,                                                          // midpoint of the scrape
    "rtt": 0.84,                                                        // its round trip
    "m": {"num_requests_running": 2, "kv_cache_usage_perc": 0.3, ...},  // gauges, counters, _sum, _count
    "h": {"iteration_tokens_total": [3, 10, 12, ...]},                 // cumulative, aligned to buckets
    "l": {"request_success_total": {"finished_reason=stop": 5}},        // label splits
    "e": {"0": {"num_requests_running": 1, ...}, "1": {...}}            // data parallel only
  }]
}
```

Metric names have the `vllm:` prefix stripped; everything else is kept as the server
reported it. `*_info` series become `info` (labels are the payload), `*_created` series are
dropped, and labels that only identify the server (`model_name`, `engine`, ...) are folded
into the totals rather than split. With more than one data-parallel engine core each sample
also carries `e`, the same totals per `engine` label, and a `*_perc` gauge in `m` is the
mean over the cores rather than their sum (summed, KV usage would read past 100%).

A sample's `t` is the midpoint of its scrape and `rtt` the scrape's round trip: the server
read its counters somewhere in between, so against the trace a sample is placed to within
`rtt / 2`. On a busy or remote server a scrape can take most of a second, longer than
`interval_s`, and the recorder then scrapes back to back. Without `rtt`, `t` is when the
response arrived; the viewer moves those samples back by half the first scrape's duration
(the first scrape starts at t=0).

`server` comes from `/version`, `/v1/models` and `/server_info?config_format=json`. The last
is served only by a vLLM started with `VLLM_SERVER_DEV_MODE=1` (it enables the other dev
endpoints too, so not in production); without it `config` is null, and the viewer knows the
engine core count but not TP, PP or the scheduler limits. `config` is a whitelist flattened
from `vllm_config` (`tp`, `pp`, `dp`, `dp_local`, `ep`, `nnodes`, `executor`, `dtype`,
`quantization`, `kv_dtype`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`,
`chunked_prefill`, `async_scheduling`, `enforce_eager`, `spec_method`, `spec_tokens`,
`max_loras`, `kv_connector`, `kv_role`, `gpus`); the server's environment variables and
package list in the same body are never written. `max_num_seqs` and `max_num_batched_tokens`
are per engine core.

What the viewer reads from it: `num_requests_running`/`waiting`, `kv_cache_usage_perc`
(`gpu_cache_usage_perc` on older servers), `iteration_tokens_total_count`/`_sum` (engine
steps, and prefill tokens as step tokens minus `generation_tokens_total`),
`num_preemptions_total`, `request_success_total`, the prefix-cache counters, and the pool
size from `cache_config_info` (`block_size` and `num_gpu_blocks`), and the latency
histograms (`time_to_first_token_seconds`, `inter_token_latency_seconds`,
`e2e_request_latency_seconds`, `request_queue_time_seconds`, `request_prefill_time_seconds`,
`request_decode_time_seconds`) for the latency and request-lifecycle panels.

## What is reconstructed, not measured

`/metrics` reports how many requests are running and one KV utilisation number; it never says
which request is where or which block holds what. The viewer infers both, and labels them:

- queue vs prefill: before its first token a request is either waiting or being prefilled.
  The running gauge minus the requests already decoding gives the prefill slots in use, handed
  out FCFS; a request counts as queued only up to the last scrape that saw it without a slot
  while `num_requests_waiting` was non-zero, and with no such scrape its whole wait for the
  first token is prefill (a lower bound on the queue). Other clients on the same server make
  this over-assign
- preemption: the trace has no preemption event and `num_preemptions_total` only counts them.
  A decode gap longer than 5× the request's median gap (and at least 50 ms) that overlaps a
  scrape window where the counter rose is read as that request preempted and recomputed; it
  shows as preempted in the batching lanes for the length of the gap
- block placement: each request needs `ceil((prompt + generated) / block_size)` blocks,
  cache hits (`cached_tokens`) map onto one shared set, and blocks are handed out fresh-first
  then FIFO from the free list. The counts and timings are measured, the arrangement is not
