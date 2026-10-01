# vllm-internals

Record what a vLLM server does with a handful of requests, then replay it: when each request
queued, prefilled and decoded, every token's arrival and content, and what the scheduler and
the KV cache were doing at the same instant.

Architecture: a small Python package (`vllm_internals/`) and a notebook drive the server's
OpenAI-compatible API and write two files per run, a per-request trace (`vllm-traces/1`) and
a `/metrics` time series (`vllm-metrics/1`). A static React app (`app/`) loads those files by
drag and drop and never talks to a server.

> [!NOTE]
> Use vllm-internals to understand how vLLM works inside, from a real inference: record a few
> requests against a running server, then replay them step by step to see scheduling,
> continuous batching, prefill and decode, and the KV cache blocks. To watch a real vLLM
> deployment live while it serves traffic (batch, queue, throughput, preemptions, latency),
> use [vllm-inspector](https://github.com/marcotrinelli/vllm-inspector).

## What the viewer shows

- KPI strip: running, waiting, KV cache, output and prefill tok/s, TTFT and inter-token p95
  (from the trace when it is loaded), prefix hits, finished, preemptions and engine steps at
  the playhead, each with a sparkline
- engine timeline (from the metrics file): KV cache utilisation, prefill and decode tokens/s,
  running and waiting requests, preemptions, on a wall-clock or engine-step x axis. Press and
  drag to zoom every chart to a range (`Esc` or × goes back); Play then replays that range
- request timeline (from the trace): one row per request, queued, prefill, decode, and a tick
  at every token's arrival; TTFT, TPOT and ITL per request and for the run
- continuous batching (from the trace): the requests at the playhead as cards in the waiting
  queue, the running batch and finished, moving between them as the run plays. With the
  metrics file a request that stalls mid-decode across a counted preemption shows as
  preempted and requeued (reconstructed: the trace has no preemption event)
- token content per request: the prompt as the model saw it (chat template included, prefix
  cache hits marked), the output streamed token by token up to the playhead, as text or as
  ids, with logprob, arrival time and KV block and slot on hover
- KV cache blocks (both files): one square per physical block, from block 0 up to the
  highest one the run allocated (the never-used rest of the pool is not drawn), coloured by
  allocated, shared, cached and free; zoom in (blocks per row) to see each block's token
  slots and then their tokens; hover a block for its request, token positions, ref
  count and the tokens it holds. The placement is reconstructed (no endpoint
  exports a block table), the counts and timings are measured
- scheduler (metrics file): decode tok/s and tok/s per request against the running batch, one
  dot per scrape, tokens per step against the step budget, and where a finished request's time
  went (queued, prefill, decode)
- latency (metrics file): p50, p95 and p99 over time from the server's histograms; hover for
  the time and the values
- deployment (metrics file, "Deployment" in the top bar): tensor, pipeline and data parallel, prefix caching, speculative
  decoding, KV connector and offloading, LoRA, one row per data-parallel engine core, and the
  cache and scheduler config. TP, PP and the scheduler limits need a server started with
  `VLLM_SERVER_DEV_MODE=1`: the recorder reads `/server_info` when it stops

## Quickstart

Start vLLM (any model; the flag makes the server report prefix-cache hits per request):

```sh
vllm serve Qwen/Qwen3-8B --port 8000 --enable-prompt-tokens-details
```

No GPU at hand? `uv run python tests/mock_vllm.py --port 8000` serves a fake engine with the
same endpoints (continuous batching, chunked prefill, paged KV with prefix caching, canned
text), enough to try everything below.

Record a run:

```sh
cp .env.example .env         # VLLM_BASE_URL, optional VLLM_MODEL and VLLM_API_KEY
uv sync                      # Python >= 3.10; installs the package, Jupyter and the dev tools
uv run jupyter lab notebooks/vllm_client.ipynb
```

Run all cells. They send one request and show its tokens, then record three runs while
`/metrics` is scraped every 100 ms, each written to `runs/<name>.trace.json` and
`runs/<name>.metrics.json`:

- `batch`: 8 concurrent requests sharing a system prompt
- `prefix`: 12 requests behind the same long document, sized to several KV blocks, so the
  later ones hit the prefix cache on its whole blocks
- `preempt`: 64 sequences that together outgrow the KV pool by half (`ignore_eos`), so the
  scheduler preempts. A production-size pool cannot be outgrown within `max_model_len`: the
  cell then prints the `--num-gpu-blocks-override` to restart vLLM with

Replay it:

```sh
cd app
npm ci
npm run dev                  # http://localhost:5274 (or `npm run build && npm run preview`)
```

Drop both files on the page (or use "Open files…"). Either file alone also works. Sample
files from the mock server are in `examples/`.

## Layout

```
vllm_internals/   client.py (config, streaming chat, /tokenize), trace.py (vllm-traces/1),
                  recorder.py (vllm-metrics/1), display.py (token chips in the notebook)
notebooks/        vllm_client.ipynb
app/              Vite + React + TypeScript viewer; parsing and validation in src/model/
docs/schemas.md   both file formats, the latency definitions, what is reconstructed
examples/         a small run recorded against tests/mock_vllm.py
tests/            pytest suite and the mock server
```

## Development

```sh
uv run ruff check . && uv run pytest              # Python
cd app && npm run typecheck && npm test && npm run build
```

Clear the notebook outputs before committing (`tests/test_notebook.py` fails otherwise).

## Configuration

`.env` (or the environment) holds three values: `VLLM_BASE_URL` (server root, a trailing
`/v1` is accepted), `VLLM_MODEL` (defaults to the first model `/v1/models` lists) and
`VLLM_API_KEY` (sent as a bearer token, to the API and to `/metrics`). Credentials never reach
the files: the metrics `source` has any `user:pass@` stripped and the key is only a header.

## License

MIT
