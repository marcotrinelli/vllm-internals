"""A tiny fake vLLM server, to exercise the client, the recorder and the viewer without a GPU.

It is not a model: the "tokenizer" splits on words and punctuation and the output is canned
text. What it does model is the scheduling the viewer draws: continuous batching capped at
`max_num_seqs`, chunked prefill with a per-step token budget, a paged KV pool with prefix
caching on whole blocks, and the `/metrics` series vLLM exports for all of it.

Endpoints: `GET /v1/models`, `POST /v1/chat/completions` (stream=true only, with logprobs
and `return_tokens_as_token_ids`), `POST /tokenize`, `POST /detokenize`, `GET /metrics`, and
`GET /version` and `GET /server_info` as a server started with `VLLM_SERVER_DEV_MODE=1`.

    uv run python tests/mock_vllm.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "mock/tiny-chat"
BLOCK = 16

_PIECE = re.compile(r"<\|[a-z_]+\|>| ?[A-Za-z]+| ?\d| ?[^\sA-Za-z\d]|\s+")

_CORPUS = (
    "Paged attention stores the key and value cache in fixed-size blocks, so a sequence "
    "never needs one contiguous slab of memory. The scheduler admits waiting requests "
    "while free blocks remain, runs prefill in chunks under a token budget, and decodes "
    "one token per running sequence at every engine step. Blocks that hold a shared "
    "prefix are hashed and reused, which is why a second request with the same system "
    "prompt skips most of its prefill. When the pool runs dry, the newest sequence is "
    "preempted and its blocks are freed until memory is available again.\n"
)


class Vocab:
    """Word-piece ids assigned on first sight, so decode(encode(s)) == s for any text"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ids: dict[str, int] = {}
        self.pieces: list[str] = []

    def encode(self, text: str) -> list[int]:
        with self.lock:
            out = []
            for p in _PIECE.findall(text):
                if p not in self.ids:
                    self.ids[p] = len(self.pieces)
                    self.pieces.append(p)
                out.append(self.ids[p])
            return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] if 0 <= i < len(self.pieces) else "�" for i in ids)


def chat_template(messages: list[dict[str, Any]], add_generation_prompt: bool = True) -> str:
    s = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class Histogram:
    def __init__(self, edges: list[float]):
        self.edges = edges
        self.counts = [0] * (len(edges) + 1)
        self.sum = 0.0

    def observe(self, v: float) -> None:
        self.sum += v
        self.counts[next((i for i, e in enumerate(self.edges) if v <= e), len(self.edges))] += 1

    def render(self, name: str, labels: str) -> list[str]:
        out, cum = [], 0
        for e, c in zip([*self.edges, math.inf], self.counts):
            cum += c
            le = "+Inf" if math.isinf(e) else repr(float(e))
            out.append(f'{name}_bucket{{{labels},le="{le}"}} {float(cum)}')
        out += [f"{name}_sum{{{labels}}} {self.sum}", f"{name}_count{{{labels}}} {float(cum)}"]
        return out


@dataclass
class Seq:
    prompt: list[int]
    output: list[int]  # the tokens this sequence will emit, decided up front
    finish: str
    events: queue.Queue = field(default_factory=queue.Queue)
    t_arrive: float = field(default_factory=time.perf_counter)
    t_last: float = 0.0
    computed: int = 0  # prompt tokens with KV in the pool (cache hits included)
    hits: list[int] = field(default_factory=list)  # hashes of the prefix blocks hit
    emitted: int = 0

    def blocks(self) -> int:
        return math.ceil((self.computed + self.emitted) / BLOCK)


def block_hashes(ids: list[int]) -> list[int]:
    out, h = [], 0
    for k in range(len(ids) // BLOCK):
        h = hash((h, tuple(ids[k * BLOCK : (k + 1) * BLOCK])))
        out.append(h)
    return out


class Engine(threading.Thread):
    def __init__(self, num_blocks: int, max_num_seqs: int, step_s: float, prefill_budget: int):
        super().__init__(daemon=True, name="mock-engine")
        self.num_blocks, self.max_num_seqs = num_blocks, max_num_seqs
        self.step_s, self.prefill_budget = step_s, prefill_budget
        self.lock = threading.Lock()
        self.waiting: deque[Seq] = deque()
        self.running: list[Seq] = []
        self.cached: set[int] = set()  # block hashes the prefix cache can hit
        self.c = dict.fromkeys(
            ["prompt", "generation", "cached", "queries", "hits", "stop", "length"], 0
        )
        self.h_iter = Histogram([1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        lat = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
        self.h_ttft, self.h_itl = Histogram(lat), Histogram(lat)
        self.h_e2e = Histogram([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0, 60.0])

    def submit(self, s: Seq) -> None:
        with self.lock:
            self.waiting.append(s)

    def used_blocks(self) -> int:
        shared = {h for s in self.running for h in s.hits}
        return len(shared) + sum(s.blocks() - len(s.hits) for s in self.running)

    def run(self) -> None:
        while True:
            with self.lock:
                self.step()
            time.sleep(self.step_s)

    def step(self) -> None:
        now = time.perf_counter()
        # admit FCFS, reserving the whole sequence's blocks so nothing is ever preempted
        while self.waiting and len(self.running) < self.max_num_seqs:
            s = self.waiting[0]
            hashes = block_hashes(s.prompt)
            hits = 0
            while hits < len(hashes) and hashes[hits] in self.cached:
                hits += 1
            hits = min(hits, (len(s.prompt) - 1) // BLOCK)  # the last token is always computed
            need = math.ceil((len(s.prompt) + len(s.output)) / BLOCK) - hits
            if self.used_blocks() + need > self.num_blocks:
                break
            self.waiting.popleft()
            s.hits, s.computed = hashes[:hits], hits * BLOCK
            self.c["queries"] += len(s.prompt)
            self.c["hits"] += s.computed
            self.c["cached"] += s.computed
            self.c["prompt"] += len(s.prompt)
            self.running.append(s)

        budget, tokens = self.prefill_budget, 0
        for s in list(self.running):
            if s.computed < len(s.prompt):
                n = min(budget, len(s.prompt) - s.computed)
                if n <= 0:
                    continue
                s.computed += n
                budget -= n
                tokens += n
                if s.computed < len(s.prompt):
                    continue
                self.cached.update(block_hashes(s.prompt))
                self.h_ttft.observe(now - s.t_arrive)  # first token sampled with the last chunk
            else:
                tokens += 1
                self.h_itl.observe(now - s.t_last)
            s.events.put(("tok", s.output[s.emitted]))
            s.emitted += 1
            s.t_last = now
            self.c["generation"] += 1
            if s.emitted == len(s.output):
                s.events.put(("done", s.finish))
                self.c[s.finish] += 1
                self.h_e2e.observe(now - s.t_arrive)
                self.running.remove(s)
        if tokens:
            self.h_iter.observe(tokens)

    def metrics(self, model: str) -> str:
        with self.lock:
            lab = f'engine="0",model_name="{model}"'
            c = self.c
            lines = [
                "# HELP vllm:num_requests_running Number of requests in model execution batches.",
                "# TYPE vllm:num_requests_running gauge",
                f"vllm:num_requests_running{{{lab}}} {float(len(self.running))}",
                f"vllm:num_requests_waiting{{{lab}}} {float(len(self.waiting))}",
                f"vllm:kv_cache_usage_perc{{{lab}}} {self.used_blocks() / self.num_blocks}",
                f"vllm:prefix_cache_queries_total{{{lab}}} {float(c['queries'])}",
                f"vllm:prefix_cache_hits_total{{{lab}}} {float(c['hits'])}",
                f"vllm:num_preemptions_total{{{lab}}} 0.0",
                f"vllm:prompt_tokens_total{{{lab}}} {float(c['prompt'])}",
                f"vllm:prompt_tokens_cached_total{{{lab}}} {float(c['cached'])}",
                f"vllm:generation_tokens_total{{{lab}}} {float(c['generation'])}",
                f"vllm:generation_tokens_created{{{lab}}} 1.7e9",
                *(
                    f'vllm:request_success_total{{{lab},finished_reason="{k}"}} {float(c[k])}'
                    for k in ("stop", "length")
                ),
                *self.h_iter.render("vllm:iteration_tokens_total", lab),
                *self.h_ttft.render("vllm:time_to_first_token_seconds", lab),
                *self.h_itl.render("vllm:inter_token_latency_seconds", lab),
                *self.h_e2e.render("vllm:e2e_request_latency_seconds", lab),
                f'vllm:cache_config_info{{block_size="{BLOCK}",cache_dtype="auto",engine="0",'
                f'enable_prefix_caching="True",gpu_memory_utilization="0.9",'
                f'num_gpu_blocks="{self.num_blocks}"}} 1.0',
            ]
        return "\n".join(lines) + "\n"


def server_info(engine: Engine) -> dict[str, Any]:
    """The `config_format=json` shape, trimmed to what a single mock GPU would report"""
    return {
        "vllm_config": {
            "model_config": {
                "dtype": "torch.bfloat16",
                "quantization": None,
                "max_model_len": 4096,
                "enforce_eager": True,
            },
            "cache_config": {"cache_dtype": "auto", "block_size": BLOCK},
            "parallel_config": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "distributed_executor_backend": "uni",
            },
            "scheduler_config": {
                "max_num_seqs": engine.max_num_seqs,
                "max_num_batched_tokens": engine.prefill_budget,
                "enable_chunked_prefill": True,
                "async_scheduling": False,
            },
            "speculative_config": None,
            "lora_config": None,
            "kv_transfer_config": None,
        },
        "vllm_env": {"VLLM_SERVER_DEV_MODE": "1"},
        "system_env": {"nvidia_gpu_models": "GPU 0: Mock GPU 1GB"},
    }


def make_handler(engine: Engine, vocab: Vocab, model: str) -> type[BaseHTTPRequestHandler]:
    corpus = vocab.encode(_CORPUS)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # keep test output clean
            pass

        def _json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/v1/models":
                card = {"id": model, "object": "model", "max_model_len": 4096, "parent": None}
                self._json(200, {"object": "list", "data": [card]})
            elif path == "/version":
                self._json(200, {"version": "mock"})
            elif path == "/server_info":
                self._json(200, server_info(engine))
            elif self.path == "/metrics":
                body = engine.metrics(model).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/tokenize":
                text = (
                    chat_template(req["messages"], req.get("add_generation_prompt", True))
                    if "messages" in req
                    else req.get("prompt", "")
                )
                ids = vocab.encode(text)
                self._json(200, {"count": len(ids), "max_model_len": 8192, "tokens": ids})
            elif self.path == "/detokenize":
                self._json(200, {"prompt": vocab.decode(req.get("tokens", []))})
            elif self.path == "/v1/chat/completions":
                self._chat(req)
            else:
                self._json(404, {"error": "not found"})

        def _chat(self, req: dict[str, Any]) -> None:
            if not req.get("stream"):
                self._json(400, {"error": "the mock supports stream=true only"})
                return
            prompt = vocab.encode(chat_template(req["messages"]))
            max_tokens = int(req.get("max_tokens") or 64)
            seed = sum(prompt) + len(prompt)
            n = min(max_tokens, 24 + seed % 72)
            off = seed % len(corpus)
            seq = Seq(
                prompt,
                [corpus[(off + j) % len(corpus)] for j in range(n)],
                "length" if n == max_tokens else "stop",
            )
            engine.submit(seq)

            as_ids = bool(req.get("return_tokens_as_token_ids"))
            rid, created = f"chatcmpl-mock-{id(seq):x}", int(time.time())

            def chunk(choice: dict[str, Any] | None, usage: dict[str, Any] | None = None) -> None:
                obj = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [choice] if choice else [],
                }
                if usage:
                    obj["usage"] = usage
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()  # HTTP/1.0: the stream ends when the connection closes
            chunk(
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            )
            while True:
                kind, val = seq.events.get()
                if kind == "done":
                    chunk({"index": 0, "delta": {}, "finish_reason": val})
                    break
                piece = vocab.pieces[val]
                lp = None
                if req.get("logprobs"):
                    lp = {
                        "content": [
                            {
                                "token": f"token_id:{val}" if as_ids else piece,
                                "logprob": -round((val * 7919 % 400) / 100, 4),
                                "bytes": list(piece.encode()),
                                "top_logprobs": [],
                            }
                        ]
                    }
                chunk(
                    {"index": 0, "delta": {"content": piece}, "logprobs": lp, "finish_reason": None}
                )
            if (req.get("stream_options") or {}).get("include_usage"):
                chunk(
                    None,
                    {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": len(seq.output),
                        "total_tokens": len(prompt) + len(seq.output),
                        "prompt_tokens_details": {"cached_tokens": len(seq.hits) * BLOCK},
                    },
                )
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256  # the default 5 refuses connections under a concurrent batch


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    model: str = MODEL,
    num_blocks: int = 96,
    max_num_seqs: int = 4,
    step_ms: float = 20.0,
    prefill_budget: int = 64,
) -> ThreadingHTTPServer:
    """Start the engine and an HTTP server on a background thread; port 0 picks a free one"""
    engine = Engine(num_blocks, max_num_seqs, step_ms / 1000, prefill_budget)
    engine.start()
    srv = _Server((host, port), make_handler(engine, Vocab(), model))
    threading.Thread(target=srv.serve_forever, daemon=True, name="mock-http").start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-blocks", type=int, default=96)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--step-ms", type=float, default=20.0)
    a = ap.parse_args()
    srv = serve(
        a.host, a.port, num_blocks=a.num_blocks, max_num_seqs=a.max_num_seqs, step_ms=a.step_ms
    )
    print(f"mock vLLM on http://{a.host}:{srv.server_address[1]} (Ctrl-C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
