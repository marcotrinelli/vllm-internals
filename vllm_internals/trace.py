"""Per-request trace of a run: timings of every streamed token plus the token content.

A `RequestTrace` is filled chunk by chunk from the OpenAI-compatible streaming response
(`on_chunk`), so the parsing is testable without a server. `build_trace` turns a list of
them into the `vllm-traces/1` document the viewer in `app/` reads (see
`docs/schemas.md`).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "vllm-traces/1"


@dataclass
class Clock:
    """t=0 of a run: a monotonic origin for the timings and its wall-clock twin.

    The wall-clock value is what lets the viewer line a trace up with a metrics
    recording started at a different instant.
    """

    t0: float = field(default_factory=time.perf_counter)
    t0_unix: float = field(default_factory=time.time)

    def now(self) -> float:
        return time.perf_counter() - self.t0


def _token_id(token: str) -> int | None:
    # note that vLLM spells ids as 'token_id:1234' when `return_tokens_as_token_ids` is set
    if token.startswith("token_id:"):
        try:
            return int(token[len("token_id:") :])
        except ValueError:
            return None
    return None


@dataclass
class RequestTrace:
    idx: int
    messages: list[dict[str, Any]]
    t_submit: float  # seconds since the run's t0, for every time field below
    t_first: float | None = None
    t_end: float | None = None
    token_t: list[float] = field(default_factory=list)  # arrival time of each output token
    token_text: list[str] = field(default_factory=list)  # decoded text of each, same order
    output_ids: list[int | None] | None = field(default_factory=list)
    token_logprob: list[float | None] | None = field(default_factory=list)
    prompt_ids: list[int] | None = None  # from /tokenize, chat template applied
    prompt_tokens: int | None = None  # server-reported, from the final usage chunk
    completion_tokens: int | None = None
    cached_tokens: int | None = None  # needs `--enable-prompt-tokens-details` on the server
    finish_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.token_t)

    def on_chunk(self, chunk: dict[str, Any], t: float) -> None:
        """Fold one streamed `chat.completion.chunk` (already JSON-decoded) into the trace."""
        usage = chunk.get("usage")
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens")
            self.completion_tokens = usage.get("completion_tokens")
            self.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        choices = chunk.get("choices") or []
        if not choices:
            return  # the usage-only chunk has an empty `choices` list
        c = choices[0]
        if c.get("finish_reason"):
            self.finish_reason = c["finish_reason"]

        delta = c.get("delta") or {}
        text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
        entries = (c.get("logprobs") or {}).get("content") or []
        if entries:
            # One entry per sampled token. A chunk can carry several (speculative decoding,
            # or a detokenizer holding back a partial UTF-8 sequence), and they share the
            # chunk's arrival time
            for e in entries:
                raw = e.get("bytes")
                tok = e.get("token") or ""
                self.token_t.append(t)
                self.token_text.append(
                    bytes(raw).decode("utf-8", "replace") if raw is not None else tok
                )
                self.output_ids.append(_token_id(tok))
                self.token_logprob.append(e.get("logprob"))
        elif text:
            # No logprobs (not requested, or not a vLLM server): one chunk ~ one token
            self.token_t.append(t)
            self.token_text.append(text)
            self.output_ids.append(None)
            self.token_logprob.append(None)
        else:
            return  # role-only / empty chunk: no token, must not move the clock
        if self.t_first is None:
            self.t_first = t

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        r4 = lambda v: None if v is None else round(v, 4)  # noqa: E731
        for k in ("t_submit", "t_first", "t_end"):
            d[k] = r4(d[k])
        d["token_t"] = [round(v, 4) for v in self.token_t]
        # a list of nulls says nothing a null does not, and costs a line per token
        if not any(v is not None for v in self.output_ids or []):
            d["output_ids"] = None
        if not any(v is not None for v in self.token_logprob or []):
            d["token_logprob"] = None
        else:
            d["token_logprob"] = [r4(v) for v in self.token_logprob]
        d["ok"] = self.ok
        return d


def build_trace(
    requests: list[RequestTrace],
    *,
    clock: Clock,
    model: str,
    vocab: dict[int, str] | None = None,
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The `vllm-traces/1` document for a list of finished requests."""
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "t0_unix": round(clock.t0_unix, 6),
        "model": model,
        "run": run or {},
        "requests": [r.to_dict() for r in sorted(requests, key=lambda r: r.idx)],
        "vocab": {str(k): v for k, v in sorted((vocab or {}).items())},
    }


def save_json(doc: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    return p


def quantile(values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default), as the viewer does"""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(trace: dict[str, Any]) -> dict[str, Any]:
    """Latency and throughput of a `vllm-traces/1` document.

    Computed from the document rather than from the in-memory requests, so the numbers
    are the ones the viewer shows for the same file (same rounding, same definitions).
    """
    reqs = trace["requests"]
    ok = [r for r in reqs if r["ok"]]
    ttft = [r["t_first"] - r["t_submit"] for r in ok]
    e2e = [r["t_end"] - r["t_submit"] for r in ok]
    itl = [b - a for r in ok for a, b in zip(r["token_t"], r["token_t"][1:])]
    tpot = [
        (r["token_t"][-1] - r["token_t"][0]) / (len(r["token_t"]) - 1)
        for r in ok
        if len(r["token_t"]) > 1
    ]
    out = sum(r["completion_tokens"] or len(r["token_t"]) for r in ok)
    wall = (max(r["t_end"] for r in ok) - min(r["t_submit"] for r in ok)) if ok else 0.0
    return {
        "n": len(reqs),
        "errors": len(reqs) - len(ok),
        "wall_s": wall,
        "out_tokens": out,
        "out_tok_per_s": out / wall if wall else None,
        "req_per_s": len(ok) / wall if wall else None,
        **{f"ttft_p{p}": quantile(ttft, p / 100) for p in (50, 95, 99)},
        **{f"itl_p{p}": quantile(itl, p / 100) for p in (50, 95, 99)},
        **{f"tpot_p{p}": quantile(tpot, p / 100) for p in (50, 95)},
        **{f"e2e_p{p}": quantile(e2e, p / 100) for p in (50, 95)},
    }
