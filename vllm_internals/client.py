"""Minimal async client for a vLLM OpenAI-compatible server.

Only what the notebook needs: model discovery, streamed chat completions recorded as
`RequestTrace`s, and `/tokenize` + `/detokenize` for the prompt's token content.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv

from .trace import Clock, RequestTrace


@dataclass
class Config:
    base_url: str = "http://localhost:8000"
    model: str | None = None  # None -> the first model `/v1/models` lists
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> Config:
        """`VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`, from the environment or a `.env`"""
        # usecwd, to find the repo's `.env` from `notebooks/` (walks up the parents)
        load_dotenv(find_dotenv(usecwd=True))
        base = os.getenv("VLLM_BASE_URL") or cls.base_url
        # `/tokenize` and `/metrics` live at the root, so accept the OpenAI-style `.../v1` too
        base = base.rstrip("/").removesuffix("/v1")
        return cls(base, os.getenv("VLLM_MODEL") or None, os.getenv("VLLM_API_KEY") or None)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @property
    def metrics_url(self) -> str:
        return self.base_url + "/metrics"


class VLLMClient:
    def __init__(self, cfg: Config, timeout_s: float = 600.0):
        self.cfg = cfg
        self.model = cfg.model
        self.http = httpx.AsyncClient(
            base_url=cfg.base_url,
            headers=cfg.headers,
            timeout=timeout_s,
            # one connection per in-flight request, so concurrency is the server's, not ours
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def __aenter__(self) -> VLLMClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def resolve_model(self) -> str:
        if not self.model:
            r = await self.http.get("/v1/models")
            r.raise_for_status()
            ids = [m["id"] for m in r.json()["data"]]
            if not ids:
                raise RuntimeError("the server lists no models on /v1/models")
            self.model = ids[0]
        return self.model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 128,
        *,
        clock: Clock | None = None,
        idx: int = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> RequestTrace:
        """One streamed chat completion, keeping every token's arrival time and content.

        Transport and HTTP errors are recorded on the trace (`error`) rather than raised,
        so one failed request does not lose the rest of a batch.
        """
        clock = clock or Clock()
        model = await self.resolve_model()
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            # vLLM only emits the final usage chunk if asked for it
            "stream_options": {"include_usage": True},
            # one logprobs entry per sampled token: id (vLLM extension) + bytes of its text
            "logprobs": True,
            "return_tokens_as_token_ids": True,
            **(extra_body or {}),
        }
        # t_submit must be taken BEFORE the request is issued, otherwise TTFT hides the
        # connection + prefill time it is meant to measure
        req = RequestTrace(idx=idx, messages=messages, t_submit=clock.now())
        try:
            async with self.http.stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    req.error = f"HTTP {resp.status_code}: {text[:300]}"
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        req.on_chunk(json.loads(data), clock.now())
        except httpx.HTTPError as e:
            req.error = f"{type(e).__name__}: {e}"
        req.t_end = clock.now()
        if req.error is None and not req.token_t:
            req.error = "no token streamed back"
        return req

    async def tokenize(
        self, messages: list[dict[str, Any]], extra_body: dict[str, Any] | None = None
    ) -> list[int]:
        """Prompt token ids exactly as the server builds them (chat template applied)"""
        body = {
            "model": await self.resolve_model(),
            "messages": messages,
            "add_generation_prompt": True,
            **{k: v for k, v in (extra_body or {}).items() if k == "chat_template_kwargs"},
        }
        r = await self.http.post("/tokenize", json=body)
        r.raise_for_status()
        return r.json()["tokens"]

    async def detokenize(self, ids: list[int]) -> str:
        r = await self.http.post(
            "/detokenize", json={"model": await self.resolve_model(), "tokens": ids}
        )
        r.raise_for_status()
        return r.json()["prompt"]

    async def attach_prompt_tokens(
        self,
        reqs: list[RequestTrace],
        extra_body: dict[str, Any] | None = None,
        concurrency: int = 16,
    ) -> dict[int, str]:
        """Fill `prompt_ids` on every request and return the vocab {id: decoded text}.

        Run after the timed load, so tokenization does not compete with it. Each distinct
        id is decoded on its own (`/detokenize` of one id), which is what a token looks
        like in isolation: a byte-level piece of a multi-byte character shows as U+FFFD.
        """
        by_prompt: dict[str, list[int]] = {}
        for r in reqs:
            key = json.dumps(r.messages, sort_keys=True)
            if key not in by_prompt:
                by_prompt[key] = await self.tokenize(r.messages, extra_body)
            r.prompt_ids = by_prompt[key]

        sem = asyncio.Semaphore(concurrency)
        distinct = sorted({i for r in reqs for i in r.prompt_ids or []})

        async def one(i: int) -> tuple[int, str]:
            async with sem:
                return i, await self.detokenize([i])

        return dict(await asyncio.gather(*(one(i) for i in distinct)))


async def run_batch(
    client: VLLMClient,
    conversations: list[list[dict[str, Any]]],
    *,
    concurrency: int,
    max_tokens: int,
    clock: Clock,
    extra_body: dict[str, Any] | None = None,
    first_idx: int = 0,
) -> list[RequestTrace]:
    sem = asyncio.Semaphore(concurrency)

    # The semaphore is what makes t_submit meaningful: a request holding it has already
    # been handed to the server, so queue time in the trace is the engine's waiting
    # queue, not the client's backlog
    async def bounded(i: int, messages: list[dict[str, Any]]) -> RequestTrace:
        async with sem:
            return await client.chat(
                messages, max_tokens, clock=clock, idx=i, extra_body=extra_body
            )

    return list(
        await asyncio.gather(*(bounded(i, m) for i, m in enumerate(conversations, start=first_idx)))
    )
