"""The client and the recorder against the mock server, writing both files as the notebook does"""

import asyncio
import json

import pytest
from mock_vllm import serve

from vllm_internals import Clock, Config, Recorder, VLLMClient, build_trace, run_batch, save_json
from vllm_internals.recorder import SCHEMA as METRICS_SCHEMA
from vllm_internals.trace import SCHEMA as TRACE_SCHEMA


@pytest.fixture(scope="module")
def base_url():
    srv = serve(port=0, max_num_seqs=2, step_ms=5)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_config_strips_v1(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://h:8000/v1/")
    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    cfg = Config.from_env()
    assert (cfg.base_url, cfg.model, cfg.metrics_url) == (
        "http://h:8000",
        None,
        "http://h:8000/metrics",
    )
    assert cfg.headers == {"Authorization": "Bearer k"}


def test_batch_writes_matching_trace_and_metrics(base_url, tmp_path):
    system = {"role": "system", "content": "Shared system prompt, long enough to fill blocks. " * 4}
    convs = [[system, {"role": "user", "content": f"question {i}"}] for i in range(4)]

    async def go():
        async with VLLMClient(Config(base_url)) as client:
            with Recorder(Config(base_url).metrics_url, interval_s=0.05) as rec:
                clock = Clock()
                reqs = await run_batch(client, convs, concurrency=4, max_tokens=16, clock=clock)
            vocab = await client.attach_prompt_tokens(reqs)
            return client.model, rec, clock, reqs, vocab

    model, rec, clock, reqs, vocab = asyncio.run(go())
    assert model == "mock/tiny-chat"
    assert all(r.ok for r in reqs), [r.error for r in reqs]
    trace = build_trace(reqs, clock=clock, model=model, vocab=vocab)
    tp, mp = save_json(trace, tmp_path / "t.json"), rec.save(tmp_path / "m.json")
    t, m = json.loads(tp.read_text()), json.loads(mp.read_text())

    assert t["schema"] == TRACE_SCHEMA and m["schema"] == METRICS_SCHEMA
    # read once when the recording stops; the mock answers as a dev-mode server
    assert m["server"]["version"] == "mock"
    assert m["server"]["models"] == [
        {"id": "mock/tiny-chat", "max_model_len": 4096, "parent": None}
    ]
    assert (m["server"]["config"]["tp"], m["server"]["config"]["max_num_seqs"]) == (1, 2)
    # the recorder starts first, so the trace's t=0 is a little later on the wall clock
    assert 0 <= t["t0_unix"] - m["t0_unix"] < 1
    for r in t["requests"]:
        assert r["t_submit"] <= r["t_first"] <= r["token_t"][-1] <= r["t_end"]
        assert (
            len(r["token_t"])
            == len(r["token_text"])
            == len(r["output_ids"])
            == r["completion_tokens"]
        )
        assert len(r["prompt_ids"]) == r["prompt_tokens"]  # /tokenize agrees with the usage block
        assert all(str(i) in t["vocab"] for i in r["prompt_ids"])
    # prompt ids decode back to the chat template around the messages
    text = "".join(t["vocab"][str(i)] for i in t["requests"][0]["prompt_ids"])
    assert "question 0" in text and text.endswith("assistant\n")
    # four requests, two slots: the later ones queue, and hit the shared prefix
    assert max(s["m"].get("num_requests_waiting", 0) for s in m["samples"]) > 0
    assert any(r["cached_tokens"] for r in t["requests"])
    assert m["info"]["cache_config_info"]["block_size"] == "16"


def test_failed_request_is_recorded(base_url):
    async def go():
        async with VLLMClient(Config("http://127.0.0.1:9", model="m")) as client:
            return await client.chat([{"role": "user", "content": "x"}])

    r = asyncio.run(go())
    assert not r.ok and r.error.startswith("ConnectError")
    assert r.to_dict()["ok"] is False
