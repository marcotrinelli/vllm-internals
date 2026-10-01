import json
from pathlib import Path

import pytest

from vllm_internals.trace import SCHEMA, Clock, RequestTrace, build_trace, quantile, summary

EXAMPLES = Path(__file__).parent.parent / "examples"


def lp(tid: int, text: str, logprob: float = -0.5) -> dict:
    return {
        "token": f"token_id:{tid}",
        "logprob": logprob,
        "bytes": list(text.encode()),
        "top_logprobs": [],
    }


def chunk(content=None, logprobs=None, finish=None, usage=None, choices=True) -> dict:
    c = {
        "index": 0,
        "delta": {"content": content} if content is not None else {},
        "finish_reason": finish,
    }
    if logprobs is not None:
        c["logprobs"] = {"content": logprobs}
    out = {"choices": [c] if choices else []}
    if usage:
        out["usage"] = usage
    return out


def test_on_chunk_keeps_ids_text_logprobs_and_times():
    r = RequestTrace(idx=0, messages=[], t_submit=0.0)
    r.on_chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}, 0.05)
    r.on_chunk(chunk("Hi", [lp(10, "Hi")]), 0.10)
    # two tokens in one chunk share its arrival time
    r.on_chunk(chunk(" th", [lp(11, " the", -1.25), lp(12, "ré")]), 0.12)
    r.on_chunk(chunk(finish="stop"), 0.13)
    usage = {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    r.on_chunk(chunk(choices=False, usage=usage), 0.14)
    assert r.t_first == 0.10  # the role-only chunk does not move the clock
    assert r.token_t == [0.10, 0.12, 0.12]
    assert r.token_text == ["Hi", " the", "ré"]
    assert r.output_ids == [10, 11, 12]
    assert r.token_logprob == [-0.5, -1.25, -0.5]
    assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (9, 3, 8)
    assert r.finish_reason == "stop" and r.ok


def test_on_chunk_without_logprobs_counts_one_token_per_chunk():
    r = RequestTrace(idx=1, messages=[], t_submit=0.0)
    r.on_chunk(chunk("Hello"), 0.2)
    r.on_chunk({"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]}, 0.3)
    d = r.to_dict()
    assert d["token_text"] == ["Hello", "hmm"]
    assert d["output_ids"] is None and d["token_logprob"] is None
    assert d["cached_tokens"] is None  # server started without --enable-prompt-tokens-details


def test_build_trace_is_json_and_versioned():
    clock = Clock()
    r = RequestTrace(idx=0, messages=[{"role": "user", "content": "x"}], t_submit=0.012345678)
    r.on_chunk(chunk("a", [lp(1, "a")]), 0.5)
    r.t_end, r.prompt_ids = 0.6, [5, 6]
    failed = RequestTrace(idx=1, messages=[], t_submit=0.0, t_end=0.1, error="HTTP 500: boom")
    doc = json.loads(
        json.dumps(build_trace([failed, r], clock=clock, model="m", vocab={6: "b", 5: "a"}))
    )
    assert doc["schema"] == SCHEMA == "vllm-traces/1"
    assert doc["t0_unix"] == pytest.approx(clock.t0_unix, abs=1e-5)
    assert [q["idx"] for q in doc["requests"]] == [0, 1]
    assert doc["requests"][0]["t_submit"] == 0.0123  # rounded to 0.1 ms
    assert doc["requests"][1] | {"ok": False, "error": "HTTP 500: boom"} == doc["requests"][1]
    assert doc["vocab"] == {"5": "a", "6": "b"}


def test_quantile_matches_numpy_linear():
    assert quantile([], 0.5) is None
    assert quantile([3.0], 0.95) == 3.0
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert quantile([4.0, 1.0, 3.0, 2.0], 0.95) == pytest.approx(3.85)


def test_summary_of_the_example_matches_the_file_the_viewer_checks():
    # the viewer's vitest suite checks its own summary against the same file
    trace = json.loads((EXAMPLES / "batch.trace.json").read_text())
    expected = json.loads((EXAMPLES / "batch.summary.json").read_text())
    assert summary(trace) == pytest.approx(expected)
