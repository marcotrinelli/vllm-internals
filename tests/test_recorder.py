import math

from vllm_internals.recorder import SCHEMA, Recorder, parse_prom, server_config, split_auth

PROM = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 5.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:generation_tokens_created{engine="0",model_name="m"} 1.7e9
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 1.0
vllm:iteration_tokens_total_bucket{engine="0",le="8.0",model_name="m"} 4.0
vllm:iteration_tokens_total_bucket{engine="0",le="+Inf",model_name="m"} 6.0
vllm:iteration_tokens_total_sum{engine="0",model_name="m"} 40.0
vllm:iteration_tokens_total_count{engine="0",model_name="m"} 6.0
vllm:cache_config_info{block_size="16",enable_prefix_caching="True",num_gpu_blocks="96"} 1.0
vllm:weird{engine="0"} NaN
process_open_fds 12.0
"""


def test_parse_prom_splits_gauges_counters_histograms_and_info():
    s = parse_prom(PROM)
    assert s["m"]["num_requests_running"] == 3.0
    assert s["m"]["process_open_fds"] == 12.0  # non-vllm series kept as is
    # a labelled counter is folded into its total and kept split by the non-identity labels
    assert s["m"]["request_success_total"] == 7.0
    assert s["l"]["request_success_total"] == {
        "finished_reason=stop": 5.0,
        "finished_reason=length": 2.0,
    }
    assert "num_requests_running" not in s["l"]  # identity labels only: no split
    assert s["h"]["iteration_tokens_total"] == {"1.0": 1.0, "8.0": 4.0, "+Inf": 6.0}
    assert s["m"]["iteration_tokens_total_count"] == 6.0
    assert s["info"]["cache_config_info"]["num_gpu_blocks"] == "96"
    # creation timestamps and non-finite values are not data
    assert "generation_tokens_created" not in s["m"]
    assert "weird" not in s["m"]


def test_split_auth_keeps_credentials_out_of_the_url():
    url, auth = split_auth("http://user:p%40ss@[::1]:8000/metrics")
    assert url == "http://[::1]:8000/metrics"
    assert auth == "Basic dXNlcjpwQHNz"  # base64("user:p@ss")
    assert split_auth("http://localhost:8000/metrics") == ("http://localhost:8000/metrics", None)


def test_to_dict_aligns_buckets_across_samples(monkeypatch):
    texts = iter([PROM.replace('le="8.0"', 'le="4.0"'), PROM, PROM])
    rec = Recorder("http://user:pw@localhost:9/metrics", interval_s=60)
    monkeypatch.setattr(rec, "_fetch", lambda: next(texts))
    rec.start()
    rec.stop()
    d = rec.to_dict()
    assert d["schema"] == SCHEMA
    assert d["source"] == "http://localhost:9/metrics"
    assert rec.headers["Authorization"].startswith("Basic ")
    assert d["t0_unix"] > 0 and d["scrapes"] == len(d["samples"]) >= 2
    # bucket edges are the union over every sample, +Inf written as null
    assert d["buckets"]["iteration_tokens_total"] == [1.0, 4.0, 8.0, None]
    assert d["samples"][0]["h"]["iteration_tokens_total"] == [1, 4, 0, 6]
    assert d["samples"][1]["h"]["iteration_tokens_total"] == [1, 0, 4, 6]
    assert d["info"]["cache_config_info"]["block_size"] == "16"
    assert all(not math.isnan(v) for s in d["samples"] for v in s["m"].values())


def test_dead_endpoint_is_reported_not_swallowed():
    rec = Recorder("http://127.0.0.1:9/metrics", interval_s=60, timeout_s=0.5).start().stop()
    assert rec.errors >= 1 and rec.last_error
    assert "no samples" in rec.describe()
    try:
        rec.to_dict()
    except RuntimeError as e:
        assert "no samples" in str(e)
    else:
        raise AssertionError("to_dict must refuse an empty recording")


def test_parse_prom_keeps_each_engine_and_averages_fractions():
    s = parse_prom(
        'vllm:num_requests_running{engine="0",model_name="m"} 3\n'
        'vllm:num_requests_running{engine="1",model_name="m"} 5\n'
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.2\n'
        'vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.4\n'
    )
    assert s["m"]["num_requests_running"] == 8
    # summed it would read 60%: the pool-wide figure is the mean of the cores
    assert math.isclose(s["m"]["kv_cache_usage_perc"], 0.3)
    assert s["e"]["1"] == {"num_requests_running": 5, "kv_cache_usage_perc": 0.4}


def test_server_config_keeps_the_whitelist_only():
    c = server_config(
        {
            "vllm_config": {
                "parallel_config": {"tensor_parallel_size": 2, "data_parallel_size": 2},
                "model_config": {"dtype": "torch.bfloat16", "max_model_len": 8192},
                "scheduler_config": {"max_num_seqs": 128},
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
            },
            "vllm_env": {"VLLM_SOME_TOKEN": "secret"},
            "system_env": {
                "nvidia_gpu_models": "GPU 0: NVIDIA H200\nGPU 1: NVIDIA H200",
                "pip_packages": "...",
            },
        }
    )
    assert c is not None
    assert (c["tp"], c["pp"], c["dp"], c["dtype"]) == (2, 1, 2, "bfloat16")
    assert (c["max_num_seqs"], c["spec_method"], c["spec_tokens"]) == (128, "mtp", 2)
    assert c["gpus"] == ["NVIDIA H200", "NVIDIA H200"]
    assert "secret" not in repr(c) and "pip" not in repr(c)
    assert server_config({"detail": "Not Found"}) is None
