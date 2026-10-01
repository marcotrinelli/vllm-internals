"""Record a vLLM server's Prometheus endpoint as a replayable time series.

Everything written here comes from `GET /metrics`. There is no client-side knowledge of
the workload, so a recording works against any vLLM server no matter who is sending it
traffic. Inside a notebook, record around load you drive yourself (the scrape runs on its
own thread, so it does not block the event loop):

    with Recorder("http://localhost:8000/metrics") as rec:
        await run_batch(...)
    rec.save("runs/metrics.json")

Output schema `vllm-metrics/1` (see `docs/schemas.md`):

    {"schema", "source", "recorded_at", "t0_unix", "interval_s", "duration_s", "scrapes", "errors",
     "info":    {"cache_config_info": {...labels...}, ...},
     "server":  {"version", "models": [...], "config": {...} | null},
     "buckets": {"<histogram>": [upper bounds, null = +Inf], ...},
     "samples": [{"t": seconds since start, mid-scrape,
                  "rtt": seconds the scrape took,
                  "m": {"<metric>": value},               # gauges, counters, _sum, _count
                  "h": {"<histogram>": [cumulative counts aligned to buckets]},
                  "l": {"<metric>": {"<label>=<value>": value}},
                  "e": {"<engine>": {"<metric>": value}}}, ...]}   # data parallel only

Metric names have the `vllm:` prefix stripped; everything else is left as the server
reported it, so new metrics show up in the recording (and in the viewer) for free.
`server` is read once, when the recording stops, from `/version`, `/v1/models` and
`/server_info` (served only with `VLLM_SERVER_DEV_MODE=1`); only the whitelisted fields of
`server_config` are kept, never the server's environment.
"""

from __future__ import annotations

import base64
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "vllm-metrics/1"
DEFAULT_URL = "http://localhost:8000/metrics"

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?[ \t]+([^ \t]+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# These identify the server, not a slice of the metric; splitting on them would turn
# every counter into a one-entry map
_IDENTITY_LABELS = frozenset({"model_name", "engine", "engine_index", "instance", "job"})


def _num(text: str) -> float | None:
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _edge(le: str) -> float:
    return math.inf if le.lstrip("+") in ("Inf", "inf") else float(le)


def parse_prom(text: str) -> dict[str, Any]:
    """One scrape of Prometheus exposition text -> plain dicts.

    Nothing is hard-coded to a metric name: buckets are recognised by the `_bucket`
    suffix and `le` label, config blobs by the `_info` suffix, and any remaining label
    is kept as a split of the metric as well as folded into its total.
    """
    m: dict[str, float] = {}
    h: dict[str, dict[str, float]] = {}
    split_by: dict[str, dict[str, float]] = {}
    info: dict[str, dict[str, str]] = {}
    engines: dict[str, dict[str, float]] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        hit = _SAMPLE.match(line)
        if not hit:
            continue
        name, label_text, value = hit.group(1), hit.group(2) or "", hit.group(3)
        labels = dict(_LABEL.findall(label_text))
        short = name.split(":", 1)[1] if name.startswith("vllm:") else name

        if short.endswith("_info"):
            info[short] = labels  # the value is a constant 1; the labels are the payload
            continue
        if short.endswith("_created"):
            continue  # unix timestamp of a counter's creation, not data
        v = _num(value)
        if v is None:
            continue
        if short.endswith("_bucket") and "le" in labels:
            b = h.setdefault(short[: -len("_bucket")], {})
            b[labels["le"]] = b.get(labels["le"], 0.0) + v
            continue

        m[short] = m.get(short, 0.0) + v
        if "engine" in labels:
            per = engines.setdefault(labels["engine"], {})
            per[short] = per.get(short, 0.0) + v
        split = {k: v2 for k, v2 in labels.items() if k not in _IDENTITY_LABELS}
        if split:
            key = ",".join(f"{k}={split[k]}" for k in sorted(split))
            d = split_by.setdefault(short, {})
            d[key] = d.get(key, 0.0) + v
    # A fraction summed over data-parallel engine cores reads past 100%: the pool-wide figure
    # is their mean
    if len(engines) > 1:
        for k in m:
            if k.endswith("_perc"):
                m[k] /= len(engines)
    return {"m": m, "h": h, "l": split_by, "info": info, "e": engines}


def _round(v: float) -> float:
    return int(v) if v == int(v) and abs(v) < 2**53 else round(v, 6)


def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode("ascii")


def split_auth(url: str) -> tuple[str, str | None]:
    """Take `user:pass@` out of a URL and return it as an Authorization header value.

    urllib does not send userinfo on its own, and it must not reach the recording's
    `source` field either.
    """
    p = urllib.parse.urlsplit(url)
    if not p.username and not p.password:
        return url, None
    host = p.netloc.rsplit("@", 1)[-1]  # keeps brackets of an IPv6 literal
    clean = urllib.parse.urlunsplit((p.scheme, host, p.path, p.query, p.fragment))
    return clean, _basic(
        urllib.parse.unquote(p.username or ""), urllib.parse.unquote(p.password or "")
    )


def _get(d: Any, *path: str) -> Any:
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def server_config(info: dict[str, Any]) -> dict[str, Any] | None:
    """The parts of a `/server_info?config_format=json` body the viewer shows, flattened.

    A whitelist on purpose: the body also carries the server's environment variables and
    package list, which have no place in a file that gets shared.
    """
    c = info.get("vllm_config")
    if not isinstance(c, dict):
        return None
    p, sch, mod, cache = (
        c.get(k) or {}
        for k in ("parallel_config", "scheduler_config", "model_config", "cache_config")
    )
    spec, lora, kv = c.get("speculative_config"), c.get("lora_config"), c.get("kv_transfer_config")
    gpus = str(_get(info, "system_env", "nvidia_gpu_models") or "")
    dtype = mod.get("dtype")
    return {
        "tp": p.get("tensor_parallel_size", 1),
        "pp": p.get("pipeline_parallel_size", 1),
        "dp": p.get("data_parallel_size", 1),
        "dp_local": p.get("data_parallel_size_local", p.get("data_parallel_size", 1)),
        "ep": bool(p.get("enable_expert_parallel")),
        "nnodes": p.get("nnodes", 1),
        "executor": p.get("distributed_executor_backend"),
        # torch dtypes serialise as `torch.bfloat16`
        "dtype": str(dtype).removeprefix("torch.") if dtype is not None else None,
        "quantization": mod.get("quantization"),
        "kv_dtype": cache.get("cache_dtype"),
        "max_model_len": mod.get("max_model_len"),
        "max_num_seqs": sch.get("max_num_seqs"),
        "max_num_batched_tokens": sch.get("max_num_batched_tokens"),
        "chunked_prefill": sch.get("enable_chunked_prefill"),
        "async_scheduling": sch.get("async_scheduling"),
        "enforce_eager": mod.get("enforce_eager"),
        "spec_method": _get(spec, "method"),
        "spec_tokens": _get(spec, "num_speculative_tokens"),
        "max_loras": _get(lora, "max_loras"),
        "kv_connector": _get(kv, "kv_connector"),
        "kv_role": _get(kv, "kv_role"),
        "gpus": [re.sub(r"^GPU \d+:\s*", "", g).strip() for g in gpus.splitlines() if g.strip()],
    }


class Recorder:
    """Scrapes `url` every `interval_s` on a background thread until stopped.

    Credentials inline in the URL (`http://user:pass@host:8000/metrics`) are sent as an
    HTTP basic Authorization header; `headers` (e.g. a bearer token) take precedence.
    Neither is written to the recording.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        interval_s: float = 0.25,
        timeout_s: float = 5.0,
        headers: dict[str, str] | None = None,
    ):
        self.url, auth = split_auth(url)
        self.headers = {"Authorization": auth} if auth else {}
        self.headers.update(headers or {})
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.samples: list[dict[str, Any]] = []
        self.info: dict[str, dict[str, str]] = {}
        self.server: dict[str, Any] = {"version": None, "models": [], "config": None}
        self.errors = 0
        self.last_error: str | None = None
        self.t0 = 0.0
        self.t0_unix = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- scraping ------------------------------------------------------------
    def scrape(self) -> dict[str, Any]:
        return parse_prom(self._fetch())

    def _fetch(self) -> str:
        req = urllib.request.Request(self.url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return r.read().decode("utf-8", "replace")

    def _json(self, path: str) -> Any:
        """GET a JSON endpoint next to /metrics. None when it is missing or fails: all of
        these are optional, and a recording is complete without them"""
        root = re.sub(r"/metrics/?$", "", self.url)
        req = urllib.request.Request(root + path, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def read_server(self) -> dict[str, Any]:
        models = _get(self._json("/v1/models"), "data")
        info = self._json("/server_info?config_format=json")
        self.server = {
            "version": _get(self._json("/version"), "version"),
            "models": [
                {
                    "id": m.get("id"),
                    "max_model_len": m.get("max_model_len"),
                    "parent": m.get("parent"),
                }
                for m in (models if isinstance(models, list) else [])
                if isinstance(m, dict)
            ],
            "config": server_config(info) if isinstance(info, dict) else None,
        }
        return self.server

    def _tick(self) -> None:
        sent = time.perf_counter()
        try:
            text = self._fetch()
        except (urllib.error.URLError, OSError, ValueError) as e:
            # counted and reported by `to_dict`/`describe`, so a dead endpoint is visible
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return
        got = time.perf_counter()
        s = parse_prom(text)
        # The server read its counters somewhere inside the round trip, which on a busy or
        # remote server can take most of a second: the midpoint is the best guess, and off by
        # at most half the round trip, written alongside so the viewer can say so
        s["t"] = (sent + got) / 2 - self.t0
        s["rtt"] = got - sent
        self.info.update(s.pop("info"))
        self.samples.append(s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            tick = time.perf_counter()
            self._tick()
            # Sleep the remainder of the interval, so the cadence does not drift by the
            # scrape cost (which grows with the number of series the server exports)
            self._stop.wait(max(0.0, self.interval_s - (time.perf_counter() - tick)))

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> Recorder:
        if self._thread:
            raise RuntimeError("recorder already started")
        self.t0 = time.perf_counter()
        # wall-clock t=0, so a trace recorded by the client can be placed on this timeline
        self.t0_unix = time.time()
        self._stop.clear()
        self._tick()  # t=0 baseline, so counters have a zero point
        self._thread = threading.Thread(target=self._loop, name="vllm-metrics", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Recorder:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=self.timeout_s + self.interval_s + 1.0)
            self._thread = None
            self._tick()  # final sample, so the last window is complete
            # after the run, so a slow /server_info (it collects the environment) never
            # delays a scrape
            self.read_server()
        return self

    def __enter__(self) -> Recorder:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- output --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        if not self.samples:
            raise RuntimeError(f"no samples: {self.last_error or 'is the server up?'}")

        # Bucket edges are collected across every sample rather than taken from the
        # first: a histogram with no observations yet may not be exported at all
        bases = sorted({b for s in self.samples for b in s["h"]})
        edges = {
            b: sorted({_edge(le) for s in self.samples for le in s["h"].get(b, ())}) for b in bases
        }

        out_samples = []
        for s in self.samples:
            h = {}
            for b in bases:
                got = s["h"].get(b)
                if not got:
                    continue
                by_edge = {_edge(le): v for le, v in got.items()}
                h[b] = [_round(by_edge.get(e, 0.0)) for e in edges[b]]
            row: dict[str, Any] = {
                "t": round(s["t"], 4),
                "rtt": round(s["rtt"], 4),
                "m": {k: _round(v) for k, v in s["m"].items()},
            }
            if h:
                row["h"] = h
            if s["l"]:
                row["l"] = {k: {kk: _round(vv) for kk, vv in d.items()} for k, d in s["l"].items()}
            # per engine only with data parallel; a single core's values are the totals
            if len(s["e"]) > 1:
                row["e"] = {k: {kk: _round(vv) for kk, vv in d.items()} for k, d in s["e"].items()}
            out_samples.append(row)

        return {
            "schema": SCHEMA,
            "source": self.url,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "t0_unix": round(self.t0_unix, 6),
            "interval_s": self.interval_s,
            "duration_s": round(self.samples[-1]["t"], 3),
            "scrapes": len(self.samples),
            "errors": self.errors,
            "info": self.info,
            "server": self.server,
            "buckets": {b: [None if math.isinf(e) else e for e in edges[b]] for b in bases},
            "samples": out_samples,
        }

    def save(self, path: str | Path = "metrics.json") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        return p

    def describe(self) -> str:
        if not self.samples:
            return f"no samples ({self.last_error or 'no scrape attempted'})"
        first, last = self.samples[0], self.samples[-1]

        def delta(*names: str) -> float:
            g = lambda s: next((s["m"][n] for n in names if n in s["m"]), 0.0)  # noqa: E731
            return g(last) - g(first)

        steps = delta("iteration_tokens_total_count")
        span = max(1e-9, last["t"] - first["t"])
        q = delta("prefix_cache_queries_total", "gpu_prefix_cache_queries_total")
        hits = delta("prefix_cache_hits_total", "gpu_prefix_cache_hits_total")
        hit = f"prefix hit {hits / q:.1%}" if q else "no prefix-cache queries"
        return (
            f"{len(self.samples)} scrapes over {last['t']:.1f}s "
            f"({self.errors} failed) · {len(last['m'])} metrics, "
            f"{len(last.get('h', {}))} histograms\n"
            f"  {steps:.0f} engine steps ({steps / span:.1f}/s) · "
            f"{delta('generation_tokens_total') / span:.0f} decode tok/s · "
            f"{delta('prompt_tokens_total') / span:.0f} prompt tok/s\n"
            f"  {delta('request_success_total'):.0f} requests finished · "
            f"{delta('num_preemptions_total'):.0f} preemptions · {hit}"
        )
