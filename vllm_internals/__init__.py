"""Record a vLLM run (per-request token trace + /metrics time series) for the viewer in `app/`"""

from .client import Config, VLLMClient, run_batch
from .display import show_tokens, tokens_html
from .recorder import Recorder, parse_prom
from .trace import Clock, RequestTrace, build_trace, save_json, summary

__all__ = [
    "Clock",
    "Config",
    "Recorder",
    "RequestTrace",
    "VLLMClient",
    "build_trace",
    "parse_prom",
    "run_batch",
    "save_json",
    "show_tokens",
    "summary",
    "tokens_html",
]
