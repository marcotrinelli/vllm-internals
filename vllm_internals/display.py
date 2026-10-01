"""Token content of one request as HTML chips, for the notebook.

Same colour code as the viewer: prompt tokens blue, prompt tokens served from the prefix
cache teal, generated tokens pink. Hover a chip for its id, position and logprob.
"""

from __future__ import annotations

import html
from typing import Any

from .trace import RequestTrace

_STYLE = """<style>
.vi-toks{font:12px/1.8 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;
  word-break:break-word;border:1px solid #8884;border-radius:6px;padding:6px 8px;margin:4px 0 10px}
.vi-toks span{border-radius:3px;padding:0 1px}
.vi-toks .p{background:rgba(90,162,255,.22)} .vi-toks .c{background:rgba(42,212,200,.30)}
.vi-toks .g{background:rgba(255,111,174,.25)} .vi-toks span:nth-child(2n){filter:brightness(1.25)}
.vi-head{font:11px ui-sans-serif,system-ui,sans-serif;color:#888;text-transform:uppercase;
  letter-spacing:.06em}
</style>"""


def _visible(s: str) -> str:
    # newlines and tabs would otherwise vanish into the layout
    return s.replace("\n", "↵\n").replace("\t", "→\t")


def _chip(cls: str, text: str, title: str) -> str:
    return f'<span class="{cls}" title="{html.escape(title)}">{html.escape(_visible(text))}</span>'


def tokens_html(r: RequestTrace, vocab: dict[int, str]) -> str:
    cached = r.cached_tokens or 0
    ids = r.prompt_ids or []
    prompt = "".join(
        _chip("c" if i < cached else "p", vocab.get(t, "�"), f"pos {i} · id {t}")
        for i, t in enumerate(ids)
    )
    out_ids = r.output_ids or [None] * len(r.token_text)
    lps = r.token_logprob or [None] * len(r.token_text)
    output = "".join(
        _chip(
            "g",
            text,
            f"pos {len(ids) + j} · id {tid if tid is not None else '?'}"
            + (f" · logprob {lp:.3f}" if lp is not None else "")
            + f" · t {r.token_t[j]:.3f}s",
        )
        for j, (text, tid, lp) in enumerate(zip(r.token_text, out_ids, lps))
    )
    ttft = f"{r.t_first - r.t_submit:.3f}s" if r.t_first is not None else "n/a"
    return (
        _STYLE
        + f'<div class="vi-head">prompt · {len(ids)} tokens'
        + (f" ({cached} from prefix cache)" if cached else "")
        + f'</div><div class="vi-toks">{prompt}</div>'
        + f'<div class="vi-head">output · {len(r.token_text)} tokens · ttft {ttft}'
        + f" · finish {r.finish_reason or 'n/a'}</div>"
        + f'<div class="vi-toks">{output}</div>'
    )


def show_tokens(r: RequestTrace, vocab: dict[int, str]) -> Any:
    from IPython.display import HTML, display

    return display(HTML(tokens_html(r, vocab)))
