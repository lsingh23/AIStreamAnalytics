"""Gemini adapter.

Gemini answers arrive over XHR from .../BardFrontendService/StreamGenerate (and stored chats from
batchexecute?rpcids=hNvQHb), not as SSE. The body is Google's batchexecute framing:

    )]}'
    <length>
    [["wrb.fr",null,"<JSON string>"]]
    ...

Each "wrb.fr" entry holds a JSON *string* of positional (unlabeled) nested arrays; each one is a
cumulative snapshot of the answer so far. Positions observed in a 2026-10 capture, inside the
response candidate c = payload[4][0]:

  payload[1][0]   conversation id (c_...)
  c[1][0]         answer markdown
  c[37][1][k]     a "Searched the web" step; [7] = [[favicon, url, title, None, []], ...]  -> retrieved
  c[2][1][i]      one citation: [[answer_text, None, None, [[start, end], ...]], [n],
                                 [[url#:~:text=..., title, favicon, snippet, spans, None, site_name, ...]], id]
                                                                                    -> cited (+ answer offsets)
  c[12]           shopping/product cards

The stream does NOT include the search queries Gemini ran (the UI only says "Searched the web").
When the known positions are empty, the adapter falls back to harvesting every external URL.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterator
from urllib.parse import unquote

from ..schema import Result, Source, domain_of, normalize_url, youtube_id
from .base import Adapter

URL_RE = re.compile(r"^https?://\S+$")
INTERNAL = ("google.com", "gstatic.com", "googleusercontent.com", "ggpht.com", "ytimg.com", "googleapis.com")


def iter_frames(raw: bytes) -> Iterator[Any]:
    text = raw.decode("utf-8", "replace")
    if text.startswith(")]}'"):
        text = text[4:]
    dec = json.JSONDecoder()
    pos, n = 0, len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        if text[pos].isdigit():  # length prefix
            while pos < n and text[pos].isdigit():
                pos += 1
            continue
        try:
            obj, pos = dec.raw_decode(text, pos)
        except json.JSONDecodeError:
            nl = text.find("\n", pos)  # skip junk to the next line
            pos = n if nl < 0 else nl + 1
            continue
        yield obj


def payloads(raw: bytes) -> list[Any]:
    out = []
    for frame in iter_frames(raw):
        for entry in frame if isinstance(frame, list) else []:
            if isinstance(entry, list) and len(entry) > 2 and entry[0] == "wrb.fr" and isinstance(entry[2], str):
                try:
                    out.append(json.loads(entry[2]))
                except json.JSONDecodeError:
                    pass
    return out


def walk_lists(obj: Any, path: tuple = ()) -> Iterator[tuple[tuple, list]]:
    if isinstance(obj, list):
        yield path, obj
        for i, v in enumerate(obj):
            yield from walk_lists(v, path + (i,))


def _get(obj: Any, *path: int) -> Any:
    for i in path:
        if not isinstance(obj, list) or len(obj) <= i:
            return None
        obj = obj[i]
    return obj


def _lists(v: Any) -> list[Any]:
    return [x for x in v if isinstance(x, list)] if isinstance(v, list) else []


def _text_fragment(url: str) -> str:
    """The highlighted passage Gemini links to (#:~:text=start,end)."""
    _, _, frag = url.partition("#:~:text=")
    return " … ".join(unquote(p) for p in frag.split(",") if p) if frag else ""


class GeminiAdapter(Adapter):
    name = "gemini"

    @classmethod
    def detect(cls, raw: bytes) -> bool:
        return raw.lstrip().startswith(b")]}'") or b'"wrb.fr"' in raw[:5000]

    @staticmethod
    def _upsert(sources: dict[str, Source], url: str, title: Any = "") -> Source:
        key = normalize_url(url)
        src = sources.get(key)
        if src is None:
            src = sources[key] = Source(url=url.split("#:~:text=")[0], type="youtube" if youtube_id(url) else "web")
        if not src.title and isinstance(title, str):
            src.title = title
        return src

    def parse(self, raw: bytes, meta: dict[str, Any] | None = None) -> Result:
        meta = meta or {}
        pls = payloads(raw)
        result = Result(platform=self.name, captured_at=meta.get("captured_at", ""))
        # the last snapshot that carries answer text is the complete one (trailing payloads are status pings)
        last = next((p for p in reversed(pls) if isinstance(_get(p, 4, 0, 1, 0), str) and _get(p, 4, 0, 1, 0)), None)
        cand = _get(last, 4, 0)
        result.answer = (_get(cand, 1, 0) or "").strip()
        conv = _get(last, 1, 0)
        result.conversation_id = conv if isinstance(conv, str) else ""
        sources: dict[str, Source] = {}

        steps = _lists(_get(cand, 37, 1))
        for step in steps:
            for item in _lists(_get(step, 7)):
                url = _get(item, 1)
                if isinstance(url, str) and URL_RE.match(url):
                    src = self._upsert(sources, url, _get(item, 2))
                    src.retrieved = True
                    src.extra.setdefault("step", _get(step, 1))

        citations = _lists(_get(cand, 2, 1))
        for cit in citations:
            span = _get(cit, 0)
            for ref in _lists(_get(cit, 2)):
                url = _get(ref, 0)
                if not isinstance(url, str) or not URL_RE.match(url):
                    continue
                src = self._upsert(sources, url, _get(ref, 1))
                src.cited = True
                if not src.snippet and isinstance(_get(ref, 3), str):
                    src.snippet = _get(ref, 3).strip()
                if isinstance(_get(ref, 6), str):
                    src.attribution = src.attribution or _get(ref, 6)
                for start, end in [r for r in _lists(_get(span, 3))[:1] if len(r) == 2]:
                    src.cited_spans.append({"type": "gemini_citation", "start": start, "end": end,
                                            "matched_text": _get(span, 0) or "", "cited_text": _text_fragment(url)})

        fallback = not sources
        paths: Counter[str] = Counter()
        if fallback:  # layout changed: harvest every external URL with a neighbouring title string
            for p in pls:
                for path, lst in walk_lists(p):
                    for url in (v for v in lst if isinstance(v, str) and URL_RE.match(v)):
                        if domain_of(url).endswith(INTERNAL) and not youtube_id(url):
                            continue
                        title = next((v for v in lst if isinstance(v, str) and not URL_RE.match(v) and " " in v.strip() and len(v) < 300), "")
                        src = self._upsert(sources, url, title)
                        src.retrieved = True
                        src.cited = src.cited or url in result.answer
                        paths["/".join(map(str, path))] += 1

        result.sources = sorted(sources.values(), key=lambda s: (not s.cited, not s.retrieved))
        result.diagnostics = {
            "input": "batchexecute",
            "payloads": len(pls),
            "search_steps": [_get(s, 1) for s in steps],
            "citations": len(citations),
            "shopping_cards": len(_lists(_get(cand, 12, 15))),
            "fallback_url_harvest": fallback,
            "url_paths": dict(paths.most_common(20)),
            "note": "Gemini does not stream the search queries it ran; only a 'Searched the web' step",
        }
        return result

    def inspect(self, raw: bytes) -> tuple[dict[str, Any], list[Any]]:
        pls = payloads(raw)
        return {"frames": sum(1 for _ in iter_frames(raw)), "wrb.fr payloads": len(pls)}, pls
