"""Perplexity adapter.

Accepts the raw SSE body of POST /rest/sse/perplexity_ask, or the stored thread JSON
(GET /rest/thread/<slug>, which only keeps query_str + web_results per entry).

Stream format (observed 2026-10): every event carries a "blocks" list. A block is either a full
snapshot ({"intended_usage": "web_results", "web_result_block": {...}}) or a JSON-patch diff against
the previous state ({"intended_usage": ..., "diff_block": {"field": "web_result_block",
"patches": [{"op", "path", "value"}]}}). The final event repeats every block in full, but a truncated
stream only has diffs, so the blocks are rebuilt by applying the patches.

Where the retrieval data lives:
  workflow_block.steps[]            one step per search round ("Searching the web", "Comparing ...")
    items[type=WORKFLOW_ITEM_QUERIES].payload.queries_payload.queries   -> fan-out queries
    items[type=WORKFLOW_ITEM_SOURCES].payload.sources_payload.sources   -> what that round returned
    items[type=WORKFLOW_ITEM_TEXT].payload.text_payload.text            -> the answer, with [n] markers
  web_result_block.web_results      every retrieved result (name, url, snippet, timestamp, meta_data)
  sources_mode_block.rows[]         per result: status SELECTED (cited, "citation": n) or REVIEWED
                                    (read but not cited) -- Perplexity labels retrieved-vs-cited itself
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ..schema import Result, Source, normalize_url, youtube_id
from ..sse import decode_data, iter_events
from .base import Adapter

MARKER_RE = re.compile(r"\[(\d{1,3})\]")
EXTRA_FLAGS = ("is_navigational", "is_knowledge_card", "is_widget", "is_image", "is_memory")


def _apply(doc: Any, op: str, path: str, value: Any) -> Any:
    """Apply one JSON-patch operation; returns the new root."""
    if path == "":
        return None if op == "remove" else value
    tokens = [t.replace("~1", "/").replace("~0", "~") for t in path.lstrip("/").split("/")]
    parent = doc
    for t in tokens[:-1]:
        parent = parent[int(t)] if isinstance(parent, list) else parent.setdefault(t, {})
    last = tokens[-1]
    if isinstance(parent, list):
        i = len(parent) if last == "-" else int(last)
        if op == "add":
            parent.insert(i, value)
        elif op == "replace":
            parent[i] = value
        elif op == "remove":
            parent.pop(i)
    elif op == "remove":
        parent.pop(last, None)
    else:
        parent[last] = value
    return doc


class BlockState:
    """Rebuilds Perplexity blocks (keyed by intended_usage) from snapshots and diff_block patches."""

    def __init__(self) -> None:
        self.blocks: dict[str, dict[str, Any]] = {}
        self.top: dict[str, Any] = {}
        self.errors: list[str] = []
        self.events = 0
        self.final = False

    def feed(self, ev: Any) -> None:
        if not isinstance(ev, dict) or not ev:
            return
        self.events += 1
        self.top.update({k: v for k, v in ev.items() if k != "blocks"})
        self.final = self.final or ev.get("status") == "COMPLETED"
        for b in ev.get("blocks") or []:
            usage = b.get("intended_usage", "?")
            state = self.blocks.setdefault(usage, {})
            diff = b.get("diff_block")
            if isinstance(diff, dict):
                field = diff.get("field", "")
                cur = state.get(field)
                for p in diff.get("patches") or []:
                    try:
                        cur = _apply(cur, p.get("op", ""), p.get("path", ""), p.get("value"))
                    except (IndexError, KeyError, ValueError, TypeError, AttributeError) as e:
                        self.errors.append(f"{usage} {p.get('op')} {p.get('path')}: {e!r}")
                state[field] = cur
            else:
                state.update({k: v for k, v in b.items() if k != "intended_usage"})

    def get(self, usage: str, field: str) -> dict[str, Any]:
        v = self.blocks.get(usage, {}).get(field)
        return v if isinstance(v, dict) else {}


class PerplexityAdapter(Adapter):
    name = "perplexity"

    @classmethod
    def detect(cls, raw: bytes) -> bool:
        head = raw[:50000]
        return b"web_result" in head or (b"backend_uuid" in head and b"thread_url_slug" in head) or b"source_entries" in head

    def _load(self, raw: bytes) -> tuple[BlockState | None, dict[str, Any] | None]:
        stripped = raw.lstrip()
        if stripped[:1] == b"{":
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                doc = None
            if isinstance(doc, dict) and ("source_entries" in doc or "entries" in doc):
                return None, doc
        state = BlockState()
        for ev in iter_events(raw):
            state.feed(decode_data(ev.data))
        return state, None

    def parse(self, raw: bytes, meta: dict[str, Any] | None = None) -> Result:
        meta = meta or {}
        state, thread = self._load(raw)
        result = Result(platform=self.name, captured_at=meta.get("captured_at", ""))
        sources: dict[str, Source] = {}
        if thread is not None:
            return self._parse_thread(thread, result, sources)
        assert state is not None

        top = state.top
        body = (meta.get("request") or {}).get("body")
        body = body if isinstance(body, dict) else {}
        params = body.get("params") or {}
        result.prompt = top.get("query_str") or body.get("query_str", "")
        result.model = top.get("display_model") or params.get("model_preference", "")
        result.conversation_id = top.get("thread_url_slug") or top.get("backend_uuid", "")

        answer = ""
        steps = state.get("workflow_root", "workflow_block").get("steps") or []
        for n, step in enumerate(steps):
            step_queries: list[str] = []
            for item in step.get("items") or []:
                payload = item.get("payload") or {}
                for q in (payload.get("queries_payload") or {}).get("queries") or []:
                    q = q.get("query") if isinstance(q, dict) else q
                    if isinstance(q, str) and q.strip():
                        step_queries.append(q.strip())
                        if q.strip().lower() not in {x.lower() for x in result.fanout_queries}:
                            result.fanout_queries.append(q.strip())
                for r in (payload.get("sources_payload") or {}).get("sources") or []:
                    if isinstance(r, dict) and isinstance(r.get("url"), str):
                        src = self._upsert(sources, r)
                        src.retrieved = True
                        src.extra.setdefault("step", step.get("title") or n)
                        for q in step_queries:
                            if q not in src.extra.setdefault("queries", []):
                                src.extra["queries"].append(q)
                text = payload.get("text_payload")
                if isinstance(text, dict):
                    answer = text.get("text") or "".join(c for c in text.get("chunks") or [] if isinstance(c, str))

        for r in state.get("web_results", "web_result_block").get("web_results") or []:
            if isinstance(r, dict) and isinstance(r.get("url"), str):
                self._upsert(sources, r).retrieved = True

        by_citation: dict[int, Source] = {}
        rows = state.get("sources_answer_mode", "sources_mode_block").get("rows") or []
        for row in rows:
            wr = row.get("web_result") if isinstance(row, dict) else None
            if not isinstance(wr, dict) or not isinstance(wr.get("url"), str):
                continue
            src = self._upsert(sources, wr)
            src.retrieved = True
            src.extra["status"] = row.get("status")
            if isinstance(row.get("citation"), int):
                src.cited = True
                src.extra["citation_number"] = row["citation"]
                by_citation[row["citation"]] = src
        for m in MARKER_RE.finditer(answer):
            src = by_citation.get(int(m.group(1)))
            if src is not None:
                src.cited_spans.append({"type": "marker", "start": m.start(), "end": m.end(), "matched_text": m.group(0)})

        result.answer = answer.strip()
        result.sources = sorted(sources.values(), key=lambda s: (not s.cited, s.extra.get("citation_number", 999)))
        result.diagnostics = {
            "input": "sse",
            "events": state.events,
            "complete": state.final,
            "blocks": sorted(state.blocks),
            "steps": [s.get("title") for s in steps],
            "row_status": dict(Counter(str(r.get("status")) for r in rows if isinstance(r, dict))),
            "errors": state.errors[:20],
        }
        return result

    def _parse_thread(self, doc: dict[str, Any], result: Result, sources: dict[str, Source]) -> Result:
        entries = doc.get("source_entries") or doc.get("entries") or []
        for e in entries:
            result.prompt = result.prompt or e.get("query_str", "")
            result.conversation_id = result.conversation_id or e.get("entry_uuid", "")
            for r in e.get("web_results") or []:
                if isinstance(r, dict) and isinstance(r.get("url"), str):
                    self._upsert(sources, r).retrieved = True
        result.sources = list(sources.values())
        result.diagnostics = {"input": "thread_json", "entries": len(entries),
                              "note": "stored thread keeps only web_results: no fan-out queries or citation status"}
        return result

    @staticmethod
    def _upsert(sources: dict[str, Source], r: dict[str, Any]) -> Source:
        url = r["url"]
        key = normalize_url(url)
        src = sources.get(key)
        if src is None:
            src = sources[key] = Source(url=url, type="youtube" if youtube_id(url) else "web")
        src.title = src.title or r.get("name") or r.get("title") or ""
        if not src.snippet and r.get("snippet"):
            src.snippet = r["snippet"]
        if not src.pub_date and r.get("timestamp"):
            src.pub_date = str(r["timestamp"])[:10]
        md = r.get("meta_data") if isinstance(r.get("meta_data"), dict) else {}
        if md.get("citation_domain_name"):
            src.attribution = src.attribution or md["citation_domain_name"]
        trust = r.get("trust")
        if isinstance(trust, dict) and trust.get("name"):
            src.extra.setdefault("trust", trust["name"])
            src.extra.setdefault("trust_reason", trust.get("description", ""))
        for k in EXTRA_FLAGS:
            if r.get(k):
                src.extra[k] = True
        return src

    def inspect(self, raw: bytes) -> tuple[dict[str, Any], list[Any]]:
        state, thread = self._load(raw)
        if thread is not None:
            return {"input": "thread_json", "keys": sorted(thread)}, [thread]
        summary = {"events": state.events, "complete": state.final, "blocks": {k: sorted(v) for k, v in state.blocks.items()},
                   "patch errors": state.errors[:10]}
        return summary, [state.blocks]
