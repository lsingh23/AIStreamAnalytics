"""Claude (claude.ai) adapter.

Accepts either
  * the raw SSE body of POST /api/organizations/<org>/chat_conversations/<id>/completion, or
  * the stored conversation JSON from GET .../chat_conversations/<id>?tree=True&rendering_mode=messages.

Claude's stream is the Messages API event protocol (message_start, content_block_start/delta/stop,
message_stop): an answer is a list of typed content blocks. Each web search is its own tool-call
block whose input is the query (= one fan-out query); the matching tool-result block lists every
retrieved result; citations sit on the text blocks. So "retrieved vs cited" falls straight out of
the block types instead of being reconstructed from reference markers as with ChatGPT.

Both the claude.ai names (tool_use / tool_result, citation_start_delta / citation_end_delta) and the
public API names (server_tool_use / web_search_tool_result, citations_delta) are handled; anything
else is counted in diagnostics.unknown_deltas so format drift shows up.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ..connect import is_connect
from ..schema import Result, Source, normalize_url, youtube_id
from ..sse import decode_data, iter_events
from .base import Adapter
from .claude_timeline import Timeline

TOOL_CALL_TYPES = {"tool_use", "server_tool_use"}
TOOL_RESULT_TYPES = {"tool_result", "web_search_tool_result", "web_fetch_tool_result"}
CONV_ID_RE = re.compile(r"/chat_conversations/([0-9a-f-]{36})")
SNIPPET_MAX = 1000  # web_fetch results carry whole pages
ANSWER_SEP = "\n\n"  # between text blocks (e.g. the "I'll search..." preamble and the answer)


class BlockAssembler:
    """Rebuilds content blocks from content_block_start/delta/stop events."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []
        self._open: dict[Any, dict[str, Any]] = {}
        self._json: dict[int, list[str]] = {}
        self._cites: dict[str, dict[str, Any]] = {}
        self.event_types: Counter[str] = Counter()
        self.delta_types: Counter[str] = Counter()
        self.unknown_deltas: Counter[str] = Counter()
        self.other_events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stopped = False

    def feed(self, event: str | None, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        etype = payload.get("type") or event or "?"
        self.event_types[etype] += 1
        idx = payload.get("index")
        if etype == "message_start":
            self.messages.append(payload.get("message") or {})
            self._open = {}  # block indexes restart per message
        elif etype == "content_block_start":
            block = dict(payload.get("content_block") or {})
            block.setdefault("citations", [])
            self._open[idx] = block
            self.blocks.append(block)
        elif etype == "content_block_delta":
            block = self._open.get(idx)
            if block is None:
                self.errors.append(f"delta for unknown block {idx}")
                return
            self._delta(block, payload.get("delta") or {})
        elif etype == "content_block_stop":
            block = self._open.get(idx)
            if block is not None:
                self._finish(block)
        elif etype == "message_stop":
            self.stopped = True
        else:  # message_delta, message_limit, ping, error, ...
            self.other_events.append(payload)

    def _delta(self, block: dict[str, Any], d: dict[str, Any]) -> None:
        dtype = d.get("type", "?")
        self.delta_types[dtype] += 1
        if dtype == "text_delta":
            block["text"] = block.get("text", "") + d.get("text", "")
        elif dtype == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + d.get("thinking", "")
        elif dtype == "thinking_summary_delta":
            summary = (d.get("summary") or {}).get("summary")
            if summary:
                block.setdefault("summaries", []).append({"summary": summary})
        elif dtype == "input_json_delta":
            self._json.setdefault(id(block), []).append(d.get("partial_json", ""))
        elif dtype == "citations_delta":  # API: one finished citation, no offsets
            block["citations"].append(dict(d.get("citation") or {}))
        elif dtype == "citation_start_delta":  # claude.ai: citation opens at the current text offset
            c = dict(d.get("citation") or {})
            c.setdefault("start_index", len(block.get("text", "")))
            block["citations"].append(c)
            if c.get("uuid"):
                self._cites[c["uuid"]] = c
        elif dtype == "citation_end_delta":
            c = self._cites.get(d.get("citation_uuid") or "")
            if c is not None:
                c.setdefault("end_index", len(block.get("text", "")))
        elif dtype in ("signature_delta", "tool_use_block_update_delta"):
            for k, v in d.items():
                if k != "type":
                    block.setdefault(k, v)
        else:
            self.unknown_deltas[dtype] += 1

    def _finish(self, block: dict[str, Any]) -> None:
        parts = self._json.pop(id(block), None)
        if parts:
            raw = "".join(parts)
            try:
                block["input"] = json.loads(raw)
            except json.JSONDecodeError:
                self.errors.append(f"unparseable tool input: {raw[:200]}")
                block["input_raw"] = raw

    def close(self) -> None:
        """Finish blocks left open by a truncated stream."""
        for block in self._open.values():
            if id(block) in self._json:
                self._finish(block)


def _cited_sentence(text: str, start: Any, end: Any) -> str:
    """The cited span; claude.ai's timeline marks a citation as a point (start == end) right after the
    sentence it supports, so fall back to that sentence."""
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    if end > start:
        return text[start:end]
    head = text[:start].rstrip()
    cut = max(head.rfind("\n"), head.rfind(". ", 0, len(head) - 1))
    return head[cut + 1:].strip()


def _find_url(obj: Any) -> str | None:
    if isinstance(obj, dict):
        if isinstance(obj.get("url"), str) and obj["url"].startswith("http"):
            return obj["url"]
        for v in obj.values():
            u = _find_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = _find_url(v)
            if u:
                return u
    return None


def _result_items(content: Any) -> list[dict[str, Any]]:
    """Every dict with a URL inside a tool result's content (knowledge items, web_search_result, ...)."""
    out: list[dict[str, Any]] = []
    if isinstance(content, dict):
        if isinstance(content.get("url"), str):
            return [content]
        for v in content.values():
            out.extend(_result_items(v))
    elif isinstance(content, list):
        for v in content:
            out.extend(_result_items(v))
    return out


class ClaudeAdapter(Adapter):
    name = "claude"

    @classmethod
    def detect(cls, raw: bytes) -> bool:
        head = raw[:20000]
        return is_connect(raw) or any(m in head for m in (b"content_block_start", b'"chat_messages"'))

    def parse(self, raw: bytes, meta: dict[str, Any] | None = None) -> Result:
        meta = meta or {}
        blocks, prompt, info, diag = self._load(raw)
        result = Result(platform=self.name, captured_at=meta.get("captured_at", ""), diagnostics=diag)
        req = meta.get("request") or {}
        body = req.get("body") if isinstance(req.get("body"), dict) else {}
        m = CONV_ID_RE.search(req.get("url", ""))
        result.conversation_id = info.get("conversation_id") or (m.group(1) if m else "")
        result.model = info.get("model") or body.get("model", "")
        result.prompt = body.get("prompt") or prompt

        sources: dict[str, Source] = {}
        queries: list[str] = []
        query_by_call: dict[str, str] = {}
        last_call: dict[str, Any] = {}
        answer_parts: list[str] = []
        tools: list[dict[str, Any]] = []

        for b in blocks:
            btype = b.get("type")
            if btype in TOOL_CALL_TYPES:
                last_call = b
                name = b.get("name", "")
                inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                tools.append({"name": name, "input": inp})
                if "fetch" in name and isinstance(inp.get("url"), str):
                    src = self._upsert(sources, {"url": inp["url"]})
                    src.retrieved = True
                    src.extra["fetched"] = True
                for q in self._queries_of(inp) if "search" in name else []:
                    query_by_call[b.get("id", "")] = q
                    if q.lower() not in {x.lower() for x in queries}:
                        queries.append(q)
            elif btype in TOOL_RESULT_TYPES:
                call_id = b.get("tool_use_id") or last_call.get("id", "")
                tool = b.get("name") or last_call.get("name", "") or btype
                query = query_by_call.get(call_id)
                for item in _result_items(b.get("content")):
                    src = self._upsert(sources, item)
                    src.retrieved = True
                    if query and query not in src.extra.setdefault("queries", []):
                        src.extra["queries"].append(query)
                    if "fetch" in tool:
                        src.extra["fetched"] = True
                    tools_seen = src.extra.setdefault("tools", [])
                    if tool not in tools_seen:
                        tools_seen.append(tool)
            elif btype == "text":
                offset = sum(len(p) + len(ANSWER_SEP) for p in answer_parts)
                text = b.get("text", "")
                answer_parts.append(text)
                for c in b.get("citations") or []:
                    url = _find_url(c)
                    if not url:
                        continue
                    details = c.get("details") if isinstance(c.get("details"), dict) else {}
                    src = self._upsert(sources, {**details, **c, "url": url})
                    src.cited = True
                    start, end = c.get("start_index"), c.get("end_index")
                    src.cited_spans.append({
                        "type": c.get("type") or details.get("type", "citation"),
                        "start": None if start is None else offset + start,
                        "end": None if end is None else offset + end,
                        "matched_text": _cited_sentence(text, start, end),
                        "cited_text": c.get("cited_text", ""),
                    })
            elif btype == "thinking":
                summaries = [s.get("summary") for s in b.get("summaries") or [] if isinstance(s, dict) and s.get("summary")]
                text = b.get("thinking") or "\n".join(summaries)
                if text:
                    result.reasoning.append(text)

        result.fanout_queries = queries
        result.sources = sorted(sources.values(), key=lambda s: (not s.cited, not s.retrieved))
        result.answer = ANSWER_SEP.join(answer_parts)
        diag["tool_calls"] = tools
        diag["block_types"] = dict(Counter(str(b.get("type")) for b in blocks))
        return result

    # -- loading -----------------------------------------------------------------------------

    def _load(self, raw: bytes) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
        stripped = raw.lstrip()
        if stripped[:1] == b"{":
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                doc = None
            if isinstance(doc, dict) and isinstance(doc.get("chat_messages"), list):
                return self._load_stored(doc)
        if is_connect(raw):  # claude.ai ConversationService/StreamTimeline
            tl = Timeline(raw)
            info = {"conversation_id": tl.conversation.get("1", ""), "model": (tl.conversation.get("6") or {}).get("2", ""),
                    "title": tl.conversation.get("2", "")}
            return tl.to_blocks(), tl.prompt, info, tl.diagnostics()

        asm = self.assemble(raw)
        info: dict[str, Any] = {}
        for m in asm.messages:
            info["model"] = info.get("model") or m.get("model", "")
        diag = {
            "input": "sse",
            "complete": asm.stopped,
            "event_types": dict(asm.event_types),
            "delta_types": dict(asm.delta_types),
            "unknown_deltas": dict(asm.unknown_deltas),
            "errors": asm.errors,
            "other_event_types": dict(Counter(str(e.get("type", "?")) for e in asm.other_events)),
        }
        return asm.blocks, "", info, diag

    @staticmethod
    def assemble(raw: bytes) -> BlockAssembler:
        asm = BlockAssembler()
        for ev in iter_events(raw):
            asm.feed(ev.event, decode_data(ev.data))
        asm.close()
        return asm

    @staticmethod
    def _load_stored(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
        msgs = sorted(doc["chat_messages"], key=lambda m: (m.get("index", 0), m.get("created_at", "")))
        last_ai = max((i for i, m in enumerate(msgs) if m.get("sender") == "assistant"), default=None)
        prompt = ""
        blocks: list[dict[str, Any]] = []
        if last_ai is not None:
            blocks = [b for b in msgs[last_ai].get("content") or [] if isinstance(b, dict)]
            human = [m for m in msgs[:last_ai] if m.get("sender") == "human"]
            if human:
                prompt = human[-1].get("text") or "".join(
                    b.get("text", "") for b in human[-1].get("content") or [] if isinstance(b, dict) and b.get("type") == "text")
        info = {"conversation_id": doc.get("uuid", ""), "model": doc.get("model") or ""}
        return blocks, prompt, info, {"input": "conversation_json", "messages": len(msgs)}

    # -- fields ------------------------------------------------------------------------------

    @staticmethod
    def _queries_of(inp: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for k in ("query", "q", "queries"):
            v = inp.get(k)
            for item in v if isinstance(v, list) else [v]:
                if isinstance(item, dict):
                    item = item.get("query") or item.get("q")
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
        return out

    @staticmethod
    def _upsert(sources: dict[str, Source], item: dict[str, Any]) -> Source:
        url = item["url"]
        key = normalize_url(url)
        src = sources.get(key)
        if src is None:
            src = sources[key] = Source(url=url, type="youtube" if youtube_id(url) else "web")
        md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if not src.title and isinstance(item.get("title"), str):
            src.title = item["title"]
        snippet = item.get("text") or item.get("snippet") or item.get("cited_text")
        if not src.snippet and isinstance(snippet, str):
            src.snippet = snippet[:SNIPPET_MAX]
            if len(snippet) > SNIPPET_MAX:
                src.extra["content_chars"] = len(snippet)
        if not src.pub_date and (item.get("page_age") or md.get("page_age")):
            src.pub_date = str(item.get("page_age") or md.get("page_age"))
        if not src.attribution:
            src.attribution = md.get("site_name") or md.get("site_domain") or ""
        for k, v in md.items():
            if k not in ("type", "site_name", "page_age") and v not in (None, "", [], {}):
                src.extra.setdefault(k, v)
        for k in ("is_missing", "is_citable", "prompt_context_metadata"):
            if item.get(k) not in (None, "", [], {}):
                src.extra.setdefault(k, item[k])
        return src

    def inspect(self, raw: bytes) -> tuple[dict[str, Any], list[Any]]:
        asm = self.assemble(raw)
        summary = {
            "event types": dict(asm.event_types),
            "delta types": dict(asm.delta_types),
            "unknown deltas": dict(asm.unknown_deltas),
            "block types": dict(Counter(str(b.get("type")) for b in asm.blocks)),
            "message_stop seen": asm.stopped,
            "errors": asm.errors[:10],
        }
        return summary, asm.blocks + asm.messages + asm.other_events
