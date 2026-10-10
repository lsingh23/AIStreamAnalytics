"""ChatGPT adapter.

Accepts either
  * the raw SSE body of POST /backend-api/(f/)conversation (v1 delta encoding), or
  * the stored conversation JSON from GET /backend-api/conversation(s)/<id> (only its last turn is parsed).

Key names are searched recursively instead of via fixed paths, because OpenAI reshuffles
the metadata layout often; diagnostics record anything the assembler did not understand.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterator

from ..schema import Result, Source, normalize_url, youtube_id
from ..sse import decode_data, iter_events
from .base import Adapter

QUERY_KEYS = {"search_model_queries", "search_queries", "search_query"}
CITATION_KEYS = {"content_references", "citations"}
SOURCE_FIELDS = {"url", "title", "snippet", "pub_date", "attribution", "type", "ref_id", "refs"}
REF_RE = re.compile(r"turn(\d+)([a-z_]+?)(\d+)")
PUA_RE = re.compile("[\ue200-\ue2ff]")  # private-use chars wrapping legacy citation markers
CITE_TAG_RE = re.compile(r"<Cite\s+refs=\{(\[.*?\])\}\s*/?>", re.S)
LINK_TAG_RE = re.compile(r"<Link\s+[^>]*?url=\"([^\"]+)\"", re.S)
MD_LINK_RE = re.compile(r"\[[^\]\n]*\]\((https?://[^)\s]+)\)")
UI_TAG_RE = re.compile(r"<(Entity|AsyncImage)\s+((?:[^>\"{}]|\"[^\"]*\"|\{[^{}]*\})*?)\s*/?>", re.S)
UI_ATTR_RE = re.compile(r"(\w+)=(?:\"([^\"]*)\"|\{([^{}]*)\})")
PRODUCT_REF_RE = re.compile(r"turn\d+product\d+")
# 2026-10: shopping answers bind product cards to records under metadata.model_dil_v2.appData.opGenui
# .componentResults.<id>.state...extraParams.product, keyed by `cite` ("turn0product0").
PRODUCT_FIELDS = ("id", "title", "price", "merchants", "rating", "num_reviews", "description", "specs",
                  "featured_tag", "price_history", "merchant_badges")
# ChatGPT serves two snippet sizes per source: a short search-engine extract (~150-200 chars) and a long
# extract from its own page index (~900-1,200 chars). result_source used to name the pipeline; it was
# removed in July 2026, so length is what tells them apart now.
LONG_SNIPPET_MIN = 400
# where a search_result_groups list lives -> human label
SRG_LOCATIONS = {
    "metadata": "search_results",                  # web.run tool messages: everything the search returned
    "inline_cot_expandable_content": "cot_panel",  # "Searched N websites" panel
    "genui_components": "answer_components",       # source lists attached to the final answer UI
}


# ----------------------------------------------------------------------------- delta assembly

def _pointer(path: str) -> list[str]:
    if not path:
        return []
    return [t.replace("~1", "/").replace("~0", "~") for t in path.lstrip("/").split("/")]


class DeltaAssembler:
    """Rebuilds message envelopes from ChatGPT's v1 JSON-patch style SSE deltas."""

    def __init__(self) -> None:
        self.docs: dict[Any, Any] = {}
        self.cur: Any = 0
        self.last_path: str | None = None
        self.last_op: str | None = None
        self.in_patch = False
        self.other_events: list[dict[str, Any]] = []
        self.event_types: Counter[str] = Counter()
        self.op_counts: Counter[str] = Counter()
        self.unknown_ops: Counter[str] = Counter()
        self.errors: list[str] = []
        self._auto = 0

    def feed(self, event: str | None, payload: Any) -> None:
        self.event_types[event or "message"] += 1
        if not isinstance(payload, dict):
            return
        if "c" in payload:
            self.cur = payload["c"]
        is_delta = "p" in payload or "o" in payload or (set(payload) <= {"v", "c"} and "v" in payload)
        if not is_delta:
            self.other_events.append(payload)
            if isinstance(payload.get("message"), dict):  # legacy non-delta stream: full envelope each event
                self._auto += 1
                self.docs[f"full-{self._auto}"] = payload
            return

        v = payload.get("v")
        if "p" in payload or "o" in payload:
            path = payload.get("p", "")
            op = payload.get("o", "add" if path == "" else "replace")
        elif self.in_patch and isinstance(v, list) and all(isinstance(x, dict) and ("p" in x or "o" in x) for x in v):
            path, op = "", "patch"  # bare {"v": [...ops]} continues a previous patch
        elif self.cur not in self.docs or (isinstance(v, dict) and "message" in v and self.last_path in (None, "")):
            path, op = "", "add"
        else:
            path, op = self.last_path or "", self.last_op or "append"

        if path == "" and op == "add" and "c" not in payload and self.cur in self.docs:
            self._auto += 1
            self.cur = f"auto-{self._auto}"

        self.in_patch = op == "patch"
        if op == "patch":
            self.op_counts["patch"] += 1
            for sub in v or []:
                if isinstance(sub, dict):
                    self._apply(sub.get("o", "replace"), sub.get("p", ""), sub.get("v"))
        else:
            self._apply(op, path, v)

    def _apply(self, op: str, path: str, v: Any) -> None:
        self.op_counts[op] += 1
        self.last_path, self.last_op = path, op
        tokens = _pointer(path)
        try:
            if not tokens:
                if op in ("add", "replace"):
                    self.docs[self.cur] = v
                elif op == "append" and isinstance(self.docs.get(self.cur), dict) and isinstance(v, dict):
                    self.docs[self.cur].update(v)
                else:
                    self.unknown_ops[f"{op}@root"] += 1
                return
            node = self.docs.setdefault(self.cur, {})
            for tok in tokens[:-1]:
                if isinstance(node, list):
                    node = node[int(tok)]
                else:
                    node = node.setdefault(tok, {})
            self._apply_at(node, tokens[-1], op, v)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as e:
            if len(self.errors) < 50:
                self.errors.append(f"{op} {path}: {type(e).__name__}: {e}")

    def _apply_at(self, parent: Any, key: str, op: str, v: Any) -> None:
        if isinstance(parent, list):
            idx = len(parent) if key == "-" else int(key)
            existing = parent[idx] if idx < len(parent) else None
        else:
            idx = key
            existing = parent.get(key)

        def put(value: Any) -> None:
            if isinstance(parent, list) and idx >= len(parent):
                parent.append(value)
            else:
                parent[idx] = value

        if op == "append":
            if existing is None:
                put(v)
            elif isinstance(existing, str):
                put(existing + (v if isinstance(v, str) else json.dumps(v)))
            elif isinstance(existing, list):
                existing.extend(v) if isinstance(v, list) else existing.append(v)
            elif isinstance(existing, dict) and isinstance(v, dict):
                existing.update(v)
            else:
                put(v)
        elif op in ("replace", "add"):
            if op == "add" and isinstance(parent, list) and key != "-":
                parent.insert(idx, v)
            else:
                put(v)
        elif op == "truncate":
            if isinstance(existing, (str, list)):
                put(existing[: int(v)])
        elif op in ("remove", "delete"):
            if isinstance(parent, list):
                parent.pop(idx)
            else:
                parent.pop(idx, None)
        else:
            self.unknown_ops[op] += 1

    def envelopes(self) -> list[dict[str, Any]]:
        return [d for d in self.docs.values() if isinstance(d, dict)]


# ----------------------------------------------------------------------------- helpers

def walk(obj: Any, parent_key: str | None = None) -> Iterator[tuple[str | None, Any, Any]]:
    """Yield (key, value, parent) for every dict entry / list item, depth first."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v, obj
            yield from walk(v, k)
    elif isinstance(obj, list):
        for item in obj:
            yield parent_key, item, obj
            yield from walk(item, parent_key)


def _strings_from_query_value(v: Any) -> Iterator[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for item in v:
            yield from _strings_from_query_value(item)
    elif isinstance(v, dict):
        for k in ("q", "query", "queries", "search_query"):
            if k in v:
                yield from _strings_from_query_value(v[k])


def _ref_key(ref: Any) -> str | None:
    if isinstance(ref, dict) and "ref_index" in ref:
        return f"turn{ref.get('turn_index', 0)}{ref.get('ref_type', 'search')}{ref['ref_index']}"
    if isinstance(ref, str) and REF_RE.fullmatch(ref):
        return ref
    return None


def walk_paths(obj: Any, path: tuple = ()) -> Iterator[tuple[tuple, Any]]:
    """Yield (path, value) for every nested value; path holds dict keys and list indexes."""
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_paths(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_paths(v, path + (i,))


def _answer_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assistant messages that make up the user-visible answer."""
    out = []
    for m in messages:
        content = m.get("content") or {}
        if ((m.get("author") or {}).get("role") == "assistant" and m.get("recipient", "all") == "all"
                and content.get("content_type") == "text" and m.get("channel") in (None, "final")
                and _message_text(m).strip()):
            out.append(m)
    return out


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or {}
    parts = content.get("parts")
    if isinstance(parts, list):
        return "".join(p for p in parts if isinstance(p, str))
    for k in ("text", "content", "result"):
        if isinstance(content.get(k), str):
            return content[k]
    return ""


# ----------------------------------------------------------------------------- adapter

class ChatGPTAdapter(Adapter):
    name = "chatgpt"

    @classmethod
    def detect(cls, raw: bytes) -> bool:
        head = raw[:20000]
        return any(m in head for m in (b'"v1"', b"delta_encoding", b'"mapping"', b"conversation_id", b"content_references"))

    def parse(self, raw: bytes, meta: dict[str, Any] | None = None) -> Result:
        meta = meta or {}
        messages, extra_blobs, diag = self._load(raw)
        result = Result(platform=self.name, captured_at=meta.get("captured_at", ""))
        result.diagnostics = diag

        for env in extra_blobs:
            if isinstance(env, dict) and env.get("conversation_id"):
                result.conversation_id = env["conversation_id"]
        result.conversation_id = result.conversation_id or meta.get("conversation_id", "")
        result.prompt = self._prompt(messages, meta)
        result.model = self._model(messages, meta)

        blobs = messages + extra_blobs
        result.fanout_queries = self._queries(messages, blobs)
        sources: dict[str, Source] = {}
        self._retrieved(blobs, sources)
        refs = self._cited(messages, blobs, sources)
        result.sources = sorted(sources.values(), key=lambda s: (not s.cited, not s.retrieved, not s.supporting))
        result.answer = self._answer(messages, refs)
        result.reasoning = self._reasoning(messages)
        result.products = self._products(blobs, result.answer)
        result.insights = self._insights(messages, blobs)
        diag["tool_calls"] = [
            {"recipient": m.get("recipient"), "text": _message_text(m)[:2000]}
            for m in messages
            if (m.get("author") or {}).get("role") == "assistant" and m.get("recipient") not in (None, "all")
        ]
        diag["content_types"] = dict(Counter((m.get("content") or {}).get("content_type", "?") for m in messages))
        return result

    # -- loading -----------------------------------------------------------------------------

    def _load(self, raw: bytes) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
        stripped = raw.lstrip()
        if stripped[:1] == b"{":
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                doc = None
            # 2026-10: GET /backend-api/conversations/<id>?num_turns=10 returns the current branch as a flat,
            # ordered `messages` list (paged via page_info) instead of the `mapping` tree.
            if isinstance(doc, dict) and (isinstance(doc.get("mapping"), dict) or isinstance(doc.get("messages"), list)):
                if isinstance(doc.get("mapping"), dict):
                    fmt = "mapping"
                    msgs = [n["message"] for n in doc["mapping"].values() if isinstance(n, dict) and isinstance(n.get("message"), dict)]
                    msgs.sort(key=lambda m: m.get("create_time") or 0)
                else:
                    fmt = "messages"
                    msgs = [m for m in doc["messages"] if isinstance(m, dict) and isinstance(m.get("author"), dict)]
                # one Result per answer: keep only the last turn so prompt, sources and answer belong together
                starts = [i for i, m in enumerate(msgs) if (m.get("author") or {}).get("role") == "user" and _message_text(m).strip()]
                turn = msgs[starts[-1]:] if starts else msgs
                env = {"conversation_id": doc.get("conversation_id") or doc.get("id"), "title": doc.get("title"),
                       "page_info": doc.get("page_info")}
                return turn, [env], {"input": "conversation_json", "format": fmt, "turns": len(starts),
                                     "messages": len(turn), "messages_total": len(msgs)}

        asm = DeltaAssembler()
        for ev in iter_events(raw):
            asm.feed(ev.event, decode_data(ev.data))

        by_id: dict[str, dict[str, Any]] = {}
        envelopes = asm.envelopes()
        for env in envelopes + [e for e in asm.other_events if isinstance(e.get("input_message"), dict)]:
            # the entity sidebar stream (/backend-api/sidebar/conversation) wraps messages as {"type": "message", "data": ...}
            msg = env.get("message") or env.get("input_message") or (env.get("data") if env.get("type") == "message" else None)
            if isinstance(msg, dict):
                by_id[msg.get("id") or f"anon-{len(by_id)}"] = msg
        diag = {
            "input": "sse",
            "event_types": dict(asm.event_types),
            "op_counts": dict(asm.op_counts),
            "unknown_ops": dict(asm.unknown_ops),
            "errors": asm.errors,
            "other_event_types": dict(Counter(str(e.get("type", "?")) for e in asm.other_events)),
            "messages": len(by_id),
        }
        extra = [{k: v for k, v in env.items() if k != "message"} for env in envelopes] + asm.other_events
        return list(by_id.values()), extra, diag

    def inspect(self, raw: bytes) -> tuple[dict[str, Any], list[Any]]:
        asm = DeltaAssembler()
        undecoded = 0
        for ev in iter_events(raw):
            payload = decode_data(ev.data)
            undecoded += isinstance(payload, str)
            asm.feed(ev.event, payload)
        summary = {
            "event types": dict(asm.event_types),
            "ops": dict(asm.op_counts),
            "unknown ops": dict(asm.unknown_ops),
            "undecoded payloads": undecoded,
            "non-delta event types": dict(Counter(str(e.get("type", "?")) for e in asm.other_events)),
        }
        return summary, asm.envelopes() + asm.other_events

    # -- fields ------------------------------------------------------------------------------

    @staticmethod
    def _prompt(messages: list[dict[str, Any]], meta: dict[str, Any]) -> str:
        body = (meta.get("request") or {}).get("body")
        if isinstance(body, dict) and isinstance((body.get("generic_entity_params") or {}).get("name"), str):
            return f"[entity sidebar] {body['generic_entity_params']['name']}"  # /backend-api/sidebar/conversation
        if isinstance(body, dict):
            for m in body.get("messages") or []:
                text = _message_text(m)
                if text:
                    return text
        for m in messages:
            if (m.get("author") or {}).get("role") == "user":
                text = _message_text(m)
                if text:
                    return text
        return ""

    @staticmethod
    def _model(messages: list[dict[str, Any]], meta: dict[str, Any]) -> str:
        for m in messages:
            md = m.get("metadata") or {}
            slug = md.get("resolved_model_slug") or md.get("model_slug")
            if slug:
                return slug
        body = (meta.get("request") or {}).get("body")
        return body.get("model", "") if isinstance(body, dict) else ""

    @staticmethod
    def _queries(messages: list[dict[str, Any]], blobs: list[Any]) -> list[str]:
        found: list[str] = []
        for k, v, parent in walk(blobs):
            if k in QUERY_KEYS and parent is not None and isinstance(parent, dict):
                found.extend(_strings_from_query_value(v))
            elif k == "queries" and isinstance(parent, dict) and "search" in str(parent.get("type", "")):
                found.extend(_strings_from_query_value(v))
        # tool calls such as recipient "web.run" with {"search_query": [{"q": ...}]} or search("...")
        for m in messages:
            if (m.get("author") or {}).get("role") != "assistant" or not str(m.get("recipient", "")).startswith("web"):
                continue
            text = _message_text(m)
            try:
                found.extend(q for kk, vv, pp in walk(json.loads(text)) if kk in QUERY_KEYS | {"queries"} and isinstance(pp, dict)
                             for q in _strings_from_query_value(vv))
            except (json.JSONDecodeError, TypeError):
                found.extend(mm.group(2) for mm in re.finditer(r"search\((['\"])(.+?)\1", text))
        seen, out = set(), []
        for q in (q.strip() for q in found):
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out

    @staticmethod
    def _upsert(sources: dict[str, Source], entry: dict[str, Any], location: str | None = None) -> Source:
        url = entry["url"]
        key = normalize_url(url)
        src = sources.get(key)
        if src is None:
            src = sources[key] = Source(url=url, type="youtube" if youtube_id(url) else "web")
        for f in ("title", "attribution"):
            if not getattr(src, f) and isinstance(entry.get(f), str):
                setattr(src, f, entry[f])
        snippet = entry.get("snippet")
        if isinstance(snippet, str) and len(snippet) > len(src.snippet):
            src.snippet = snippet  # keep the longest variant as the main snippet
        if location and "snippet" in entry:  # record every variant, including empty ones (e.g. YouTube)
            text = snippet if isinstance(snippet, str) else ""
            if not any(v["text"] == text for v in src.snippets):  # the same text repeats across panels; keep the first
                kind = "empty" if not text else "long" if len(text) >= LONG_SNIPPET_MIN else "short"
                src.snippets.append({"location": location, "kind": kind, "length": len(text), "text": text})
        pub = entry.get("pub_date")
        if not src.pub_date and pub is not None:
            if isinstance(pub, (int, float)):  # ChatGPT sends unix seconds
                pub = datetime.fromtimestamp(pub, tz=timezone.utc).strftime("%Y-%m-%d")
            src.pub_date = str(pub)
        for ref in [entry.get("ref_id")] + list(entry.get("refs") or []):
            rk = _ref_key(ref)
            if rk and rk not in src.ref_ids:
                src.ref_ids.append(rk)
        for k, v in entry.items():
            if k not in SOURCE_FIELDS and k not in ("supporting_websites", "items") and v not in (None, "", [], {}):
                src.extra.setdefault(k, v)
        return src

    def _retrieved(self, blobs: list[Any], sources: dict[str, Source]) -> None:
        for path, v in walk_paths(blobs):
            if path and path[-1] == "search_result_groups" and isinstance(v, list):
                parent = next((p for p in reversed(path[:-1]) if isinstance(p, str) and p != "v"), "")
                where = SRG_LOCATIONS.get(parent, parent)
                owner = blobs[path[0]] if isinstance(path[0], int) and isinstance(blobs[path[0]], dict) else {}
                owner = owner.get("message", owner)  # blobs hold bare messages or {"message": ...} envelopes
                if where == "search_results" and isinstance(owner, dict) and (owner.get("author") or {}).get("role") == "assistant":
                    where = "answer_sources"  # attached to the final answer: what the "Sources" button lists
                for group in v:
                    for entry in (group or {}).get("entries") or []:
                        if isinstance(entry, dict) and isinstance(entry.get("url"), str):
                            src = self._upsert(sources, entry, where)
                            src.retrieved = True
                            found = src.extra.setdefault("found_in", [])
                            if where not in found:
                                found.append(where)
            elif isinstance(v, dict) and v.get("type") == "search_result" and isinstance(v.get("url"), str):
                self._upsert(sources, v, "search_result").retrieved = True

    def _cited(self, messages: list[dict[str, Any]], blobs: list[Any], sources: dict[str, Source]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for k, v, _ in walk(blobs):
            if k in CITATION_KEYS and isinstance(v, list):
                refs.extend(r for r in v if isinstance(r, dict))
        by_ref: dict[str, list[Source]] = {}
        for s in sources.values():
            for rk in s.ref_ids:
                by_ref.setdefault(rk, []).append(s)

        seen_spans = set()
        for ref in refs:
            rtype = ref.get("type", "")
            span = {
                "type": rtype,
                "start": ref.get("start_idx", ref.get("start_ix")),
                "end": ref.get("end_idx", ref.get("end_ix")),
                "matched_text": ref.get("matched_text", ""),
            }
            span_id = json.dumps(span, sort_keys=True)
            if span_id in seen_spans:
                continue
            seen_spans.add(span_id)
            footnote = rtype == "sources_footnote"
            targets: list[Source] = []
            location = "sources_footnote" if footnote else "citation"
            for kk, vv, _ in walk(ref):
                if not (isinstance(vv, dict) and isinstance(vv.get("url"), str)) or kk == "safe_urls":
                    continue
                if kk == "supporting_websites":  # runner-up pages for this claim: shown nowhere, not cited
                    sup = self._upsert(sources, vv, "supporting_websites")
                    sup.supporting = True
                    if span["matched_text"].strip() and span["matched_text"] not in sup.extra.setdefault("supports", []):
                        sup.extra["supports"].append(span["matched_text"])
                else:
                    targets.append(self._upsert(sources, vv, location))
            if isinstance(ref.get("url"), str):
                targets.append(self._upsert(sources, ref, location))
            if isinstance(ref.get("metadata"), dict) and isinstance(ref["metadata"].get("url"), str):
                targets.append(self._upsert(sources, {**ref["metadata"], "snippet": ref["metadata"].get("text", "")}, location))
            for rk in REF_RE.findall(span["matched_text"] or ""):
                targets.extend(by_ref.get(f"turn{rk[0]}{rk[1]}{rk[2]}", []))
            for t in {id(t): t for t in targets}.values():
                if footnote:
                    t.extra["in_sources_footnote"] = True
                else:
                    t.cited = True
                    t.cited_spans.append(span)

        # Newer answers are generative-UI markup: citations are inline tags in the answer text,
        # e.g. <Cite refs={["turn972733youtube14"]}/>, while metadata.content_references stays empty.
        # <Link url="https://..." title="..."/> is a link the model wrote itself (a vendor homepage, a
        # pricing page), usually not a search result, so it marks the source `linked`, not `cited`.
        # 2026-10: answers also link retrieved results with plain markdown, e.g. [Watch on YouTube](https://youtube...)
        # with no <Cite> at all; those are `linked` too.
        for m in _answer_messages(messages):
            text = _message_text(m)
            for mm in CITE_TAG_RE.finditer(text):
                for rk in REF_RE.finditer(mm.group(1)):
                    for t in by_ref.get(rk.group(0), []):
                        t.cited = True
                        t.cited_spans.append({"type": "cite_tag", "start": mm.start(), "end": mm.end(), "matched_text": rk.group(0)})
            for mm in [*LINK_TAG_RE.finditer(text), *MD_LINK_RE.finditer(text)]:
                t = self._upsert(sources, {"url": html.unescape(mm.group(1))})
                t.linked = True
                t.extra.setdefault("link_spans", []).append({"start": mm.start(), "end": mm.end()})
        return refs

    @staticmethod
    def _answer(messages: list[dict[str, Any]], refs: list[dict[str, Any]]) -> str:
        answers = [_message_text(m) for m in _answer_messages(messages)]
        answer = answers[-1] if answers else ""
        for ref in refs:
            mt = ref.get("matched_text")
            if isinstance(mt, str) and mt.strip():  # sources_footnote uses matched_text " "
                alt = ref.get("alt")
                answer = answer.replace(mt, alt if isinstance(alt, str) else "")
        return PUA_RE.sub("", answer).strip()

    @staticmethod
    def _reasoning(messages: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for m in messages:
            content = m.get("content") or {}
            ctype = content.get("content_type")
            if ctype == "thoughts":
                for t in content.get("thoughts") or []:
                    if isinstance(t, dict):
                        text = "\n".join(x for x in (t.get("summary"), t.get("content")) if isinstance(x, str) and x)
                        if text:
                            out.append(text)
            elif ctype == "reasoning_recap" and isinstance(content.get("content"), str):
                out.append(content["content"])
            elif m.get("channel") == "analysis" and ctype == "text":
                text = _message_text(m)
                if text:
                    out.append(text)
        return out

    @staticmethod
    def _products(blobs: list[Any], answer: str) -> list[dict[str, Any]]:
        by_ref: dict[str, dict[str, Any]] = {}
        for _, v, _ in walk(blobs):
            if not (isinstance(v, dict) and isinstance(v.get("cite"), str) and PRODUCT_REF_RE.fullmatch(v["cite"])
                    and isinstance(v.get("title"), str)):
                continue
            p = by_ref.setdefault(v["cite"], {"ref": v["cite"]})
            for f in PRODUCT_FIELDS:
                if f not in p and v.get(f) not in (None, "", [], {}):
                    p[f] = v[f]
            binding = (v.get("__internal") or {}).get("model_data_binding_fields_v1") or {}
            url = v.get("url") or binding.get("product_page")
            if url and "url" not in p:
                p["url"] = url
            images = v.get("image_urls")
            if isinstance(images, list) and images and "image_url" not in p:
                p["image_url"] = images[0]
            if isinstance(v.get("offers"), list) and v["offers"]:
                p["offers"] = len(v["offers"])
        for p in by_ref.values():
            p["in_answer"] = p["ref"] in answer
        return list(by_ref.values())

    @staticmethod
    def _insights(messages: list[dict[str, Any]], blobs: list[Any]) -> dict[str, Any]:
        """Fields the API sends but the page does not show (or shows only partly)."""
        out: dict[str, Any] = {}
        for m in reversed(_answer_messages(messages)):
            md = m.get("metadata") or {}
            adjustments = [a for a in md.get("model_adjustments") or [] if isinstance(a, str)]
            requested, resolved = md.get("default_model_slug"), md.get("resolved_model_slug") or md.get("model_slug")
            if adjustments or (requested and resolved and requested != resolved):
                out["model_routing"] = {"requested": requested, "resolved": resolved, "adjustments": adjustments}
            break

        hidden, empty = [], 0
        for m in messages:
            md = m.get("metadata") or {}
            if md.get("is_visually_hidden_from_conversation"):
                if not _message_text(m).strip() and not md.get("exclusive_key") and not md.get("message_source"):
                    empty += 1  # system/context placeholders: the server strips their contents
                    continue
                item = {"role": (m.get("author") or {}).get("role"), "content_type": (m.get("content") or {}).get("content_type")}
                for k in ("exclusive_key", "message_source"):
                    if md.get(k):
                        item[k] = md[k]
                if _message_text(m).strip():
                    item["text"] = _message_text(m)[:500]
                hidden.append(item)
        if hidden:
            out["hidden_messages"] = hidden
        if empty:
            out["hidden_empty_messages"] = empty

        # generative-UI answers: nodes the renderer dropped, plain-text fallback, injected surveys
        omitted: Counter[tuple] = Counter()
        for k, v, _ in walk(blobs):
            if k == "recoveryDiagnostics" and isinstance(v, list):
                for d in v:
                    if isinstance(d, dict):
                        omitted[(d.get("component"), d.get("prop"), d.get("action"), d.get("code"))] += 1
            elif k == "fallbackMarkdown" and isinstance(v, str) and v.strip():
                out["fallback_markdown"] = v
            elif k == "user_signal_metadata" and isinstance(v, dict):
                for name, sig in v.items():
                    payload = (sig.get("payload") if isinstance(sig, dict) else None) or {}
                    if isinstance(payload.get("question"), str):
                        options = [o.get("label") for o in payload.get("options") or [] if isinstance(o, dict)]
                        survey = {"name": name, "question": payload["question"], "options": options}
                        if survey not in out.setdefault("surveys", []):
                            out["surveys"].append(survey)
            elif k == "page_info" and isinstance(v, dict) and v.get("has_previous_page"):
                out["history_truncated"] = True
        if omitted:
            out["render_omissions"] = [{"component": c, "prop": p, "action": a, "code": code, "count": n}
                                       for (c, p, a, code), n in omitted.items()]

        entities, images = [], []
        for m in _answer_messages(messages):
            for mm in UI_TAG_RE.finditer(_message_text(m)):
                attrs = {a.group(1): a.group(2) if a.group(2) is not None else "{" + a.group(3) + "}"
                         for a in UI_ATTR_RE.finditer(mm.group(2))}
                (entities if mm.group(1) == "Entity" else images).append(attrs)
        if entities:
            out["entities"] = entities
        if images:
            out["image_queries"] = images
        return out
