"""Hand-made streams mimicking Claude, Perplexity and Gemini.  Run: python3 -m unittest discover tests

Perplexity and Gemini fixtures mirror real captures from 2026-10-10 (data/raw); the Claude
timeline fixture mirrors a real StreamTimeline capture; the Claude SSE fixture follows the public
Messages API docs (the older /completion transport).
"""
import json
import unittest

from extractor.adapters import get_adapter
from extractor.adapters.claude import ClaudeAdapter
from extractor.adapters.gemini import GeminiAdapter
from extractor.adapters.perplexity import PerplexityAdapter

A = "https://asana.com/resources/project-management-tools"
B = "https://www.zapier.com/blog/best-project-management-software/?utm_source=x"
C = "https://www.youtube.com/watch?v=pm123"
D = "https://monday.com/blog/agency"


def sse(*payloads):
    return "".join(f"event: {p['type']}\ndata: {json.dumps(p)}\n\n" for p in payloads).encode()


def start(i, block):
    return {"type": "content_block_start", "index": i, "content_block": block}


def delta(i, d):
    return {"type": "content_block_delta", "index": i, "delta": d}


def stop(i):
    return {"type": "content_block_stop", "index": i}


CLAUDE_EVENTS = [
    {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-5-5", "content": []}},
    start(0, {"type": "thinking", "thinking": ""}),
    delta(0, {"type": "thinking_delta", "thinking": "Need current tools."}),
    stop(0),
    start(1, {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {}}),
    delta(1, {"type": "input_json_delta", "partial_json": '{"query": "best project man'}),
    delta(1, {"type": "input_json_delta", "partial_json": 'agement tools small agencies"}'}),
    stop(1),
    start(2, {"type": "tool_result", "tool_use_id": "toolu_1", "name": "web_search", "content": [
        {"type": "knowledge", "title": "PM tools", "url": A, "text": "Asana is ...",
         "metadata": {"type": "webpage_metadata", "site_domain": "asana.com", "site_name": "Asana"}},
        {"type": "knowledge", "title": "Best PM software", "url": B, "text": "Zapier tested ..."},
        {"type": "knowledge", "title": "PM video", "url": C, "text": "In this video ..."},
    ]}),
    stop(2),
    start(3, {"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search", "input": {}}),
    delta(3, {"type": "input_json_delta", "partial_json": '{"query": "agency project management software 2026"}'}),
    stop(3),
    start(4, {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_2", "content": [
        {"type": "web_search_result", "title": "Agency PM", "url": D, "page_age": "March 3, 2026", "encrypted_content": "xx"},
        {"type": "web_search_result", "title": "Best PM software", "url": B.split("?")[0]},
    ]}),
    stop(4),
    start(5, {"type": "text", "text": "", "citations": []}),
    delta(5, {"type": "text_delta", "text": "Try "}),
    delta(5, {"type": "citation_start_delta", "citation": {"uuid": "c1", "url": A, "title": "PM tools"}}),
    delta(5, {"type": "text_delta", "text": "Asana"}),
    delta(5, {"type": "citation_end_delta", "citation_uuid": "c1"}),
    delta(5, {"type": "text_delta", "text": " or Monday."}),
    delta(5, {"type": "citations_delta", "citation": {"type": "web_search_result_location", "url": D,
                                                       "title": "Agency PM", "cited_text": "Monday for agencies"}}),
    stop(5),
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    {"type": "message_stop"},
]
CLAUDE_META = {"captured_at": "2026-10-10T10:00:00Z", "request": {
    "url": "https://claude.ai/api/organizations/o/chat_conversations/11111111-2222-3333-4444-555555555555/completion",
    "body": {"prompt": "best project management tools for small agencies"}}}


class ClaudeTest(unittest.TestCase):
    def check(self, r, complete=True):
        self.assertEqual(r["fanout_queries"], ["best project management tools small agencies", "agency project management software 2026"])
        by = {s["url"]: s for s in r["sources"]}
        self.assertEqual(len(r["sources"]), 4)  # B deduped across both searches
        self.assertTrue(by[A]["cited"] and by[A]["retrieved"])
        self.assertEqual(by[A]["attribution"], "Asana")
        self.assertEqual(by[A]["cited_spans"][0]["matched_text"], "Asana")
        self.assertTrue(by[D]["cited"])
        self.assertEqual(by[D]["pub_date"], "March 3, 2026")
        self.assertFalse(by[B]["cited"])
        self.assertEqual(len(by[B]["extra"]["queries"]), 2)
        self.assertEqual(by[C]["type"], "youtube")
        self.assertEqual(r["stats"]["retrieved_not_cited"], 2)
        self.assertEqual(r["answer"], "Try Asana or Monday.")
        self.assertEqual(r["reasoning"], ["Need current tools."])
        self.assertEqual(r["diagnostics"].get("unknown_deltas", {}), {})
        self.assertEqual(r["diagnostics"].get("complete", True), complete)

    def test_stream(self):
        raw = sse(*CLAUDE_EVENTS)
        self.assertIsInstance(get_adapter("auto", raw), ClaudeAdapter)
        r = ClaudeAdapter().parse(raw, CLAUDE_META).to_dict()
        self.check(r)
        self.assertEqual(r["model"], "claude-opus-5-5")
        self.assertEqual(r["prompt"], "best project management tools for small agencies")
        self.assertEqual(r["conversation_id"], "11111111-2222-3333-4444-555555555555")

    def test_truncated_stream(self):
        r = ClaudeAdapter().parse(sse(*CLAUDE_EVENTS[:-2])).to_dict()
        self.check(r, complete=False)

    def test_stored_conversation(self):
        blocks = ClaudeAdapter.assemble(sse(*CLAUDE_EVENTS)).blocks
        doc = {"uuid": "conv-1", "model": "claude-opus-5-5", "chat_messages": [
            {"index": 0, "sender": "human", "text": "best project management tools for small agencies", "content": []},
            {"index": 1, "sender": "assistant", "content": blocks},
        ]}
        raw = json.dumps(doc).encode()
        self.assertIsInstance(get_adapter("auto", raw), ClaudeAdapter)
        r = ClaudeAdapter().parse(raw).to_dict()
        self.check(r)
        self.assertEqual(r["prompt"], "best project management tools for small agencies")
        self.assertEqual(r["conversation_id"], "conv-1")


def pb(fields):
    """Tiny protobuf encoder for fixtures: {field: str | int | dict | list}."""
    def varint(n):
        out = b""
        while True:
            b, n = n & 0x7F, n >> 7
            out += bytes([b | (0x80 if n else 0)])
            if not n:
                return out
    out = b""
    for f, v in fields.items():
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, int):
                out += varint(int(f) << 3) + varint(item)
            else:
                data = pb(item) if isinstance(item, dict) else item.encode()
                out += varint(int(f) << 3 | 2) + varint(len(data)) + data
    return out


def envelope(msg, flags=0):
    data = json.dumps(msg).encode() if flags & 2 else pb(msg)
    return bytes([flags]) + len(data).to_bytes(4, "big") + data


def up(**entities):
    return {"1": {"10": {"1": "1-0"}, "11": 1, "1": entities}}


TL_ANSWER = "Teamwork suits agencies. Trello is simplest."
TL_ENVELOPES = [
    {"1": {"6": b"".decode()}},  # heartbeat
    up(**{"2": {"1": "conv-1", "2": "PM tools", "6": {"2": "claude-sonnet-5-5"}, "7": "msg-ai"},
          "3": {"1": "msg-user", "2": "conv-1", "3": 1}}),
    up(**{"3": {"1": "msg-ai", "2": "conv-1", "3": 2, "11": "msg-user"},
          "4": [{"1": "dgrp-u", "2": "msg-user", "4": 1}, {"1": "dgrp-s", "2": "msg-ai", "4": 2, "5": "Searching the web"},
                {"1": "dgrp-t", "2": "msg-ai", "4": 1}]}),
    up(**{"5": {"1": "cblk-p", "2": "dgrp-u", "4": 1, "9": "best pm tools for agencies"}}),
    up(**{"5": {"1": "cblk-th", "2": "dgrp-s", "5": "Thought for 2s", "12": {"1": "Finding PM tools."}}}),
    up(**{"5": {"1": "cblk-ws", "2": "dgrp-s", "5": "Searched the web", "15": "Web search", "16": "pm software agencies 2026",
                "40": "WebSearch", "41": "toolu_1", "13": {"4": {"1": [
                    {"1": "PM tools", "2": "asana.com", "3": A, "6": "asana"},
                    {"1": "Agency PM", "2": "monday.com", "3": D, "6": "monday"}]}}}}),
    up(**{"5": {"1": "cblk-a", "2": "dgrp-t", "4": 1, "9": "Teamwork"}}),
    {"1": {"10": {"1": "2-0"}, "11": 1, "2": {"1": "cblk-a", "2": " suits agencies."}}},
    {"1": {"10": {"1": "3-0"}, "11": 1, "2": {"1": "cblk-a", "2": " Trello is simplest."}}},
    up(**{"5": {"1": "cblk-a", "2": "dgrp-t", "4": 1, "9": TL_ANSWER, "11": [
        {"1": "c1", "2": "PM tools", "3": A, "4": 24, "5": 24, "10": {"1": "asana.com", "3": "asana"}}]}}),
]


class ClaudeTimelineTest(unittest.TestCase):
    def raw(self, closed=True):
        out = b"".join(envelope(e) for e in TL_ENVELOPES)
        return out + (envelope({"metadata": {"Stream-Close-Reason": ["cadence"]}}, flags=2) if closed else b"")

    def check(self, r, complete):
        self.assertEqual(r["prompt"], "best pm tools for agencies")
        self.assertEqual((r["model"], r["conversation_id"]), ("claude-sonnet-5-5", "conv-1"))
        self.assertEqual(r["fanout_queries"], ["pm software agencies 2026"])
        by = {s["url"]: s for s in r["sources"]}
        self.assertTrue(by[A]["cited"] and by[A]["retrieved"])
        self.assertFalse(by[D]["cited"])
        self.assertEqual(by[D]["attribution"], "monday")
        self.assertEqual(by[A]["cited_spans"][0]["matched_text"], "Teamwork suits agencies.")
        self.assertEqual(r["answer"], TL_ANSWER)
        self.assertEqual(r["reasoning"], ["Finding PM tools."])
        self.assertEqual(r["diagnostics"]["complete"], complete)

    def test_stream(self):
        raw = self.raw()
        self.assertIsInstance(get_adapter("auto", raw), ClaudeAdapter)
        self.check(ClaudeAdapter().parse(raw).to_dict(), complete=True)

    def test_without_close_trailer(self):
        self.check(ClaudeAdapter().parse(self.raw(closed=False)).to_dict(), complete=False)


def px_event(status, *blocks):
    return {"backend_uuid": "u1", "thread_url_slug": "best-pm-abc", "display_model": "turbo",
            "query_str": "best pm tools", "status": status, "blocks": list(blocks)}


def diff(usage, field, *patches):
    return {"intended_usage": usage, "diff_block": {"field": field, "patches": [
        {"op": op, "path": path, "value": value} for op, path, value in patches]}}


PX_RESULTS = [{"name": "PM tools", "url": A, "snippet": "Asana is ...", "timestamp": "2026-01-02T00:00:00",
               "trust": {"name": "trusted", "description": "first-party"}},
              {"name": "Best PM", "url": B, "snippet": "Zapier tested ..."},
              {"name": "Agency PM", "url": D, "snippet": "Monday ..."}]
PX_STEPS = [{"title": "Searching the web", "items": [
    {"type": "WORKFLOW_ITEM_QUERIES", "payload": {"queries_payload": {"queries": ["best project management tools agencies"]}}},
    {"type": "WORKFLOW_ITEM_SOURCES", "payload": {"sources_payload": {"sources": PX_RESULTS[:2]}}}]}]
# real Perplexity streams are diffs (JSON patch) against earlier blocks; only the last event is a full snapshot
PX_EVENTS = [
    px_event("PENDING", {"intended_usage": "workflow_root", "workflow_block": {"steps": []}}),
    px_event("PENDING", diff("workflow_root", "workflow_block", ("add", "/steps/0", PX_STEPS[0])),
             diff("web_results", "web_result_block", ("replace", "", {"web_results": PX_RESULTS[:2]}))),
    px_event("PENDING", diff("workflow_root", "workflow_block", ("add", "/steps/1", {"title": "Comparing", "items": [
        {"type": "WORKFLOW_ITEM_QUERIES", "payload": {"queries_payload": {"queries": ["asana vs monday agencies"]}}},
        {"type": "WORKFLOW_ITEM_SOURCES", "payload": {"sources_payload": {"sources": PX_RESULTS[2:]}}}]})),
             diff("web_results", "web_result_block", ("add", "/web_results/2", PX_RESULTS[2]))),
    px_event("PENDING", diff("workflow_root", "workflow_block", ("add", "/steps/2", {"items": [
        {"type": "WORKFLOW_ITEM_TEXT", "payload": {"text_payload": {"text": "", "chunks": ["Use Asana"]}}}]})),
             diff("sources_answer_mode", "sources_mode_block", ("replace", "", {"rows": [
                 {"web_result": PX_RESULTS[0], "status": "SELECTED", "citation": 1},
                 {"web_result": PX_RESULTS[1], "status": "REVIEWED"},
                 {"web_result": PX_RESULTS[2], "status": "SELECTED", "citation": 2}]}))),
    px_event("PENDING", diff("workflow_root", "workflow_block",
                             ("add", "/steps/2/items/0/payload/text_payload/chunks/1", "[1] or Monday[2]."),
                             ("replace", "/steps/2/items/0/payload/text_payload/text", "Use Asana[1] or Monday[2]."))),
]


class PerplexityTest(unittest.TestCase):
    def check(self, r, complete):
        self.assertEqual(r["fanout_queries"], ["best project management tools agencies", "asana vs monday agencies"])
        by = {s["url"]: s for s in r["sources"]}
        self.assertTrue(by[A]["cited"] and by[D]["cited"])
        self.assertFalse(by[B]["cited"])
        self.assertEqual(by[B]["extra"]["status"], "REVIEWED")
        self.assertEqual(by[D]["extra"]["queries"], ["asana vs monday agencies"])
        self.assertEqual(by[A]["extra"]["trust"], "trusted")
        self.assertEqual(by[A]["pub_date"], "2026-01-02")
        self.assertEqual(by[A]["cited_spans"][0]["matched_text"], "[1]")
        self.assertEqual(r["stats"]["retrieved"], 3)
        self.assertEqual(r["answer"], "Use Asana[1] or Monday[2].")
        self.assertEqual((r["prompt"], r["model"], r["conversation_id"]), ("best pm tools", "turbo", "best-pm-abc"))
        self.assertEqual(r["diagnostics"]["complete"], complete)
        self.assertEqual(r["diagnostics"]["errors"], [])

    def test_diff_stream(self):
        raw = "".join(f"event: message\ndata: {json.dumps(e)}\n\n" for e in PX_EVENTS).encode()
        self.assertIsInstance(get_adapter("auto", raw), PerplexityAdapter)
        self.check(PerplexityAdapter().parse(raw).to_dict(), complete=False)

    def test_final_snapshot(self):
        from extractor.adapters.perplexity import BlockState
        st = BlockState()
        for e in PX_EVENTS:
            st.feed(e)
        final = px_event("COMPLETED", *[{"intended_usage": u, **b} for u, b in st.blocks.items()])
        raw = f"event: message\ndata: {json.dumps(final)}\n\n".encode()
        self.check(PerplexityAdapter().parse(raw).to_dict(), complete=True)


class GeminiTest(unittest.TestCase):
    def test_positions(self):
        answer = "Try Asana for agencies."
        retrieved = [["https://t0.gstatic.com/favicon", A, "PM tools", None, []],
                     ["https://t0.gstatic.com/favicon", D, "Agency PM", None, []]]
        citation = [["Asana", None, None, [[4, 9], [4, 9]]], [1],
                    [[A + "#:~:text=Asana%20is%20great", "PM tools", "https://fav", " Asana is great ", [], None, "Asana"]], "spp_1"]
        cand = [None] * 38
        cand[0], cand[1], cand[2] = "rc_1", [answer], [None, [citation]]
        cand[37] = [None, [[[], "Searched the web", "Google Search", None, [], "Searched the web", "", retrieved]]]
        inner = [None, ["c_1", "r_1"], None, None, [cand]]
        frame = json.dumps([["wrb.fr", None, json.dumps(inner)]])
        raw = f")]}}'\n\n{len(frame)}\n{frame}\n25\n[[\"di\",42]]\n".encode()
        self.assertIsInstance(get_adapter("auto", raw), GeminiAdapter)
        r = GeminiAdapter().parse(raw).to_dict()
        self.assertEqual(r["conversation_id"], "c_1")
        self.assertEqual(r["answer"], answer)
        by = {s["url"]: s for s in r["sources"]}
        self.assertEqual(set(by), {A, D})
        self.assertTrue(by[A]["cited"] and by[A]["retrieved"])
        self.assertFalse(by[D]["cited"])
        span = by[A]["cited_spans"][0]
        self.assertEqual(answer[span["start"]:span["end"]], "Asana")
        self.assertEqual(span["cited_text"], "Asana is great")
        self.assertEqual(by[A]["attribution"], "Asana")
        self.assertFalse(r["diagnostics"]["fallback_url_harvest"])


if __name__ == "__main__":
    unittest.main()
