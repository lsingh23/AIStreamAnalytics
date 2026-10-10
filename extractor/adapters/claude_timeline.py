"""claude.ai StreamTimeline decoder (Connect-RPC, application/connect+proto).

Since ~2026-10 claude.ai sends a message with ConversationService/PerformAction and streams the
answer from ConversationService/StreamTimeline. Each envelope is a protobuf message (no schema, so
fields are numbers); field meanings below were mapped from a real capture:

  env["1"]["6"]                      heartbeat (empty)
  env["1"]["10"]["1"]                event id ("<ms>-<n>")
  env["1"]["1"]                      upsert of timeline entities:
      ["2"]  conversation  {1: uuid, 2: title, 6: {2: model}, 7: current leaf message uuid}
      ["3"]  message       {1: uuid, 2: conversation, 3: sender (1 human, 2 assistant), 11: parent}
      ["4"]  display group {1: dgrp id, 2: message uuid, 5: label ("Thinking", "Searched the web")}  (one or a list)
      ["5"]  content block {1: cblk id, 2: dgrp id, ...} -- latest full snapshot of the block:
               text:      9: text, 11: citations [{1: id, 2: title, 3: url, 4: start, 5: end,
                                                   10: {1: domain, 2: favicon, 3: site name}}]
               thinking:  5: "Thought for 3s", 12: {1: summary}
               tool call: 40: tool ("WebSearch"), 15: label ("Web search"), 16: query, 41: toolu id,
                          13: {4: {1: [{1: title, 2: domain, 3: url, 5: favicon, 6: site name}]}}  (results)
  env["1"]["2"]                      streaming delta {1: cblk id, 2: text chunk | 6: {1: thinking chunk}}
  final envelope (flag 0x02)         {"metadata": {"Stream-Close-Reason": [...]}}

to_blocks() rewrites this into Messages-API-shaped blocks (tool_use / tool_result / text+citations /
thinking) so ClaudeAdapter can treat both transports the same way.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..connect import decode_message, iter_envelopes

SENDER_HUMAN, SENDER_ASSISTANT = 1, 2


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v] if v is not None else []


class Timeline:
    def __init__(self, raw: bytes) -> None:
        self.conversation: dict[str, Any] = {}
        self.messages: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, dict[str, Any]] = {}
        self.blocks: dict[str, dict[str, Any]] = {}  # insertion order = first appearance
        self.deltas: dict[str, list[str]] = {}
        self.thinking: dict[str, list[str]] = {}
        self.events = self.heartbeats = 0
        self.closed: Any = None
        self.unknown: Counter[str] = Counter()
        for env in iter_envelopes(raw):
            msg = decode_message(env.data)
            if env.end_of_stream:
                self.closed = msg.get("metadata", msg) if isinstance(msg, dict) else msg
                continue
            self._feed(msg.get("1") if isinstance(msg, dict) else None)

    def _feed(self, ev: Any) -> None:
        if not isinstance(ev, dict):
            return
        if "6" in ev and len(ev) == 1:
            self.heartbeats += 1
            return
        self.events += 1
        up, delta = ev.get("1"), ev.get("2")
        if isinstance(up, dict):
            if isinstance(up.get("2"), dict):
                self.conversation.update(up["2"])
            for m in _as_list(up.get("3")):
                if isinstance(m, dict) and isinstance(m.get("1"), str):
                    self.messages.setdefault(m["1"], {}).update(m)
            for g in _as_list(up.get("4")):
                if isinstance(g, dict) and isinstance(g.get("1"), str):
                    self.groups.setdefault(g["1"], {}).update(g)
            for b in _as_list(up.get("5")):
                if isinstance(b, dict) and isinstance(b.get("1"), str):
                    self.blocks.setdefault(b["1"], {}).update(b)
            self.unknown.update(k for k in up if k not in ("1", "2", "3", "4", "5", "11", "12", "20", "28"))
        if isinstance(delta, dict) and isinstance(delta.get("1"), str):
            if isinstance(delta.get("2"), str):
                self.deltas.setdefault(delta["1"], []).append(delta["2"])
            elif isinstance(delta.get("6"), dict) and isinstance(delta["6"].get("1"), str):
                self.thinking.setdefault(delta["1"], []).append(delta["6"]["1"])

    def sender(self, block: dict[str, Any]) -> int | None:
        msg = (self.groups.get(block.get("2", "")) or {}).get("2")
        return (self.messages.get(msg) or {}).get("3") if msg else None

    def text_of(self, cid: str, block: dict[str, Any]) -> str:
        snap = block.get("9") if isinstance(block.get("9"), str) else ""
        streamed = "".join(self.deltas.get(cid, []))
        return streamed if len(streamed) > len(snap) and streamed.startswith(snap[:20]) else snap

    @property
    def prompt(self) -> str:
        for cid, b in self.blocks.items():
            if self.sender(b) == SENDER_HUMAN and isinstance(b.get("9"), str):
                return b["9"]
        return ""

    def to_blocks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cid, b in self.blocks.items():
            if self.sender(b) == SENDER_HUMAN:
                continue
            if "16" in b or "40" in b or "41" in b:  # tool call (+ its results in the same block)
                tool_id = b.get("41") or cid
                name = {"WebSearch": "web_search", "WebFetch": "web_fetch"}.get(b.get("40"), str(b.get("40") or b.get("15") or "tool"))
                inp = {"query": b["16"]} if isinstance(b.get("16"), str) else {}
                out.append({"type": "tool_use", "id": tool_id, "name": name, "input": inp, "label": b.get("5")})
                results = []
                for r in _as_list((((b.get("13") or {}).get("4") or {}).get("1"))):
                    if isinstance(r, dict) and isinstance(r.get("3"), str):
                        results.append({"type": "knowledge", "url": r["3"], "title": r.get("1", ""),
                                        "metadata": {"site_domain": r.get("2", ""), "site_name": r.get("6", ""), "favicon_url": r.get("5", "")}})
                out.append({"type": "tool_result", "tool_use_id": tool_id, "name": name, "content": results})
            elif "9" in b or cid in self.deltas:
                text = self.text_of(cid, b)
                cites = []
                for c in _as_list(b.get("11")):
                    if not isinstance(c, dict) or not isinstance(c.get("3"), str):
                        continue
                    src = c.get("10") if isinstance(c.get("10"), dict) else {}
                    cites.append({"type": "claude_citation", "uuid": c.get("1"), "url": c["3"], "title": c.get("2", ""),
                                  "start_index": c.get("4"), "end_index": c.get("5"),
                                  "metadata": {"site_domain": src.get("1", ""), "site_name": src.get("3", "")}})
                out.append({"type": "text", "text": text, "citations": cites})
            elif "12" in b or cid in self.thinking:
                summary = (b.get("12") or {}).get("1") if isinstance(b.get("12"), dict) else None
                out.append({"type": "thinking", "thinking": "".join(self.thinking.get(cid, [])) or summary or "",
                            "label": b.get("5")})
        return out

    def diagnostics(self) -> dict[str, Any]:
        return {
            "input": "connect_timeline",
            "events": self.events,
            "heartbeats": self.heartbeats,
            "complete": self.closed is not None,
            "close": self.closed,
            "groups": [g.get("5") for g in self.groups.values() if g.get("5")],
            "unknown_upsert_fields": dict(self.unknown),
        }
