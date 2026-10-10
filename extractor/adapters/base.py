from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..connect import decode_message, is_connect, iter_envelopes
from ..schema import Result
from ..sse import decode_data, iter_events


class Adapter:
    """One adapter per AI platform: raw captured bytes in, normalized Result out."""

    name = "base"

    @classmethod
    def detect(cls, raw: bytes) -> bool:
        return False

    def parse(self, raw: bytes, meta: dict[str, Any] | None = None) -> Result:
        raise NotImplementedError

    def inspect(self, raw: bytes) -> tuple[dict[str, Any], list[Any]]:
        """(summary counters, decoded objects to scan for interesting keys). Platform-agnostic fallback."""
        if is_connect(raw):
            envs = list(iter_envelopes(raw))
            objs = [decode_message(e.data) for e in envs]
            return {"input": "connect-rpc envelopes", "envelopes": len(envs),
                    "end_of_stream": any(e.end_of_stream for e in envs),
                    "json messages": sum(e.data.lstrip()[:1] == b"{" for e in envs)}, objs
        stripped = raw.lstrip()
        if stripped[:1] in (b"{", b"["):
            try:
                return {"input": "json document"}, [json.loads(stripped)]
            except json.JSONDecodeError:
                pass
        payloads = [decode_data(ev.data) for ev in iter_events(raw)]
        objs = [p for p in payloads if isinstance(p, (dict, list))]
        summary = {
            "decoded payloads": len(objs),
            "undecoded payloads": sum(isinstance(p, str) for p in payloads),
            "payload types": dict(Counter(str(p.get("type", "?")) for p in objs if isinstance(p, dict))),
        }
        return summary, objs
