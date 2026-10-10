"""Pull answer streams and stored conversations out of a browser HAR export (DevTools > Network > Save all as HAR).

Writes the same files the userscript downloads (<platform>_<ts>.sse / .meta.json / .conversation.json), so a
HAR recorded without the userscript can go through the normal parse/dashboard path. Only response bodies and
request URL/body are written: a HAR also holds cookies and auth headers, which are never copied.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

# Mirrors the PLATFORMS table in capture/ai_capture.user.js; keep the two in sync.
ROUTES = {
    "chatgpt": (re.compile(r"(^|\.)chatgpt\.com$"), re.compile(r"/backend-api/(f/)?conversation(\?|$)"),
                re.compile(r"/backend-api/conversations?/[0-9a-f-]{36}(\?|$)")),
    "claude": (re.compile(r"(^|\.)claude\.ai$"),
               re.compile(r"/chat_conversations/[^/?]+/(retry_)?completion(\?|$)|ConversationService/(StreamTimeline|PerformAction)"),
               re.compile(r"/chat_conversations/[0-9a-f-]{36}(\?|$)")),
    "perplexity": (re.compile(r"(^|\.)perplexity\.ai$"), re.compile(r"/rest/sse/"), re.compile(r"/rest/thread/(?!list)[^/?]+")),
    "gemini": (re.compile(r"(^|\.)gemini\.google\.com$"), re.compile(r"StreamGenerate"),
               re.compile(r"batchexecute\?.*rpcids=[^&]*hNvQHb")),
}
STREAM_TYPE_RE = re.compile(r"event-stream|ndjson|x-protobuf|connect\+")


def _stamp(started: str) -> str:
    """HAR startedDateTime -> the userscript's filename timestamp (UTC, ':' and '.' replaced by '-')."""
    try:
        dt = datetime.fromisoformat(started.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H-%M-%S-") + f"{dt.microsecond // 1000:03d}Z"


def _body(content: dict[str, Any]) -> bytes | None:
    text = content.get("text")
    if not isinstance(text, str) or not text:
        return None
    return base64.b64decode(text) if content.get("encoding") == "base64" else text.encode()


def iter_captures(har: dict[str, Any]) -> Iterator[tuple[str, str, str, bytes, dict[str, Any]]]:
    """Yield (platform, kind, ts, body, meta) for every answer stream ('stream') or stored conversation ('stored')."""
    for e in (har.get("log") or {}).get("entries") or []:
        req, res = e.get("request") or {}, e.get("response") or {}
        url, method = req.get("url", ""), req.get("method", "GET").upper()
        host = urlsplit(url).hostname or ""
        for platform, (host_re, stream_re, stored_re) in ROUTES.items():
            if not host_re.search(host):
                continue
            content = res.get("content") or {}
            ctype = content.get("mimeType") or ""
            if (method == "POST" and stream_re.search(url)) or STREAM_TYPE_RE.search(ctype):
                kind = "stream"
            elif stored_re.search(url) and (method == "GET" or platform == "gemini"):
                kind = "stored"
            else:
                break
            body = _body(content)
            if body:
                post = (req.get("postData") or {}).get("text")
                try:
                    post = json.loads(post) if post else None
                except json.JSONDecodeError:
                    pass
                meta = {"platform": platform, "captured_at": e.get("startedDateTime", ""), "source": "har",
                        "request": {"url": url, "method": method, "body": post},
                        "response": {"status": res.get("status"), "bytes": len(body)}}
                yield platform, kind, _stamp(e.get("startedDateTime", "")), body, meta
            break


def extract(har_path: Path, out_dir: Path) -> list[Path]:
    """Write every capture found in the HAR into out_dir; returns the written body files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for platform, kind, ts, body, meta in iter_captures(json.loads(har_path.read_text(encoding="utf-8"))):
        stem = f"{platform}_{ts}"
        target = out_dir / (f"{stem}.sse" if kind == "stream" else f"{stem}.conversation.json")
        target.write_bytes(body)
        (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        written.append(target)
    return written
