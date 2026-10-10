"""Generic Server-Sent Events parsing with transparent gzip/base64 handling."""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import zlib
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Event:
    event: str | None
    data: str


def _maybe_decompress(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError, zlib.error):
            # truncated gzip stream: decompress as much as possible
            return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    return raw


def iter_events(raw: bytes | str) -> Iterator[Event]:
    """Yield SSE events (event name + joined data lines) from a raw response body."""
    if isinstance(raw, bytes):
        raw = _maybe_decompress(raw).decode("utf-8", "replace")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    for block in text.split("\n\n"):
        event, data = None, []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        if data or event:
            yield Event(event, "\n".join(data))


def decode_data(data: str) -> Any:
    """Decode an SSE data payload: JSON, or base64(gzip(JSON)) if the stream is compressed per event.

    Returns None for the terminal "[DONE]" marker and the raw string if nothing else works.
    """
    data = data.strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        pass
    try:
        blob = base64.b64decode(data, validate=True)
        return json.loads(_maybe_decompress(blob).decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError, OSError, zlib.error):
        return data
