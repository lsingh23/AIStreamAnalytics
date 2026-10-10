"""Connect-RPC streaming bodies (used by claude.ai's /claudeai-rpc/... ConversationService endpoints).

A streaming Connect response is a sequence of envelopes:
    1 byte flags (0x01 = compressed, 0x02 = end-of-stream trailer) + 4 bytes big-endian length + message
Messages are JSON (application/connect+json) or protobuf (application/connect+proto). Protobuf is
decoded schema-less into {"<field number>": value or [values]} so it can at least be searched by
content (URLs, strings) until the field meanings are mapped.
"""
from __future__ import annotations

import base64
import gzip
import json
import zlib
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Envelope:
    flags: int
    data: bytes

    @property
    def end_of_stream(self) -> bool:
        return bool(self.flags & 0x02)


def is_connect(raw: bytes) -> bool:
    if len(raw) < 5 or raw[0] not in (0, 1, 2, 3):
        return False
    return 5 + int.from_bytes(raw[1:5], "big") <= len(raw)


def iter_envelopes(raw: bytes) -> Iterator[Envelope]:
    pos = 0
    while pos + 5 <= len(raw):
        flags, size = raw[pos], int.from_bytes(raw[pos + 1:pos + 5], "big")
        data = raw[pos + 5:pos + 5 + size]
        pos += 5 + size
        if flags & 0x01:
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError, zlib.error):
                pass
        yield Envelope(flags, data)
        if len(data) < size:  # truncated capture
            break


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def decode_proto(buf: bytes, depth: int = 0) -> dict[str, Any]:
    """Schema-less protobuf decode. Raises ValueError if buf is not a valid message."""
    out: dict[str, Any] = {}
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if field == 0:
            raise ValueError("field 0")
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = int.from_bytes(buf[pos:pos + 8], "little"), pos + 8
        elif wire == 5:
            value, pos = int.from_bytes(buf[pos:pos + 4], "little"), pos + 4
        elif wire == 2:
            size, pos = _varint(buf, pos)
            if pos + size > len(buf):
                raise ValueError("length overflows")
            value = _decode_bytes(buf[pos:pos + size], depth)
            pos += size
        else:
            raise ValueError(f"wire type {wire}")
        prev = out.get(str(field))
        out[str(field)] = value if prev is None else (prev + [value] if isinstance(prev, list) else [prev, value])
    return out


def _decode_bytes(b: bytes, depth: int) -> Any:
    # "\n\x05hello" is both printable-ish text and a nested message {1: "hello"}: if it starts with a
    # control byte, a nested message is the likelier reading.
    if b[:1] and b[0] < 0x20 and depth < 40:
        try:
            return decode_proto(b, depth + 1)
        except ValueError:
            pass
    try:
        text = b.decode("utf-8")
        if text and all(c.isprintable() or c in "\n\r\t" for c in text):
            if text[:1] in "{[":
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            return text
    except UnicodeDecodeError:
        pass
    if depth < 40 and b:
        try:
            return decode_proto(b, depth + 1)
        except ValueError:
            pass
    return {"_bytes": base64.b64encode(b).decode()}


def decode_message(data: bytes) -> Any:
    stripped = data.lstrip()
    if stripped[:1] in (b"{", b"["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    try:
        return decode_proto(data)
    except ValueError:
        return {"_bytes": base64.b64encode(data).decode()}


def messages(raw: bytes) -> list[Any]:
    return [decode_message(e.data) for e in iter_envelopes(raw)]
