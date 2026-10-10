# CLAUDE.md

Browser userscript captures raw AI-assistant answer streams; a stdlib-only Python package parses them into a normalized schema. See README.md for usage.

## Commands

- Tests: `python3 -m unittest discover tests`
- CLI: `python3 -m extractor {inspect,parse,render,serve} ...` (run from repo root)

## Architecture

- `capture/ai_capture.user.js` – all per-platform URL patterns live in its `PLATFORMS` table. Output filenames are `<platform>_<ts>.<ext>`; `extractor/dashboard.py` globs on those prefixes, so keep them in sync.
- `extractor/schema.py` – `Source` / `Result` dataclasses; every adapter returns a `Result`. `Result.to_dict()` computes `stats`. Sources are deduped by `normalize_url` (strips tracking params, maps YouTube URLs to `youtube:<id>`).
- `extractor/adapters/` – one `Adapter` subclass per platform with `detect(raw)`, `parse(raw, meta)` and optional `inspect(raw)`. Registration order in `adapters/__init__.py` matters for auto-detection (most specific first).
- `extractor/sse.py` and `extractor/connect.py` – low-level decoders (SSE events; Connect-RPC envelopes with JSON or schema-less protobuf).
- Unrecognized stream content is recorded in `Result.diagnostics` (`unknown_ops`, `errors`), not raised.

## Conventions

- Standard library only; no third-party dependencies.
- Tests build synthetic streams by hand (see helpers in `tests/test_chatgpt.py`). Never commit real captures as fixtures.
- Platform APIs are undocumented and change often. When a format changes, use `extractor inspect` on a fresh capture and add a dated comment noting the observed format.

## Do not commit

`data/raw/`, `private/`, and `docs/` (session logs) are git-ignored because they contain account identifiers, tokens or private conversations. The repo is public.
