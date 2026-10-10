# AI Stream Analytics

Capture the raw answer streams from ChatGPT, Claude, Perplexity and Gemini in the browser, then parse them into one normalized format showing what each assistant searched for, which sources it retrieved, which it actually cited, and what it answered.

## How it works

1. **Capture** – `capture/ai_capture.user.js` is a userscript (Tampermonkey/Violentmonkey) that patches `fetch`, `XMLHttpRequest` and `WebSocket` on the supported sites. For every answer it downloads:
   - `<platform>_<ts>.sse` – the raw response body exactly as the page received it
   - `<platform>_<ts>.meta.json` – request URL/body and response status/headers (no auth headers)
   - `<platform>_<ts>.conversation.json` – the stored conversation the app fetches on reload
   - `<platform>_<ts>.ws.jsonl` – WebSocket frames (ChatGPT only)

   Chrome asks once per site to allow multiple automatic downloads. Allow it, otherwise only the first file arrives.

2. **Extract** – the `extractor` Python package turns a raw capture into a normalized JSON `Result`:
   - prompt, model, conversation id
   - fan-out search queries
   - sources, flagged as retrieved, cited, supporting or linked, with snippets and cited spans
   - reasoning/thinking text and the final answer
   - stats (e.g. retrieved-but-not-cited, cited-but-not-retrieved, YouTube sources) and parser diagnostics

3. **Report** – render a result as HTML + Markdown, or run the local dashboard that watches your downloads folder and parses captures as they arrive.

## Requirements

Python 3.10+ (standard library only, no dependencies).

## Usage

```sh
# auto-parse new captures from ~/Downloads and serve a dashboard at http://localhost:8765
python3 -m extractor serve [--watch ~/Downloads] [--port 8765]

# summarize what's in an unfamiliar raw capture
python3 -m extractor inspect data/raw/claude_X.sse

# raw capture -> normalized JSON (platform auto-detected; meta.json picked up if next to the file)
python3 -m extractor parse data/raw/chatgpt_X.sse -o data/out/chatgpt_X.json [--platform chatgpt] [--meta X.meta.json]

# normalized JSON -> .html + .md next to it
python3 -m extractor render data/out/chatgpt_X.json
```

## Supported platforms

| Platform   | Stream format                                         |
|------------|-------------------------------------------------------|
| ChatGPT    | SSE (v1 delta encoding), WebSocket resume path        |
| Claude     | Connect-RPC `StreamTimeline` envelopes; legacy SSE completion |
| Perplexity | SSE from `/rest/sse/`                                 |
| Gemini     | `)]}'`-framed JSON arrays over XHR (`StreamGenerate`) |

These are private web-app APIs and change without notice. When a capture stops parsing, run `inspect` on it and check `window.__captureUrls` in the browser console to find the new answer endpoint.

## Project layout

```
capture/              userscript (legacy/ holds the original ChatGPT-only version)
extractor/
  __main__.py         CLI: parse, render, inspect, serve
  schema.py           Source / Result dataclasses, URL normalization
  sse.py, connect.py  SSE and Connect-RPC (JSON + schema-less protobuf) decoders
  adapters/           one adapter per platform
  render.py           HTML / Markdown reports
  dashboard.py        downloads watcher + local web server
tests/                unit tests with hand-made streams
data/out/             sample normalized outputs
```

## Tests

```sh
python3 -m unittest discover tests
```

## Privacy

Raw captures (`data/raw/`) and anything in `private/` can contain account identifiers and tokens. Both are git-ignored, so keep them that way.
