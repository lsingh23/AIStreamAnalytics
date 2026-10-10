"""Local dashboard: watches a downloads folder for new captures, parses them, serves the reports.

  python3 -m extractor serve                      # watches ~/Downloads, http://localhost:8765
  python3 -m extractor serve --watch ~/Downloads --port 8765
"""
from __future__ import annotations

import html
import json
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .adapters import get_adapter
from .har import extract as extract_har
from .render import to_html

PLATFORMS = ("chatgpt", "claude", "perplexity", "gemini")  # file prefixes written by capture/ai_capture.user.js
CAPTURE_GLOBS = tuple(f"{p}_*.{ext}" for p in PLATFORMS for ext in ("sse", "conversation.json", "meta.json"))
SETTLE_SECONDS = 2.5  # the capture script spaces downloads ~1s apart; wait for the meta file too
STORED_PLATFORMS = ("chatgpt", "claude")  # adapters that parse the stored-conversation JSON as well as the stream


def _stem(p: Path) -> str:
    return p.name.split(".")[0]


class Pipeline:
    def __init__(self, watch: Path, raw_dir: Path, out_dir: Path) -> None:
        self.watch, self.raw_dir, self.out_dir = watch, raw_dir, out_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.errors: dict[str, str] = {}
        self.lock = threading.Lock()
        self.hars: dict[str, tuple[int, float]] = {}  # HAR exports already unpacked: name -> (size, mtime)
        # outputs older than the parser code are re-parsed, so a restart after an adapter fix refreshes everything
        self.code_mtime = max(p.stat().st_mtime for p in Path(__file__).parent.rglob("*.py"))

    def sync(self) -> list[str]:
        """Copy settled new captures into raw_dir and (re)parse anything whose inputs changed."""
        now = time.time()
        if self.watch.is_dir():
            for pattern in CAPTURE_GLOBS:
                for src in self.watch.glob(pattern):
                    dst = self.raw_dir / src.name
                    if now - src.stat().st_mtime < SETTLE_SECONDS:
                        continue
                    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                        shutil.copy2(src, dst)
            # DevTools HAR exports: only the answer/conversation bodies are written out, never the headers
            for har in self.watch.glob("*.har"):
                st = har.stat()
                if now - st.st_mtime < SETTLE_SECONDS or self.hars.get(har.name) == (st.st_size, st.st_mtime):
                    continue
                try:
                    extract_har(har, self.raw_dir)
                    self.errors.pop(har.name, None)
                except (ValueError, OSError):  # not JSON / still being written
                    self.errors[har.name] = traceback.format_exc(limit=1)
                self.hars[har.name] = (st.st_size, st.st_mtime)

        parsed = []
        stored = [f for p in STORED_PLATFORMS for f in self.raw_dir.glob(f"{p}_*.conversation.json")]
        for raw in sorted([*self.raw_dir.glob("*.sse"), *stored]):
            out = self.out_dir / f"{_stem(raw)}.json"
            meta = self.raw_dir / f"{_stem(raw)}.meta.json"
            newest_input = max(raw.stat().st_mtime, meta.stat().st_mtime if meta.exists() else 0, self.code_mtime)
            if out.exists() and out.stat().st_mtime >= newest_input:
                continue
            data = raw.read_bytes()
            try:
                adapter = get_adapter("auto", data)
            except ValueError:  # not an answer stream (e.g. a side request the capture caught): skip it
                continue
            try:
                meta_d = json.loads(meta.read_text()) if meta.exists() else {}
                result = adapter.parse(data, meta_d).to_dict()
                if not result["sources"] and not result["answer"]:  # e.g. a cancelled or idle stream
                    continue
                result["source_file"] = raw.name
                result["captured_at"] = result["captured_at"] or _stem(raw).split("_", 1)[-1]
                out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                self.errors.pop(raw.name, None)
                parsed.append(raw.name)
            except Exception:  # keep serving; show the error on the dashboard
                self.errors[raw.name] = traceback.format_exc(limit=3)
        return parsed

    def captures(self) -> list[dict]:
        items = []
        for out in self.out_dir.glob("*.json"):
            if out.stem.endswith(".conversation"):
                continue
            try:
                r = json.loads(out.read_text())
            except json.JSONDecodeError:
                continue
            if "stats" not in r:
                continue
            items.append({
                "id": out.stem, "platform": r["platform"], "prompt": r["prompt"], "model": r["model"],
                "file": r.get("source_file", out.stem), "stats": r["stats"],
                "fanout_queries": r["fanout_queries"], "mtime": out.stat().st_mtime,
                "stored": (r.get("diagnostics") or {}).get("input") == "conversation_json",
                "hidden": len(r.get("insights") or {}),
                "ts": out.stem.split("_", 1)[-1],  # capture timestamp from the filename; ISO-like, sorts as text
            })
        return sorted(items, key=lambda x: x["ts"], reverse=True)


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Stream Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#fafafa;--surface:#fff;--fg:#0f172a;--fg2:#475569;--muted:#94a3b8;--line:#eef0f3;--line2:#e2e8f0;
--acc:#4f46e5;--acc-soft:#eef2ff;--ok:#059669;--ok-soft:#ecfdf5;--warn:#d97706;--bad:#dc2626;--bad-soft:#fef2f2;
--chatgpt:#10a37f;--claude:#d97757;--perplexity:#20808d;--gemini:#4285f4;
--r:14px;--shadow:0 1px 2px rgba(15,23,42,.04),0 1px 3px rgba(15,23,42,.04)}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.5 Inter,system-ui,-apple-system,sans-serif;margin:0;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.top{background:rgba(255,255,255,.85);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.top .in,.page{margin:0 auto;max-width:1240px;padding:0 24px}
.top .in{align-items:center;display:flex;gap:16px;height:60px;justify-content:space-between}
.brand{align-items:center;display:flex;font-weight:600;gap:10px;letter-spacing:-.01em}
.logo{background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:8px;display:grid;height:28px;place-items:center;width:28px}
.logo i{background:#fff;border-radius:2px;display:inline-block;margin:0 1px;width:3px}
.live{align-items:center;background:var(--surface);border:1px solid var(--line2);border-radius:999px;color:var(--fg2);display:flex;font-size:12px;gap:8px;max-width:60vw;padding:5px 12px 5px 10px}
.live code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pulse{background:var(--ok);border-radius:50%;flex:none;height:7px;position:relative;width:7px}
.pulse:after{animation:p 2s infinite;border:2px solid var(--ok);border-radius:50%;content:"";inset:-4px;position:absolute}
@keyframes p{0%{opacity:.6;transform:scale(.6)}100%{opacity:0;transform:scale(1.6)}}
.page{padding-bottom:64px;padding-top:32px}
h1{font-size:24px;font-weight:600;letter-spacing:-.02em;margin:0}
.sub{color:var(--fg2);margin:4px 0 0}
.kpis{display:grid;gap:12px;grid-template-columns:repeat(5,1fr);margin:24px 0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:16px 18px}
.kpi .l{color:var(--fg2);font-size:12px;font-weight:500}
.kpi .v{font-size:26px;font-variant-numeric:tabular-nums;font-weight:600;letter-spacing:-.02em;margin-top:6px}
.kpi .v small{color:var(--muted);font-size:14px;font-weight:500;margin-left:2px}
.bar{background:var(--line);border-radius:99px;height:4px;margin-top:10px;overflow:hidden}.bar b{background:var(--acc);display:block;height:100%}
.toolbar{align-items:center;display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;margin-bottom:12px}
.seg{background:#f1f5f9;border-radius:10px;display:flex;flex-wrap:wrap;gap:2px;padding:3px}
.seg button{align-items:center;background:none;border:0;border-radius:8px;color:var(--fg2);cursor:pointer;display:flex;font:500 13px Inter,system-ui,sans-serif;gap:6px;padding:6px 12px}
.seg button:hover{color:var(--fg)}.seg button.on{background:var(--surface);box-shadow:0 1px 2px rgba(15,23,42,.08);color:var(--fg)}
.seg .c{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.tools{display:flex;flex-wrap:wrap;gap:8px}
.input{background:var(--surface);border:1px solid var(--line2);border-radius:10px;color:var(--fg);font:13px Inter,system-ui,sans-serif;height:36px;outline:none;padding:0 12px}
.input:focus{border-color:#a5b4fc;box-shadow:0 0 0 3px var(--acc-soft)}
input.input{min-width:220px;padding-left:34px;background:var(--surface) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='none' stroke='%2394a3b8' stroke-width='2' viewBox='0 0 24 24'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.5-3.5'/%3E%3C/svg%3E") 11px center no-repeat}
select.input{cursor:pointer}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}
th{background:#fcfcfd;border-bottom:1px solid var(--line);color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.04em;padding:10px 16px;text-align:left;text-transform:uppercase;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:14px 16px;vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr.row{cursor:pointer;transition:background .12s}tbody tr.row:hover{background:#f8fafc}
th.n,td.n{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.prompt{color:var(--fg);font-weight:500;max-width:520px}
.meta{align-items:center;color:var(--muted);display:flex;flex-wrap:wrap;font-size:12px;gap:8px;margin-bottom:6px}
.chip{align-items:center;border-radius:6px;display:inline-flex;font-size:11px;font-weight:600;gap:5px;padding:2px 8px}
.chip:before{background:currentColor;border-radius:50%;content:"";height:6px;width:6px}
.p-chatgpt{background:#e7f6f2;color:var(--chatgpt)}.p-claude{background:#fbefe9;color:#c4623f}
.p-perplexity{background:#e6f2f3;color:var(--perplexity)}.p-gemini{background:#e9f1fe;color:var(--gemini)}
.qs{margin-top:8px}.q{color:var(--fg2);font-size:12px;overflow:hidden;padding-left:14px;position:relative;text-overflow:ellipsis;white-space:nowrap;max-width:520px}
.q:before{color:var(--muted);content:"↳";left:0;position:absolute}.more{color:var(--acc);font-size:12px;font-weight:500;padding-left:14px}
.tag{background:#f1f5f9;border-radius:6px;color:var(--fg2);font-size:11px;font-weight:500;padding:1px 7px}
.model{background:#f1f5f9;border-radius:6px;color:var(--fg2);font:12px ui-monospace,SFMono-Regular,Menlo,monospace;padding:2px 7px;white-space:nowrap}
.cite{align-items:center;display:flex;gap:8px;justify-content:flex-end}
.ring{--v:0;background:conic-gradient(var(--ok) calc(var(--v)*1%),var(--line) 0);border-radius:50%;height:16px;width:16px;-webkit-mask:radial-gradient(circle 5px,transparent 98%,#000);mask:radial-gradient(circle 5px,transparent 98%,#000)}
.dim{color:var(--muted)}
.empty{padding:64px 24px;text-align:center}.empty .ic{background:var(--acc-soft);border-radius:14px;display:grid;font-size:22px;height:48px;margin:0 auto 14px;place-items:center;width:48px}
.empty h3{font-size:15px;font-weight:600;margin:0 0 4px}.empty p{color:var(--fg2);margin:0}
.err{background:var(--bad-soft);border:1px solid #fecaca;border-radius:12px;color:#7f1d1d;font-size:12px;margin-bottom:12px;padding:12px 14px}
.err b{display:block;margin-bottom:4px}.err pre{font:11px/1.5 ui-monospace,Menlo,monospace;margin:0;white-space:pre-wrap}
@media (max-width:960px){.kpis{grid-template-columns:repeat(2,1fr)}.hide-md{display:none}}
@media (max-width:600px){.top .in,.page{padding:0 16px}.page{padding-top:24px}.hide-sm{display:none}input.input{min-width:0;flex:1}.tools{width:100%}td,th{padding-left:12px;padding-right:12px}}
</style></head><body>
<div class="top"><div class="in">
  <div class="brand"><span class="logo"><span><i style="height:8px"></i><i style="height:14px"></i><i style="height:10px"></i></span></span>AI Stream Analytics</div>
  <div class="live"><span class="pulse"></span><span class="hide-sm">Watching</span><code>__WATCH__</code></div>
</div></div>
<main class="page">
  <h1>Captures</h1>
  <p class="sub">Fan-out queries, retrieved sources, citations, products and hidden metadata from ChatGPT, Claude, Perplexity and Gemini.</p>
  <section class="kpis" id="kpis"></section>
  <div id="errors"></div>
  <div class="toolbar">
    <div class="seg" id="seg"></div>
    <div class="tools">
      <input class="input" id="q" type="search" placeholder="Search prompts, queries, models" autocomplete="off">
      <select class="input" id="sort"><option value="newest">Newest first</option><option value="oldest">Oldest first</option>
      <option value="platform">Platform</option><option value="cited">Most cited</option></select>
    </div>
  </div>
  <div class="card"><div class="wrap"><table>
    <thead><tr><th>Prompt</th><th class="hide-md">Model</th><th class="n">Queries</th><th class="n hide-sm">Retrieved</th>
    <th class="n">Cited</th><th class="n hide-md">Not cited</th><th class="n hide-md">YouTube</th><th class="n hide-md">Products</th>
    <th class="n hide-sm" title="Fields the platform sent but did not show">Hidden</th></tr></thead>
    <tbody id="rows"></tbody></table></div></div>
</main>
<script>
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const PLATFORMS = {chatgpt: 'ChatGPT', claude: 'Claude', perplexity: 'Perplexity', gemini: 'Gemini'};
const $ = (id) => document.getElementById(id);
let last = '', data = null, platform = 'all';
const store = {get: (k, d) => { try { return localStorage.getItem(k) || d; } catch (e) { return d; } },
               set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} }};
$('sort').value = store.get('sort', 'newest'); platform = store.get('platform', 'all');
const sorters = {
  newest: (a, b) => b.ts.localeCompare(a.ts),
  oldest: (a, b) => a.ts.localeCompare(b.ts),
  platform: (a, b) => a.platform.localeCompare(b.platform) || b.ts.localeCompare(a.ts),
  cited: (a, b) => b.stats.cited - a.stats.cited || b.ts.localeCompare(a.ts),
};
$('sort').addEventListener('change', () => { store.set('sort', $('sort').value); render(); });
$('q').addEventListener('input', render);
$('seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; platform = b.dataset.p; store.set('platform', platform); render(); });
$('rows').addEventListener('click', (e) => { const tr = e.target.closest('tr[data-href]'); if (tr && !e.target.closest('a')) location.href = tr.dataset.href; });
function when(ts) {
  const d = new Date(ts.replace(/T(\d\d)-(\d\d)-(\d\d)-(\d+)Z$/, 'T$1:$2:$3.$4Z'));
  if (isNaN(d)) return ts;
  const s = (Date.now() - d) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return d.toLocaleDateString(undefined, {month: 'short', day: 'numeric'}) + ', ' + d.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'});
}
const chip = (p) => `<span class="chip p-${esc(p)}">${esc(PLATFORMS[p] || p)}</span>`;
async function refresh() {
  try {
    const res = await fetch('/api/captures'); const d = await res.json();
    const sig = JSON.stringify(d); if (sig === last) return; last = sig; data = d; render();
  } catch (e) { console.warn(e); }
}
function render() {
  if (!data) return;
  const all = data.captures;
  $('errors').innerHTML = Object.entries(data.errors).map(([f, e]) => `<div class="err"><b>Couldn't parse ${esc(f)}</b><pre>${esc(e)}</pre></div>`).join('');
  const counts = {all: all.length}; all.forEach((c) => { counts[c.platform] = (counts[c.platform] || 0) + 1; });
  if (platform !== 'all' && !counts[platform]) platform = 'all';
  $('seg').innerHTML = ['all', ...Object.keys(PLATFORMS)].map((p) => `<button data-p="${p}" class="${p === platform ? 'on' : ''}">${p === 'all' ? 'All' : PLATFORMS[p]}<span class="c">${counts[p] || 0}</span></button>`).join('');
  const needle = $('q').value.trim().toLowerCase();
  const shown = all.filter((c) => (platform === 'all' || c.platform === platform) &&
    (!needle || [c.prompt, c.model, ...c.fanout_queries].join(' ').toLowerCase().includes(needle)))
    .sort(sorters[$('sort').value] || sorters.newest);
  const sum = (k) => shown.reduce((n, c) => n + (c.stats[k] || 0), 0);
  const ret = sum('retrieved'), cited = sum('cited'), rate = ret ? Math.round(cited / ret * 100) : 0;
  $('kpis').innerHTML = [
    ['Captures', shown.length], ['Fan-out queries', sum('fanout_queries')], ['Sources retrieved', ret], ['Sources cited', cited],
  ].map(([l, v]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v.toLocaleString()}</div></div>`).join('') +
    `<div class="kpi"><div class="l">Citation rate</div><div class="v">${rate}<small>%</small></div><div class="bar"><b style="width:${rate}%"></b></div></div>`;
  if (!shown.length) {
    $('rows').innerHTML = `<tr><td colspan="9"><div class="empty"><div class="ic">${all.length ? '🔍' : '📡'}</div>
      <h3>${all.length ? 'No matching captures' : 'Waiting for your first capture'}</h3>
      <p>${all.length ? 'Try a different search or platform filter.' : 'Send a prompt in ChatGPT, Claude, Perplexity or Gemini with the capture script installed, or save a DevTools HAR into the watched folder.'}</p></div></td></tr>`;
    return;
  }
  $('rows').innerHTML = shown.map((c) => {
    const pct = c.stats.retrieved ? Math.round(c.stats.cited / c.stats.retrieved * 100) : 0;
    const qs = c.fanout_queries.slice(0, 2).map((q) => `<div class="q" title="${esc(q)}">${esc(q)}</div>`).join('') +
      (c.fanout_queries.length > 2 ? `<div class="more">+${c.fanout_queries.length - 2} more queries</div>` : '');
    const n = (v) => v ? v : `<span class="dim">0</span>`;
    return `<tr class="row" data-href="/capture/${encodeURIComponent(c.id)}">
      <td><div class="meta">${chip(c.platform)}<span title="${esc(c.ts)}">${esc(when(c.ts))}</span>${c.stored ? '<span class="tag" title="Parsed from the stored conversation the app loads on reload">Stored</span>' : ''}</div>
        <a class="prompt" href="/capture/${encodeURIComponent(c.id)}">${esc(c.prompt) || '<span class="dim">(no prompt)</span>'}</a>${qs ? `<div class="qs">${qs}</div>` : ''}</td>
      <td class="hide-md"><span class="model">${esc(c.model || '—')}</span></td>
      <td class="n">${n(c.stats.fanout_queries)}</td><td class="n hide-sm">${n(c.stats.retrieved)}</td>
      <td class="n"><div class="cite" title="${pct}% of retrieved sources cited"><span class="ring" style="--v:${pct}"></span>${n(c.stats.cited)}</div></td>
      <td class="n hide-md">${n(c.stats.retrieved_not_cited)}</td><td class="n hide-md">${n(c.stats.youtube)}</td>
      <td class="n hide-md">${n(c.stats.products)}</td><td class="n hide-sm">${n(c.hidden)}</td></tr>`;
  }).join('');
}
render(); refresh(); setInterval(refresh, 3000); setInterval(render, 60000);
</script></body></html>
"""


def make_handler(pipe: Pipeline):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, ctype: str = "text/html; charset=utf-8", status: int = 200) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(self.path.split("?")[0])
            if path == "/":
                return self._send(INDEX_HTML.replace("__WATCH__", html.escape(str(pipe.watch))))
            if path == "/api/captures":
                with pipe.lock:
                    return self._send(json.dumps({"captures": pipe.captures(), "errors": pipe.errors}), "application/json")
            if path.startswith("/capture/") or path.startswith("/json/"):
                cid = Path(path.split("/", 2)[2]).name
                f = pipe.out_dir / f"{cid}.json"
                if not f.exists():
                    return self._send("not found", "text/plain", 404)
                if path.startswith("/json/"):
                    return self._send(f.read_text(), "application/json")
                page = to_html(json.loads(f.read_text())).replace(
                    "<!--nav-->", f"<a class='btn' href='/'>← All captures</a><a class='btn' href='/json/{html.escape(cid)}'>Raw JSON</a>", 1)
                return self._send(page)
            self._send("not found", "text/plain", 404)

        def log_message(self, *args) -> None:
            pass

    return Handler


def serve(watch: Path, raw_dir: Path, out_dir: Path, port: int, interval: float = 2.0) -> None:
    pipe = Pipeline(watch.expanduser(), raw_dir, out_dir)

    def loop() -> None:
        while True:
            with pipe.lock:
                for name in pipe.sync():
                    print(f"  parsed {name}")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(pipe))
    print(f"dashboard: http://localhost:{port}   (watching {pipe.watch}, Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
