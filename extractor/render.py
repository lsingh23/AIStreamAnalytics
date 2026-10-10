"""Render a normalized result (dict from Result.to_dict()) as a static HTML report and a Markdown summary."""
from __future__ import annotations

import html
import json
from typing import Any


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _extra_summary(extra: dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in extra.items() if not isinstance(v, (dict, list)) and len(str(v)) < 80)


def _status(src: dict[str, Any]) -> list[str]:
    flags = [("cited", src["cited"]), ("retrieved", src["retrieved"]),
             ("supporting", src.get("supporting")), ("linked", src.get("linked"))]
    return [name for name, on in flags if on]


def _snippet_kinds(src: dict[str, Any]) -> str:
    return ", ".join(f"{v['kind']} {v['length']}" if v["length"] else v["kind"] for v in src.get("snippets", []))


def to_markdown(r: dict[str, Any]) -> str:
    s = r["stats"]
    lines = [
        f"# {r['platform']} — {r['prompt']}",
        "",
        f"Model: `{r['model']}` · Conversation: `{r['conversation_id']}` · Captured: {r['captured_at']}",
        "",
        f"**{s['fanout_queries']} fan-out queries · {s['retrieved']} retrieved · {s['cited']} cited · "
        f"{s['retrieved_not_cited']} retrieved but not cited · {s.get('supporting_only', 0)} supporting only · "
        f"{s.get('linked_only', 0)} linked only · {s.get('long_snippets', 0)} with a long snippet · {s['youtube']} YouTube**",
        "",
        "## Fan-out queries",
        "",
        *[f"{i}. {q}" for i, q in enumerate(r["fanout_queries"], 1)],
        "",
        "## Sources",
        "",
        "| # | Status | Type | Domain | Title | Date | Snippets | Snippet |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, src in enumerate(r["sources"], 1):
        snippet = (src["snippet"] or "").replace("\n", " ").replace("|", "\\|")
        title = (src["title"] or "").replace("|", "\\|")
        lines.append(
            f"| {i} | {', '.join(_status(src))} | {src['type']} | "
            f"{src['domain']} | [{title or src['url']}]({src['url']}) | {src['pub_date'] or ''} | {_snippet_kinds(src)} | "
            f"{snippet[:160]}{'…' if len(snippet) > 160 else ''} |"
        )
    snaps = [x for x in r["sources"] if x["snippet"] or x["type"] == "youtube"]
    if snaps:
        lines += ["", "## Snapshots (full snippet text as streamed)", ""]
        for src in snaps:
            lines += [f"### {src['title'] or src['url']}", "", f"- URL: {src['url']}",
                      f"- Status: {', '.join(_status(src)) or '-'} · Type: {src['type']}"]
            lines += [f"- {k}: {v}" for k, v in src["extra"].items() if not isinstance(v, (dict, list))]
            variants = [v for v in src.get("snippets", []) if v["text"]] or \
                [{"location": "-", "kind": "-", "length": len(src["snippet"]), "text": src["snippet"]}]
            for v in variants:
                lines += ["", f"{v['kind']} snippet, {v['length']} chars, from `{v['location']}`:", "```", v["text"], "```"]
            lines.append("")
    return "\n".join(lines) + "\n"


BADGE_CLASS = {"cited": "cited", "retrieved": "ret", "supporting": "sup", "linked": "lnk"}
PLATFORM_NAMES = {"chatgpt": "ChatGPT", "claude": "Claude", "perplexity": "Perplexity", "gemini": "Gemini"}


def to_html(r: dict[str, Any]) -> str:
    s = r["stats"]
    rows = []
    for i, src in enumerate(r["sources"], 1):
        badges = "".join(f"<span class='b {BADGE_CLASS[f]}'>{f.capitalize()}</span>" for f in _status(src))
        details = ""
        variants = [v for v in src.get("snippets", []) if v["text"]] or \
            ([{"location": "-", "kind": "-", "length": len(src["snippet"]), "text": src["snippet"]}] if src["snippet"] else [])
        if variants or src["extra"]:
            snaps = "".join(f"<div class='sk'>{_e(v['kind'])} · {v['length']} chars · {_e(v['location'])}</div><pre>{_e(v['text'])}</pre>"
                            for v in variants)
            details = (f"<details><summary>Snapshot{' · ' + _e(_snippet_kinds(src)) if src.get('snippets') else ''}</summary>{snaps}"
                       f"<pre class='x'>{_e(json.dumps(src['extra'], indent=2, ensure_ascii=False))}</pre></details>")
        date = f"<span>{_e(src['pub_date'])}</span>" if src["pub_date"] else ""
        fav = src["extra"].get("favicon_url") or f"https://www.google.com/s2/favicons?sz=64&domain={_e(src['domain'])}"
        rows.append(
            f"<tr><td class='i'>{i}</td><td class='st'>{badges}</td>"
            f"<td><div class='src'><img src='{_e(fav)}' alt='' loading='lazy' onerror=\"this.style.visibility='hidden'\">"
            f"<div class='body'><a class='t' href='{_e(src['url'])}' target='_blank' rel='noopener'>{_e(src['title'] or src['url'])}</a>"
            f"<div class='meta'><span>{_e(src['domain'])}</span><span class='tag'>{_e(src['type'])}</span>"
            f"{date}</div>"
            f"<div class='meta x'>{_e(_extra_summary(src['extra']))}</div>{details}</div></div></td>"
            f"<td class='fi'>{_e(', '.join(src['extra'].get('found_in', [])))}</td></tr>"
        )
    queries = "".join(f"<li><span class='n'>{i}</span>{_e(q)}</li>" for i, q in enumerate(r["fanout_queries"], 1)) \
        or "<li class='none'>No fan-out queries found</li>"
    reasoning = "".join(f"<pre>{_e(t)}</pre>" for t in r["reasoning"]) or "<p class='none'>None streamed</p>"
    rate = round(s["cited"] / s["retrieved"] * 100) if s["retrieved"] else 0
    kpis = [("Fan-out queries", s["fanout_queries"]), ("Retrieved", s["retrieved"]), ("Cited", s["cited"]),
            ("Retrieved, not cited", s["retrieved_not_cited"]), ("Supporting only", s.get("supporting_only", 0)),
            ("Linked only", s.get("linked_only", 0)), ("Long snippets", s.get("long_snippets", 0)), ("YouTube", s["youtube"])]
    kpi_html = "".join(f"<div class='kpi'><div class='l'>{l}</div><div class='v'>{v}</div></div>" for l, v in kpis)
    platform = r["platform"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(PLATFORM_NAMES.get(platform, platform))} capture</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#fafafa;--surface:#fff;--fg:#0f172a;--fg2:#475569;--muted:#94a3b8;--line:#eef0f3;--line2:#e2e8f0;
--acc:#4f46e5;--acc-soft:#eef2ff;--ok:#059669;--ok-soft:#ecfdf5;--r:14px;--shadow:0 1px 2px rgba(15,23,42,.04),0 1px 3px rgba(15,23,42,.04)}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font:14px/1.55 Inter,system-ui,-apple-system,sans-serif;margin:0;-webkit-font-smoothing:antialiased}}
a{{color:inherit;text-decoration:none}}
.top{{background:rgba(255,255,255,.85);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}}
.top .in,.page{{margin:0 auto;max-width:1240px;padding:0 24px}}
.top .in{{align-items:center;display:flex;gap:16px;height:60px;justify-content:space-between}}
.brand{{align-items:center;display:flex;font-weight:600;gap:10px;letter-spacing:-.01em}}
.logo{{background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:8px;display:grid;height:28px;place-items:center;width:28px}}
.logo i{{background:#fff;border-radius:2px;display:inline-block;margin:0 1px;width:3px}}
.nav{{display:flex;gap:8px}}
.btn{{background:var(--surface);border:1px solid var(--line2);border-radius:10px;color:var(--fg2);font-size:13px;font-weight:500;padding:6px 12px;white-space:nowrap}}
.btn:hover{{border-color:#cbd5e1;color:var(--fg)}}
.page{{padding-bottom:64px;padding-top:32px}}
.chip{{align-items:center;border-radius:6px;display:inline-flex;font-size:11px;font-weight:600;gap:5px;padding:2px 8px}}
.chip:before{{background:currentColor;border-radius:50%;content:"";height:6px;width:6px}}
.p-chatgpt{{background:#e7f6f2;color:#10a37f}}.p-claude{{background:#fbefe9;color:#c4623f}}
.p-perplexity{{background:#e6f2f3;color:#20808d}}.p-gemini{{background:#e9f1fe;color:#4285f4}}
h1{{font-size:26px;font-weight:600;letter-spacing:-.02em;line-height:1.3;margin:12px 0 8px;max-width:900px}}
h2{{font-size:15px;font-weight:600;margin:0}}
.sub{{color:var(--fg2);display:flex;flex-wrap:wrap;font-size:13px;gap:6px 16px}}
.sub code{{background:#f1f5f9;border-radius:6px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;padding:2px 7px}}
.kpis{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));margin:24px 0}}
.kpi,.card{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow)}}
.kpi{{padding:14px 16px}}.kpi .l{{color:var(--fg2);font-size:12px;font-weight:500}}
.kpi .v{{font-size:24px;font-variant-numeric:tabular-nums;font-weight:600;letter-spacing:-.02em;margin-top:4px}}
.card{{margin-bottom:16px;overflow:hidden}}
.hd{{align-items:center;border-bottom:1px solid var(--line);display:flex;gap:12px;justify-content:space-between;padding:14px 20px}}
.hd .c{{color:var(--muted);font-size:12px}}
.bar{{align-items:center;display:flex;gap:10px}}.bar .track{{background:var(--line);border-radius:99px;height:6px;overflow:hidden;width:120px}}
.bar .track b{{background:var(--ok);display:block;height:100%}}
ol.qs{{list-style:none;margin:0;padding:8px 20px 12px}}
ol.qs li{{align-items:baseline;border-bottom:1px dashed var(--line);display:flex;gap:12px;padding:10px 0}}ol.qs li:last-child{{border:0}}
ol.qs .n{{background:var(--acc-soft);border-radius:6px;color:var(--acc);flex:none;font-size:11px;font-weight:600;min-width:22px;padding:1px 0;text-align:center}}
.none{{color:var(--muted);font-style:normal}}
.wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}
th{{background:#fcfcfd;border-bottom:1px solid var(--line);color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.04em;padding:10px 16px;text-align:left;text-transform:uppercase;white-space:nowrap}}
td{{border-bottom:1px solid var(--line);padding:14px 16px;vertical-align:top}}tr:last-child td{{border-bottom:0}}
td.i{{color:var(--muted);font-variant-numeric:tabular-nums;width:1%}}td.st{{white-space:nowrap;width:1%}}td.fi{{color:var(--fg2);font-size:12px}}
.b{{border-radius:6px;display:inline-block;font-size:11px;font-weight:600;margin:0 4px 4px 0;padding:2px 8px}}
.cited{{background:var(--ok-soft);color:var(--ok)}}.ret{{background:var(--acc-soft);color:var(--acc)}}
.sup{{background:#fef3c7;color:#b45309}}.lnk{{background:#f1f5f9;color:var(--fg2)}}.sk{{color:var(--fg2);font-size:11px;margin-top:8px}}
.src{{display:flex;gap:12px}}.src img{{border-radius:4px;flex:none;height:18px;margin-top:2px;width:18px}}.src .body{{min-width:0}}
.src .t{{color:var(--fg);font-weight:500;word-break:break-word}}.src .t:hover{{color:var(--acc)}}
.meta{{color:var(--muted);display:flex;flex-wrap:wrap;font-size:12px;gap:4px 10px;margin-top:2px}}.meta.x:empty{{display:none}}
.tag{{background:#f1f5f9;border-radius:4px;color:var(--fg2);padding:0 6px}}
details{{margin-top:8px}}summary{{color:var(--acc);cursor:pointer;font-size:12px;font-weight:500;user-select:none}}
pre{{background:#f8fafc;border:1px solid var(--line);border-radius:10px;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0 0;max-height:420px;overflow:auto;padding:12px 14px;white-space:pre-wrap;word-break:break-word}}
.x{{font-size:11px}}.pad{{padding:4px 20px 20px}}
.answer{{background:none;border:0;font:14px/1.7 Inter,system-ui,sans-serif;max-height:none;padding:16px 20px;margin:0}}
@media (max-width:600px){{.top .in,.page{{padding:0 16px}}.page{{padding-top:24px}}.kpis{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:21px}}
.brand span.name{{display:none}}td.fi,th.fi{{display:none}}td,th{{padding-left:12px;padding-right:12px}}}}
</style></head><body>
<div class="top"><div class="in">
  <div class="brand"><span class="logo"><span><i style="height:8px"></i><i style="height:14px"></i><i style="height:10px"></i></span></span><span class="name">AI Stream Analytics</span></div>
  <div class="nav"><!--nav--></div>
</div></div>
<main class="page">
<span class="chip p-{_e(platform)}">{_e(PLATFORM_NAMES.get(platform, platform))}</span>
<h1>{_e(r['prompt'])}</h1>
<div class="sub"><span>Model <code>{_e(r['model'])}</code></span><span>Conversation <code>{_e(r['conversation_id'])}</code></span>
<span>Captured {_e(r['captured_at'])}</span></div>
<section class="kpis">{kpi_html}</section>
<div class="card"><div class="hd"><h2>Fan-out queries</h2><span class="c">{len(r['fanout_queries'])}</span></div><ol class="qs">{queries}</ol></div>
<div class="card"><div class="hd"><h2>Sources</h2><div class="bar"><span class="c">{rate}% cited</span><span class="track"><b style="width:{rate}%"></b></span></div></div>
<div class="wrap"><table><tr><th>#</th><th>Status</th><th>Source</th><th class="fi">Found in</th></tr>
{''.join(rows) or "<tr><td colspan='4' class='none'>No sources</td></tr>"}</table></div></div>
<div class="card"><div class="hd"><h2>Reasoning</h2></div><div class="pad">{reasoning}</div></div>
<div class="card"><div class="hd"><h2>Final answer</h2></div><pre class="answer">{_e(r['answer'])}</pre></div>
</main>
</body></html>
"""
