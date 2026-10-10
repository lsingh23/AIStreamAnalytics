"""CLI.

  python3 -m extractor inspect data/raw/claude_X.sse             # what's in the stream (run this first)
  python3 -m extractor parse   data/raw/chatgpt_X.sse -o data/out/pizza.json [--meta X.meta.json]
  python3 -m extractor render  data/out/pizza.json                # -> pizza.html + pizza.md next to it
  python3 -m extractor serve   [--watch ~/Downloads] [--port 8765]  # auto-parse new captures + local dashboard
  python3 -m extractor har     ~/Downloads/chatgpt.com.har          # DevTools HAR -> capture files in data/raw
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .adapters import ADAPTERS, get_adapter
from .render import to_html, to_markdown
from .sse import decode_data, iter_events

INTERESTING = re.compile(r"search|cite|citation|ref|quer|url|snippet|youtube|video|channel|view|like|follow|verif|thought|reason", re.I)


def _find_meta(path: Path) -> dict:
    stem = path.name.split(".")[0]
    candidate = path.with_name(f"{stem}.meta.json")
    return json.loads(candidate.read_text()) if candidate.exists() else {}


def cmd_parse(args: argparse.Namespace) -> None:
    path = Path(args.file)
    raw = path.read_bytes()
    meta = json.loads(Path(args.meta).read_text()) if args.meta else _find_meta(path)
    adapter = get_adapter(args.platform, raw)
    result = adapter.parse(raw, meta).to_dict()
    out = Path(args.output) if args.output else Path("data/out") / f"{path.name.split('.')[0]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[{adapter.name}] wrote {out}  stats={json.dumps(result['stats'])}")
    if result["diagnostics"].get("unknown_ops") or result["diagnostics"].get("errors"):
        print("  ! diagnostics:", json.dumps({k: result["diagnostics"].get(k) for k in ("unknown_ops", "errors")})[:500])


def cmd_render(args: argparse.Namespace) -> None:
    path = Path(args.file)
    result = json.loads(path.read_text())
    path.with_suffix(".html").write_text(to_html(result))
    path.with_suffix(".md").write_text(to_markdown(result))
    print(f"wrote {path.with_suffix('.html')} and {path.with_suffix('.md')}")


def cmd_inspect(args: argparse.Namespace) -> None:
    raw = Path(args.file).read_bytes()
    events = list(iter_events(raw))
    adapter = get_adapter(args.platform, raw, fallback=True)
    is_gzip = raw[:2] == b"\x1f\x8b"
    print(f"{len(raw)} bytes, {len(events)} SSE events, gzip={is_gzip}, adapter={adapter.name}")
    summary, objects = adapter.inspect(raw)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("first events:" if events else "first bytes:")
    for ev in events[: args.head]:
        print(f"  [{ev.event}] {ev.data[:200]}")
    if not events:
        print("  " + raw[:600].decode("utf-8", "replace"))

    paths: Counter[str] = Counter()

    def collect(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}"
                if INTERESTING.search(k):
                    paths[p] += 1
                collect(v, p)
        elif isinstance(obj, list):
            for item in obj:
                collect(item, prefix + "[]")

    collect(objects)
    print("\ninteresting key paths (count):")
    for p, n in sorted(paths.items()):
        print(f"  {n:4d}  {p}")


def cmd_har(args: argparse.Namespace) -> None:
    from .har import extract
    written = extract(Path(args.file).expanduser(), Path(args.out))
    for f in written:
        print(f"  wrote {f}")
    print(f"{len(written)} capture(s) extracted (headers and cookies are not copied)")


def cmd_serve(args: argparse.Namespace) -> None:
    from .dashboard import serve
    serve(Path(args.watch), Path("data/raw"), Path("data/out"), args.port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="extractor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="raw capture -> normalized JSON")
    p.add_argument("file")
    p.add_argument("--platform", default="auto", choices=["auto", *ADAPTERS])
    p.add_argument("--meta", help="meta.json saved by the capture script (auto-detected if next to the file)")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_parse)
    r = sub.add_parser("render", help="normalized JSON -> HTML + Markdown report")
    r.add_argument("file")
    r.set_defaults(func=cmd_render)
    i = sub.add_parser("inspect", help="summarize a raw capture (any platform)")
    i.add_argument("file")
    i.add_argument("--platform", default="auto", choices=["auto", *ADAPTERS])
    i.add_argument("--head", type=int, default=15)
    i.set_defaults(func=cmd_inspect)
    h = sub.add_parser("har", help="extract answer streams / stored conversations from a DevTools HAR export")
    h.add_argument("file")
    h.add_argument("--out", default="data/raw")
    h.set_defaults(func=cmd_har)
    s = sub.add_parser("serve", help="watch downloads folder, auto-parse captures, serve local dashboard")
    s.add_argument("--watch", default="~/Downloads")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
