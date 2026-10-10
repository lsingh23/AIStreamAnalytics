"""Platform-agnostic output schema shared by every adapter."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = re.compile(r"^(utm_\w+|fbclid|gclid|ref|ref_src|si)$")


def domain_of(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def youtube_id(url: str) -> str | None:
    parts = urlsplit(url)
    host = domain_of(url)
    if host == "youtu.be":
        return parts.path.strip("/") or None
    if host.endswith("youtube.com"):
        if parts.path == "/watch":
            return dict(parse_qsl(parts.query)).get("v")
        m = re.match(r"^/(shorts|embed|live)/([\w-]+)", parts.path)
        if m:
            return m.group(2)
    return None


def normalize_url(url: str) -> str:
    """Canonical key used to dedupe sources and match citations against retrieved results."""
    vid = youtube_id(url)
    if vid:
        return f"youtube:{vid}"
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)])
    path = parts.path.rstrip("/")
    return urlunsplit(("", domain_of(url), path, query, "")).lstrip("/")


@dataclass
class Source:
    url: str
    title: str = ""
    snippet: str = ""
    pub_date: str | None = None
    attribution: str = ""
    type: str = "web"  # web | youtube | ...
    retrieved: bool = False
    cited: bool = False
    supporting: bool = False  # listed as a runner-up under a citation (ChatGPT supporting_websites), not cited itself
    linked: bool = False  # linked in the answer text by the model itself, not through a search citation
    ref_ids: list[str] = field(default_factory=list)
    cited_spans: list[dict[str, Any]] = field(default_factory=list)
    snippets: list[dict[str, Any]] = field(default_factory=list)  # every snippet seen: location, kind, length, text
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return normalize_url(self.url)


@dataclass
class Result:
    platform: str
    prompt: str = ""
    model: str = ""
    captured_at: str = ""
    conversation_id: str = ""
    fanout_queries: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    answer: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for s, src in zip(d["sources"], self.sources):
            s["domain"] = domain_of(src.url)
        d["stats"] = {
            "fanout_queries": len(self.fanout_queries),
            "sources_total": len(self.sources),
            "retrieved": sum(s.retrieved for s in self.sources),
            "cited": sum(s.cited for s in self.sources),
            "retrieved_not_cited": sum(s.retrieved and not s.cited for s in self.sources),
            "cited_not_in_retrieved": sum(s.cited and not s.retrieved for s in self.sources),
            "supporting_only": sum(s.supporting and not s.cited for s in self.sources),
            "linked_only": sum(s.linked and not s.cited and not s.retrieved for s in self.sources),
            "long_snippets": sum(any(v["kind"] == "long" for v in s.snippets) for s in self.sources),
            "youtube": sum(s.type == "youtube" for s in self.sources),
        }
        return d
