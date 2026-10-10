"""Hand-made v1 delta stream mimicking ChatGPT's format.  Run: python3 -m unittest discover tests"""
import gzip
import json
import unittest

from extractor.adapters.chatgpt import ChatGPTAdapter

YT = "https://www.youtube.com/watch?v=abc123XYZ"
WEB1 = "https://www.seriouseats.com/basic-neapolitan-pizza-dough-recipe"
WEB2 = "https://example.com/pizza?utm_source=chatgpt.com"
WEB3 = "https://en.wikipedia.org/wiki/Neapolitan_pizza"
LONG = "Neapolitan dough is " + "flour, water, salt and yeast, fermented slowly. " * 20


def sse(*events):
    out = []
    for ev in events:
        name, data = ev if isinstance(ev, tuple) else (None, ev)
        out.append((f"event: {name}\n" if name else "") + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n")
    return "".join(out).encode()


def msg(mid, role, content, recipient="all", metadata=None, channel=None):
    return {"id": mid, "author": {"role": role}, "content": content, "recipient": recipient,
            "metadata": metadata or {}, "channel": channel, "status": "in_progress"}


STREAM = sse(
    ("delta_encoding", '"v1"'),
    {"type": "input_message", "input_message": msg("u1", "user", {"content_type": "text", "parts": ["youtube: pizza dough"]})},
    ("delta", {"p": "", "o": "add", "c": 0, "v": {"conversation_id": "conv-1", "message": msg(
        "a1", "assistant", {"content_type": "code", "text": ""}, recipient="web.run")}}),
    ("delta", {"p": "/message/content/text", "o": "append", "v": '{"search_query": [{"q": "best pizza dough recipe"},'}),
    ("delta", {"v": ' {"q": "neapolitan pizza dough youtube"}]}'}),
    ("delta", {"v": {"conversation_id": "conv-1", "message": msg("t1", "tool", {"content_type": "text", "parts": [""]}, metadata={
        "search_result_groups": [
            {"type": "search_result_group", "domain": "youtube.com", "entries": [
                {"type": "search_result", "url": YT, "title": "Perfect Pizza Dough", "snippet": "00:00 Intro\n01:30 Mixing",
                 "pub_date": 1700000000, "attribution": "YouTube", "ref_id": {"turn_index": 0, "ref_type": "video", "ref_index": 0},
                 "channel_name": "Vito Iacopelli", "view_count": 1200000, "verified": True}]},
            {"type": "search_result_group", "domain": "seriouseats.com", "entries": [
                {"type": "search_result", "url": WEB1, "title": "Neapolitan Dough", "snippet": "Use 00 flour",
                 "ref_id": {"turn_index": 0, "ref_type": "search", "ref_index": 1}}]},
            {"type": "search_result_group", "domain": "example.com", "entries": [
                {"type": "search_result", "url": WEB2, "title": "Not cited", "snippet": "meh"}]},
        ]})}, "c": 1}),
    ("delta", {"p": "", "o": "add", "c": 2, "v": {"conversation_id": "conv-1", "message": msg(
        "a2", "assistant", {"content_type": "text", "parts": [""]}, metadata={"model_slug": "gpt-5"})}}),
    ("delta", {"p": "/message/content/parts/0", "o": "append", "v": "Use 00 flour "}),
    ("delta", {"v": "citeturn0search1 and watch this video citeturn0video0."}),
    ("delta", {"p": "", "o": "patch", "v": [
        {"p": "/message/metadata", "o": "append", "v": {"content_references": [
            {"matched_text": "citeturn0search1", "start_idx": 13, "end_idx": 30, "type": "grouped_webpages",
             "alt": "([Serious Eats](https://www.seriouseats.com/))",
             "items": [{"title": "Neapolitan Dough", "url": WEB1 + "?utm_source=chatgpt.com", "snippet": LONG,
                        "supporting_websites": [{"title": "Neapolitan pizza", "url": WEB3, "snippet": None}]}]},
            {"matched_text": "citeturn0video0", "start_idx": 50, "end_idx": 70, "type": "video", "alt": ""},
            {"matched_text": " ", "start_idx": 80, "end_idx": 80, "type": "sources_footnote",
             "sources": [{"title": "Not cited", "url": WEB2}]},
        ]}},
        {"p": "/message/status", "o": "replace", "v": "finished_successfully"},
    ]}),
    "[DONE]",
)


class ChatGPTAdapterTest(unittest.TestCase):
    def check(self, raw):
        r = ChatGPTAdapter().parse(raw).to_dict()
        self.assertEqual(r["prompt"], "youtube: pizza dough")
        self.assertEqual(r["model"], "gpt-5")
        self.assertEqual(r["conversation_id"], "conv-1")
        self.assertEqual(r["fanout_queries"], ["best pizza dough recipe", "neapolitan pizza dough youtube"])
        by_url = {s["url"]: s for s in r["sources"]}
        self.assertEqual(len(r["sources"]), 4)
        self.assertTrue(by_url[YT]["cited"] and by_url[YT]["retrieved"])
        self.assertEqual(by_url[YT]["type"], "youtube")
        self.assertEqual(by_url[YT]["extra"]["view_count"], 1200000)
        self.assertTrue(by_url[WEB1]["cited"])
        self.assertFalse(by_url[WEB2]["cited"])
        self.assertTrue(by_url[WEB2]["extra"]["in_sources_footnote"])
        self.assertEqual(r["stats"]["retrieved_not_cited"], 1)
        # short + long snippet variants are both kept; the long one becomes the main snippet
        self.assertEqual(by_url[WEB1]["snippet"], LONG)
        self.assertEqual([(v["location"], v["kind"]) for v in by_url[WEB1]["snippets"]],
                         [("search_results", "short"), ("citation", "long")])
        self.assertEqual(by_url[YT]["snippets"][0]["kind"], "short")
        # supporting_websites are runner-ups: listed, not cited, not retrieved
        sup = by_url[WEB3]
        self.assertTrue(sup["supporting"])
        self.assertFalse(sup["cited"] or sup["retrieved"])
        self.assertEqual(sup["snippets"][0]["kind"], "empty")
        self.assertEqual(sup["extra"]["supports"], ["\ue200cite\ue202turn0search1\ue201"])
        self.assertEqual(r["stats"]["supporting_only"], 1)
        self.assertEqual(r["stats"]["long_snippets"], 1)
        self.assertEqual(r["answer"], "Use 00 flour ([Serious Eats](https://www.seriouseats.com/)) and watch this video .")
        self.assertEqual(r["diagnostics"].get("unknown_ops", {}), {})
        self.assertEqual(r["diagnostics"].get("errors", []), [])

    def test_sse_stream(self):
        self.check(STREAM)

    def test_gzipped_stream(self):
        self.check(gzip.compress(STREAM))

    def test_conversation_json(self):
        conv = {"conversation_id": "conv-1", "mapping": {}}
        from extractor.adapters.chatgpt import DeltaAssembler
        from extractor.sse import decode_data, iter_events
        asm = DeltaAssembler()
        for ev in iter_events(STREAM):
            asm.feed(ev.event, decode_data(ev.data))
        msgs = [msg("u1", "user", {"content_type": "text", "parts": ["youtube: pizza dough"]})]
        msgs += [e["message"] for e in asm.envelopes()]
        for i, m in enumerate(msgs):
            m["create_time"] = i
            conv["mapping"][m["id"]] = {"message": m}
        self.check(json.dumps(conv).encode())


class GenUIAnswerTest(unittest.TestCase):
    """Generative-UI answers: <Cite refs=...> is a citation, <Link url=...> is a link the model wrote itself."""

    def test_link_is_not_a_citation(self):
        raw = sse(
            ("delta_encoding", '"v1"'),
            ("delta", {"p": "", "o": "add", "c": 0, "v": {"conversation_id": "conv-2", "message": msg(
                "t1", "tool", {"content_type": "text", "parts": [""]}, metadata={"search_result_groups": [
                    {"type": "search_result_group", "domain": "seriouseats.com", "entries": [
                        {"type": "search_result", "url": WEB1, "title": "Neapolitan Dough", "snippet": "Use 00 flour",
                         "ref_id": {"turn_index": 0, "ref_type": "search", "ref_index": 0}}]}]})}}),
            ("delta", {"p": "", "o": "add", "c": 1, "v": {"conversation_id": "conv-2", "message": msg(
                "a1", "assistant", {"content_type": "text", "parts": [
                    'Use 00 flour <Cite refs={["turn0search0"]}/>. Try <Link url="https://linear.app" title="Linear"/> '
                    f'and <Link url="{WEB1}" title="Serious Eats"/>.']})}}),
            "[DONE]",
        )
        r = ChatGPTAdapter().parse(raw).to_dict()
        by_url = {s["url"]: s for s in r["sources"]}
        self.assertTrue(by_url[WEB1]["cited"] and by_url[WEB1]["linked"])
        linear = by_url["https://linear.app"]
        self.assertTrue(linear["linked"])
        self.assertFalse(linear["cited"] or linear["retrieved"])
        self.assertEqual(r["stats"]["cited"], 1)
        self.assertEqual(r["stats"]["linked_only"], 1)
        self.assertEqual(r["stats"]["cited_not_in_retrieved"], 0)


if __name__ == "__main__":
    unittest.main()
