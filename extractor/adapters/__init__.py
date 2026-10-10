from .base import Adapter
from .chatgpt import ChatGPTAdapter
from .claude import ClaudeAdapter
from .gemini import GeminiAdapter
from .perplexity import PerplexityAdapter

# order matters for auto-detection: more specific markers first
ADAPTERS: dict[str, type[Adapter]] = {
    ClaudeAdapter.name: ClaudeAdapter,
    GeminiAdapter.name: GeminiAdapter,
    PerplexityAdapter.name: PerplexityAdapter,
    ChatGPTAdapter.name: ChatGPTAdapter,
}


def get_adapter(platform: str, raw: bytes, fallback: bool = False) -> Adapter:
    if platform != "auto":
        return ADAPTERS[platform]()
    for cls in ADAPTERS.values():
        if cls.detect(raw):
            return cls()
    if fallback:
        return Adapter()
    raise ValueError("could not detect platform; pass --platform explicitly")
