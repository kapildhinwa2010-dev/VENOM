"""One-file drop-in. Copy this next to your AI and it just works.

    import venom
    model = venom.wrap(your_model, your_tokenizer)
    print(model.chat("what is the capital of france?"))

No install, no imports to wire up, no config. If you want the full package:

    from venom.shell import attach

This file is deliberately self-contained so that the "plug into any AI" claim
can be tested by pasting one file into any project.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

__version__ = "0.2.0"

_PUNCT = re.compile(r"[^\w\s]")


# --------------------------------------------------------------------- memory
def _norm(key: str) -> str:
    return " ".join(_PUNCT.sub(" ", key.strip().lower()).split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class Recall:
    status: str
    value: Optional[str] = None
    reason: str = ""
    key: Optional[str] = None

    def as_dict(self):
        return asdict(self)


class MemoryStore:
    """Exact recall, digest-guarded, bounded. Abstains instead of guessing."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self.evictions = 0
        self.corruptions = 0

    def remember(self, key: str, value: str) -> dict:
        k = _norm(key)
        if not k:
            return {"ok": False, "reason": "empty_key"}
        if k in self._data:
            self._data.move_to_end(k)
        self._data[k] = {"value": value, "digest": _digest(value)}
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
            self.evictions += 1
        return {"ok": True, "key": k}

    def has(self, key: str) -> bool:
        return _norm(key) in self._data

    def recall(self, query: str) -> Recall:
        k = _norm(query)
        if not k:
            return Recall("abstain", reason="empty_key")
        rec = self._data.get(k)
        if rec is None:
            return Recall("abstain", reason="not_found", key=k)
        if _digest(rec["value"]) != rec["digest"]:
            self.corruptions += 1
            return Recall("abstain", reason="digest_mismatch", key=k)
        self._data.move_to_end(k)
        return Recall("hit", value=rec["value"], reason="exact", key=k)

    def stats(self) -> dict:
        return {"slots": len(self._data), "capacity": self.capacity,
                "evictions": self.evictions, "corruptions": self.corruptions}


# --------------------------------------------------------------------- reflex
class Reflex:
    """Rule-based. Fires on a memory hit; never guesses."""

    def __init__(self, store: MemoryStore, max_ngram: int = 6,
                 min_key_words: int = 2, min_key_chars: int = 6):
        self.store = store
        self.max_ngram = max_ngram
        self.min_key_words = min_key_words
        self.min_key_chars = min_key_chars

    def matching_keys(self, text: str):
        words = _norm(text).split()
        seen = set()
        for n in range(min(self.max_ngram, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                key = " ".join(words[i:i + n])
                if not key or key in seen:
                    continue
                seen.add(key)
                if self.store.has(key) and self._specific(key):
                    yield key

    def _specific(self, key: str) -> bool:
        words = key.split()
        if len(words) >= self.min_key_words:
            return True
        return len(key) >= self.min_key_chars

    def decide(self, text: str):
        for key in self.matching_keys(text):
            return True, "memory_hit", key
        return False, "no_trigger", None


# ------------------------------------------------------------------- the bus
class _Bus:
    """Read-only forward hooks. Returns None so the output is untouched."""

    def __init__(self, model, blocks):
        self.model = model
        self.blocks = list(blocks)
        self.taps: Dict[str, Any] = {}
        self._handles = []
        self._attached = False

    def _hook(self, name):
        def fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.taps[name] = tensor.detach().clone()
            return None
        return fn

    def attach(self):
        if self._attached:
            raise RuntimeError("already attached")
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(self._hook(f"block_{i}")))
        self._attached = True
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._attached = False

    def release(self):
        self.detach()
        self.taps.clear()

    @property
    def attached(self):
        return self._attached


def _discover_blocks(model):
    """Find the transformer blocks of ANY architecture."""
    try:
        import torch.nn as nn
    except ImportError:
        return None, None

    def looks_like_block(obj):
        names = {n.lower() for n, _ in obj.named_children()}
        attn = any(any(h in n for h in ("attn", "attention")) for n in names)
        mlp = any(any(h in n for h in ("mlp", "ffn", "feed_forward")) for n in names)
        return attn and mlp

    best = None
    for path, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        blocks = [c for c in module if looks_like_block(c)]
        if len(blocks) < max(1, len(module) // 2):
            continue
        score = len(blocks) / len(module) + path.count(".") * 0.01
        if best is None or score > best[0]:
            best = (score, path, module)
    if best is None:
        return None, None
    return best[1], best[2]


def _resolve(model, path):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ------------------------------------------------------------------- the host
class VenomHost:
    """Wraps any model+tokenizer. The original model is untouched."""

    def __init__(self, model, tokenizer=None, capacity: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.store = MemoryStore(capacity=capacity)
        self.reflex = Reflex(self.store)
        self.bus = None
        self.blocks_path = None
        self.injections = 0
        self._attached = False

    # ------------------------------------------------------------- lifecycle
    def attach(self):
        if self._attached:
            raise RuntimeError("already attached")
        path, blocks = _discover_blocks(self.model)
        if blocks is None:
            raise RuntimeError("could not discover the transformer blocks of this model")
        self.blocks_path = path
        self.bus = _Bus(self.model, blocks)
        self.bus.attach()
        self._attached = True
        return self

    def detach(self):
        if self.bus is not None and self.bus.attached:
            self.bus.detach()
        self._attached = False

    def release(self):
        self.detach()
        if self.bus is not None:
            self.bus.release()

    @property
    def attached(self):
        return self._attached

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.release()

    # ---------------------------------------------------------------- memory
    def remember(self, key, value):
        return self.store.remember(key, value)

    def recall(self, query):
        return self.store.recall(query).as_dict()

    # ------------------------------------------------------------- inference
    def prompt_for(self, text: str) -> str:
        fire, reason, key = self.reflex.decide(text)
        if not fire:
            return text
        r = self.store.recall(key)
        if r.status != "hit":
            return text
        self.injections += 1
        return f"CONTEXT: {r.value} | {text}"

    def generate(self, prompt: str, **kwargs) -> str:
        if self.tokenizer is None:
            raise RuntimeError("generate() needs a tokenizer")
        ids = self.tokenizer(self.prompt_for(prompt), return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in ids.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    def chat(self, text: str, **kwargs) -> str:
        return self.generate(text, **kwargs)

    def stats(self) -> dict:
        return {"attached": self._attached, "injections": self.injections,
                "memory": self.store.stats(),
                "blocks_path": self.blocks_path}


def wrap(model, tokenizer=None, capacity: int = 1024) -> VenomHost:
    """The one-call entry point."""
    return VenomHost(model, tokenizer, capacity)


def tool_defs() -> List[Dict[str, Any]]:
    """MCP tool schemas."""
    return [
        {"name": "venom.recall",
         "description": "Exact lookup of a stored fact. Abstains instead of guessing.",
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"]}},
        {"name": "venom.remember",
         "description": "Store a key/value fact in the session memory.",
         "inputSchema": {"type": "object",
                         "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                         "required": ["key", "value"]}},
        {"name": "venom.time_now",
         "description": "Current UTC time (host models have no clock).",
         "inputSchema": {"type": "object", "properties": {}}},
    ]


if __name__ == "__main__":
    print(json.dumps({"version": __version__, "tools": tool_defs()}, indent=2))
