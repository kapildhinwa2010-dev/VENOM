"""Locate the sibling packages without ever raising.

Two strategies are tried, in order:

  1. **installed** - `import inner_vote.vote` (the normal case, after
     `pip install "venomlayer[integrations]"`);
  2. **source**    - load the sibling's own file straight out of a local
     checkout (`VENOM_SIBLINGS_DIR`, default `/workspace/siblings`).

Strategy 2 exists because four of the sibling repos currently build an
**empty wheel**: their flat-layout package discovery is hijacked by a stray
`ci_logs/` directory, so `pip install` succeeds but ships no module
(`never-lie`, `drift-point`, `powerlaw-probe`, `variance-lies`).  See
`integrations/README.md`.  The source load still runs the sibling's own
code - only the (broken) packaging is bypassed.

Nothing in this module raises.  A sibling that is missing is simply
`None`, and every adapter turns that into VENOM's `{"ok": False, ...}`
convention.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

#: Where the sibling checkouts live (overridable for other machines).
SIBLINGS_DIR = os.environ.get("VENOM_SIBLINGS_DIR", "/workspace/siblings")


@dataclass(frozen=True)
class Sibling:
    """One sibling repository and the way to import it.

    kind:
        "pkg"  - src-layout; the package lives in ``<repo>/src/<package>/``
        "flat" - a single module file at the repository root
    """

    repo: str
    module: str
    import_path: str
    kind: str
    file: str = ""
    note: str = ""


#: The registry of sibling repositories.  Keys are the importable module
#: names; ``repo`` is the directory name used for the source fallback.
SIBLINGS: Dict[str, Sibling] = {
    "never_lie": Sibling(
        "NEVER_LIE", "never_lie", "never_lie", "flat", "never_lie.py",
        "digest-guarded slot memory with put/get/corrupt/fuzz",
    ),
    "factbase": Sibling(
        "FactBase", "factbase", "factbase", "flat", "factbase.py",
        "durable (subject, predicate, object) fact store",
    ),
    "know_vault": Sibling(
        "KNOW-VAULT", "know_vault", "know_vault.vault", "pkg",
        note="knowledge-slot capacity estimation",
    ),
    "fluidlayer": Sibling(
        "fluidtech", "fluidlayer", "fluidlayer.memory", "pkg",
        note="frozen register (exact slot memory) + fluid/combined layers",
    ),
    "tmu": Sibling(
        "TMU", "tmu", "tmu.clock", "pkg",
        note="internal clock + time management unit",
    ),
    "inner_vote": Sibling(
        "INNER-VOTE-", "inner_vote", "inner_vote.vote", "pkg",
        note="layer-wise agreement across paraphrases",
    ),
    "logit_stream": Sibling(
        "LOGIT-STREAM", "logit_stream", "logit_stream.stream", "pkg",
        note="per-layer logit increments and the birth layer of an answer",
    ),
    "state_mirror": Sibling(
        "STATE-MIRROR", "state_mirror", "state_mirror.mirror", "pkg",
        note="layer-by-layer comparison of twin prompts",
    ),
    "variance_lies": Sibling(
        "VARIEANCE-LIES", "variance_lies", "variance_lies", "flat",
        "variance_lies.py",
        note="variance captured vs behaviour preserved",
    ),
    "trace_path": Sibling(
        "TRACE-PATH", "trace_path", "trace_path.trace", "pkg",
        note="activation patching: which block carries a fact",
    ),
    "bus_trace": Sibling(
        "BUS-TRACE", "bus_trace", "bus_trace.trace", "pkg",
        note="per-block writes into the residual stream",
    ),
    "geometry_scan": Sibling(
        "GEOMETRY-SCAN", "geometry_scan", "geometry_scan.scan", "pkg",
        note="per-layer hidden-state geometry",
    ),
    "layer_census": Sibling(
        "LAYER-CENSUS-", "layer_census", "layer_census.census", "pkg",
        note="per-layer weight-matrix atlas",
    ),
    "dead_cell": Sibling(
        "DEAD-CELL", "dead_cell", "dead_cell.census", "pkg",
        note="dead weight rows and never-active neurons",
    ),
    "twin_tower": Sibling(
        "TWIN-TOWER", "twin_tower", "twin_tower.scan", "pkg",
        note="duplicated / redundant layer detection",
    ),
    "skip_run": Sibling(
        "SKIP-RUN-", "skip_run", "skip_run.depth", "pkg",
        note="how much of the depth is real (block skipping)",
    ),
    "powerlaw_probe": Sibling(
        "POWERLAW-PROBE", "powerlaw_probe", "powerlaw_probe", "flat",
        "powerlaw_probe.py",
        note="power-law exponent of activations vs weights",
    ),
    "drift_point": Sibling(
        "Draft_ponit_harness", "drift_point", "drift_point", "flat",
        "drift_point.py",
        note="first token where a compressed replay diverges",
    ),
    "noise_storm": Sibling(
        "NOISE-STORM", "noise_storm", "noise_storm.storm", "pkg",
        note="per-block fragility under calibrated noise",
    ),
    "complexityrouter": Sibling(
        "ComplexityRouter", "complexityrouter", "complexityrouter", "flat",
        "complexityrouter.py",
        note="scale-invariant complexity routing to experts",
    ),
}

#: Sibling repositories that the project brief mentions but that are not
#: present in the local checkout.  They are reported as missing rather
#: than silently omitted.
ABSENT_SIBLINGS: Dict[str, str] = {
    "knowledgebench": "KnowledgeBench",
    "memorybench": "MemoryBench",
    "robustnessbench": "RobustnessBench",
}

_LOCK = threading.Lock()
_CACHE: Dict[str, Optional[Any]] = {}
_MODE: Dict[str, str] = {}


def _load_installed(sib: Sibling):
    try:
        return importlib.import_module(sib.import_path)
    except Exception:  # noqa: BLE001 - a broken sibling must not break VENOM
        return None


def _load_source(sib: Sibling):
    root = os.path.join(SIBLINGS_DIR, sib.repo)
    if not os.path.isdir(root):
        return None

    if sib.kind == "flat":
        path = os.path.join(root, sib.file)
        if not os.path.isfile(path):
            return None
        try:
            spec = importlib.util.spec_from_file_location(sib.module, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[sib.module] = module
            spec.loader.exec_module(module)
            return module
        except Exception:  # noqa: BLE001
            sys.modules.pop(sib.module, None)
            return None

    src = os.path.join(root, "src")
    if not os.path.isdir(src):
        return None
    added = src not in sys.path
    if added:
        sys.path.insert(0, src)
    try:
        return importlib.import_module(sib.import_path)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if added:
            try:
                sys.path.remove(src)
            except ValueError:
                pass


def load(name: str):
    """Return the sibling module, or None when it cannot be loaded.

    Never raises.  The result is cached, so repeated calls are cheap.
    """
    if name not in SIBLINGS:
        return None
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        sib = SIBLINGS[name]
        module = _load_installed(sib)
        mode = "installed"
        if module is None:
            module = _load_source(sib)
            mode = "source"
        _CACHE[name] = module
        _MODE[name] = mode if module is not None else "missing"
        return module


def mode(name: str) -> str:
    """How a sibling was reached: "installed", "source" or "missing"."""
    load(name)
    return _MODE.get(name, "missing")


def installed(name: str) -> bool:
    """True only when the sibling is importable as an installed package.

    This is the strict check - it does not fall back to a local checkout.
    """
    sib = SIBLINGS.get(name)
    if sib is None:
        return False
    try:
        importlib.import_module(sib.import_path)
        return True
    except Exception:  # noqa: BLE001
        return False


def available_map() -> Dict[str, bool]:
    """{module_name: bool} - is each sibling usable at all?"""
    return {name: load(name) is not None for name in sorted(SIBLINGS)}


def source_map() -> Dict[str, str]:
    """{module_name: "installed" | "source" | "missing"}."""
    return {name: mode(name) for name in sorted(SIBLINGS)}
