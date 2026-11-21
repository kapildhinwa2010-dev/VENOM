"""Benchmark integrations: KnowledgeBench, MemoryBench, RobustnessBench.

**None of these three repositories exist in the local checkout** under the
names the project brief gives, and no directory in `/workspace/siblings`
matches them.  Rather than guess an API or silently omit them, each
adapter here reports the same thing: the repository is not present, so the
integration cannot run.

When one of these repos is cloned, put it in ``VENOM_SIBLINGS_DIR`` (or
install it as a package) and fill in the corresponding adapter - the
loader will find it as soon as the import path is registered in
``_loader.SIBLINGS``.

The bench family would be the natural place for the measurements that
LIMITATIONS.md lists as *not yet run* (fix rate, harm rate, tool-call
recall).  Until a bench repo is actually present, VENOM's own numbers
stay unmeasured.
"""
from __future__ import annotations

from typing import Any, Dict

from venom.integrations._base import fail

ABSENT_MESSAGE = (
    "repository not present in the sibling checkout - nothing to call, "
    "and no result is reported"
)


def _absent(source: str) -> Dict[str, Any]:
    return fail(source, ABSENT_MESSAGE)


def knowledge_bench_available() -> bool:
    """Is the KnowledgeBench sibling importable? (not cloned - always False)."""
    return False


def knowledge_bench_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the KnowledgeBench sibling.

    What it does: nothing yet - the repository is not in the sibling
    checkout.

    What it does NOT prove: any benchmark result.  A missing repository is
    reported as a missing repository, never as a passing or failing score.
    """
    return _absent("KnowledgeBench")


def memory_bench_available() -> bool:
    """Is the MemoryBench sibling importable? (not cloned - always False)."""
    return False


def memory_bench_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the MemoryBench sibling.

    What it does: nothing yet - the repository is not in the sibling
    checkout.

    What it does NOT prove: any memory-quality result.  VENOM's memory
    numbers in LIMITATIONS.md remain unmeasured until a bench actually
    runs.
    """
    return _absent("MemoryBench")


def robustness_bench_available() -> bool:
    """Is the RobustnessBench sibling importable? (not cloned - always False)."""
    return False


def robustness_bench_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the RobustnessBench sibling.

    What it does: nothing yet - the repository is not in the sibling
    checkout.

    What it does NOT prove: any robustness result.
    """
    return _absent("RobustnessBench")
