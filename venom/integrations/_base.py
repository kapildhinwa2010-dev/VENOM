"""Shared helpers for the sibling adapters.

Every adapter returns VENOM's dict convention:

    success -> {"ok": True,  "source": "<sibling-repo>", ...sibling output...}
    failure -> {"ok": False, "error": "...",             "source": "<sibling-repo>"}

The rule that matters: **an integration never raises**.  A missing sibling
package, a missing optional dependency (numpy / torch) or a bad argument
must all come back as a dict the caller can inspect - not as an exception
that kills the host.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from venom.integrations._loader import SIBLINGS, load


def ok(source: str, **payload: Any) -> Dict[str, Any]:
    """Build a success payload."""
    out: Dict[str, Any] = {"ok": True, "source": source}
    out.update(payload)
    return out


def fail(source: str, error: Any) -> Dict[str, Any]:
    """Build a failure payload."""
    return {"ok": False, "error": str(error), "source": source}


def guard(source: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Run ``fn`` and convert any exception into a failure payload.

    This is the only place where a sibling's exception is swallowed, and it
    always reports the exception type as well as its message so the cause
    is not hidden.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - an integration must never raise
        return fail(source, "{}: {}".format(type(exc).__name__, exc))


def require(name: str, source: str):
    """Return the sibling module or a failure payload.

    Returns ``(module, None)`` on success and ``(None, failure_dict)`` when
    the sibling is not importable.
    """
    module = load(name)
    if module is None:
        return None, fail(
            source,
            "sibling package '{}' is not importable - install it or check "
            "VENOM_SIBLINGS_DIR".format(name),
        )
    return module, None


def sibling_info(name: str) -> Dict[str, Any]:
    """Metadata about a registered sibling (repo, import path, description)."""
    sib = SIBLINGS.get(name)
    if sib is None:
        return {"module": name, "repo": None, "import_path": None, "provides": None}
    return {
        "module": sib.module,
        "repo": sib.repo,
        "import_path": sib.import_path,
        "provides": sib.note,
    }
