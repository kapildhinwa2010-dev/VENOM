"""Optional integrations with the sibling VENOM projects.

VENOM has **zero required dependencies**.  Everything in this package is
optional: if a sibling package is not installed, the corresponding
adapter returns ``{"ok": False, "error": ..., "source": ...}`` and VENOM's
own built-in implementations (in ``venom/sensors/*``, ``venom/memory/*``,
``venom/core/*``) remain the fallback.

The adapters do not re-implement the siblings' logic - each one imports
the sibling's own module and calls it.  Every adapter:

* returns VENOM's dict convention,
* never raises (exceptions become ``{"ok": False, "error": ...}``),
* and carries a docstring that states what the sibling code shows *and*
  what it does not prove.

Usage::

    from venom.integrations import available, integration_status
    available()             # {"bus_trace": True, "never_lie": True, ...}
    integration_status()    # {"available": [...], "missing": [...], "count": N}

    from venom.integrations import census
    census.layer_census_run()   # {"ok": True, "source": "LAYER-CENSUS-", ...}
"""
from __future__ import annotations

from typing import Any, Dict, List

from venom.integrations import _loader
from venom.integrations._base import fail, guard, ok, require, sibling_info
from venom.integrations._loader import (
    ABSENT_SIBLINGS,
    SIBLINGS,
    available_map,
    installed,
    load,
    mode,
    source_map,
)

__all__ = [
    "available",
    "available_installed",
    "integration_status",
    "sibling_info",
    "SIBLINGS",
    "ABSENT_SIBLINGS",
]


def available() -> Dict[str, bool]:
    """{module_name: bool} - is each sibling package usable?

    A sibling counts as available when it can be imported as an installed
    package **or** loaded directly from a local checkout (see
    ``_loader`` for why the second path exists).  This function never
    raises: an unavailable sibling is reported as ``False``.

    This says nothing about whether the sibling's *own* behaviour is
    correct - only whether VENOM can reach its code.
    """
    try:
        return available_map()
    except Exception:  # noqa: BLE001 - availability must never raise
        return {name: False for name in sorted(SIBLINGS)}


def available_installed() -> Dict[str, bool]:
    """{module_name: bool} - importable as an installed package only.

    Stricter than :func:`available`: it ignores the local-checkout
    fallback.  This is the check that shows a real packaging gap.
    """
    try:
        return {name: installed(name) for name in sorted(SIBLINGS)}
    except Exception:  # noqa: BLE001
        return {name: False for name in sorted(SIBLINGS)}


def integration_status() -> Dict[str, Any]:
    """A summary of which siblings are usable.

    Returns::

        {"available": [...],   # module names that can be reached
         "missing": [...],     # module names that cannot
         "count": int,         # the number of registered siblings
         "installed": [...],   # reachable as an installed package
         "source_only": [...], # reachable only via a local checkout
         "absent": [...]}      # repositories not present at all

    ``count`` is the size of the registry (available + missing), not the
    number that happen to be available - the two are also given
    separately as ``len(available)`` and ``len(missing)``.
    """
    try:
        reachable = available()
        modes = source_map()
    except Exception:  # noqa: BLE001
        reachable = {name: False for name in sorted(SIBLINGS)}
        modes = {name: "missing" for name in sorted(SIBLINGS)}

    available_names: List[str] = sorted(n for n, v in reachable.items() if v)
    missing_names: List[str] = sorted(n for n, v in reachable.items() if not v)
    return {
        "available": available_names,
        "missing": missing_names,
        "count": len(SIBLINGS),
        "installed": sorted(n for n, m in modes.items() if m == "installed"),
        "source_only": sorted(n for n, m in modes.items() if m == "source"),
        "absent": sorted(ABSENT_SIBLINGS.keys()),
    }
