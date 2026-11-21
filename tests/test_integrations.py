"""Tests for the sibling-repo integration layer.

The integrations are OPTIONAL. VENOM must import and run with none of the
sibling packages installed, so the tests here assert two things:

  1. the layer never raises, whatever is installed;
  2. every integration function returns a dict with an "ok" key.

The sibling repos are the ones the master plan named as the sources for the
consolidation (TABLE 2). This test file is what keeps that consolidation
honest: an adapter that silently stops working must fail here, not in
production.
"""
import importlib

import pytest

from venom.integrations import available, integration_status

MODULES = ["memory", "sensors", "census", "trace", "verify", "time", "bench", "router"]

EXPECTED_SIBLINGS = [
    "never_lie", "factbase", "know_vault", "fluidlayer", "inner_vote",
    "logit_stream", "state_mirror", "variance_lies", "layer_census",
    "dead_cell", "twin_tower", "skip_run", "powerlaw_probe",
    "bus_trace", "trace_path", "geometry_scan", "drift_point",
    "noise_storm", "tmu", "complexityrouter",
]

# The sibling repo name -> the integration key it is known by.
ALIASES = {"fluidtech": "fluidlayer"}


def test_available_never_raises():
    result = available()
    assert isinstance(result, dict)


def test_integration_status_shape():
    status = integration_status()
    assert isinstance(status, dict)
    assert isinstance(status.get("available", []), list)
    assert isinstance(status.get("missing", []), list)


def test_every_named_sibling_has_an_adapter():
    """Each repo named in the plan's TABLE 2 must be wired to something."""
    status = integration_status()
    known = set(status.get("available", [])) | set(status.get("missing", []))
    missing_adapters = [s for s in EXPECTED_SIBLINGS
                        if s not in known and ALIASES.get(s) not in known]
    assert not missing_adapters, f"no adapter for: {missing_adapters}"


@pytest.mark.parametrize("module_name", MODULES)
def test_every_integration_function_returns_a_dict_with_ok(module_name):
    module = importlib.import_module(f"venom.integrations.{module_name}")
    runners = [n for n in dir(module)
               if not n.startswith("_")
               and callable(getattr(module, n))
               and (n.endswith("_run") or n.endswith("_available"))]
    assert runners, f"{module_name} exposes no integration functions"

    for name in runners:
        fn = getattr(module, name)
        if name.endswith("_available"):
            assert isinstance(fn(), bool), f"{module_name}.{name} must return a bool"
            continue
        out = fn()
        assert isinstance(out, dict), f"{module_name}.{name} must return a dict"
        assert "ok" in out, f"{module_name}.{name} must report 'ok'"


def test_a_missing_sibling_degrades_instead_of_raising(monkeypatch):
    """If a sibling package cannot be loaded the adapter must say so, not crash."""
    from venom.integrations import memory
    from venom.integrations import _loader

    # the loader has two paths: an installed package and a source checkout.
    # both must fail for the adapter to report "not available".
    monkeypatch.setattr(_loader, "_load_installed", lambda sib: None)
    monkeypatch.setattr(_loader, "_load_source", lambda sib: None)
    _loader._CACHE.clear()
    out = memory.never_lie_run()
    assert isinstance(out, dict)
    assert out.get("ok") is False
    assert out.get("error")
