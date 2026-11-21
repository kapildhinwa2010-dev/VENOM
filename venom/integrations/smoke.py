"""Call every integration once and print what came back.

    python -m venom.integrations.smoke

Every call is wrapped so that a missing sibling package shows up as a
failed payload instead of a traceback.  The exit code is 0 when every
integration that *could* run did run, and 1 otherwise.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

#: (label, callable) for every adapter in the package.
def _cases() -> List[Tuple[str, Callable[[], Dict[str, Any]]]]:
    from venom.integrations import bench, census, memory, router, sensors, time
    from venom.integrations import trace, verify

    return [
        ("never_lie", memory.never_lie_run),
        ("factbase", memory.factbase_run),
        ("know_vault", memory.know_vault_run),
        ("fluidtech", memory.fluidtech_run),
        ("inner_vote", sensors.inner_vote_run),
        ("logit_stream", sensors.logit_stream_run),
        ("state_mirror", sensors.state_mirror_run),
        ("variance_lies", sensors.variance_lies_run),
        ("trace_path", trace.trace_path_run),
        ("bus_trace", trace.bus_trace_run),
        ("geometry_scan", trace.geometry_scan_run),
        ("layer_census", census.layer_census_run),
        ("dead_cell", census.dead_cell_run),
        ("twin_tower", census.twin_tower_run),
        ("skip_run", census.skip_run_run),
        ("powerlaw_probe", census.powerlaw_probe_run),
        ("drift_point", verify.drift_point_run),
        ("noise_storm", verify.noise_storm_run),
        ("tmu", time.tmu_run),
        ("complexity_router", router.complexity_router_run),
        ("knowledge_bench", bench.knowledge_bench_run),
        ("memory_bench", bench.memory_bench_run),
        ("robustness_bench", bench.robustness_bench_run),
    ]


def main() -> int:
    from venom.integrations import available, integration_status

    status = integration_status()
    print("integrations status:")
    print(json.dumps(status, indent=2))
    print()
    print("available (usable):  ", json.dumps(available()))
    print()

    failures = []
    for label, fn in _cases():
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - the smoke test reports, never crashes
            out = {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}
        if not isinstance(out, dict) or "ok" not in out:
            failures.append((label, "no 'ok' key in the returned dict"))
            print("[FAIL] {:<18} did not return a dict with an 'ok' key".format(label))
            continue
        if out.get("ok"):
            print("[ OK ] {:<18} {}".format(label, out.get("source")))
        else:
            failures.append((label, str(out.get("error"))))
            print("[SKIP] {:<18} {}".format(label, out.get("error")))

    print()
    print("{}/{} integrations returned ok=True".format(
        len(_cases()) - len(failures), len(_cases())))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
