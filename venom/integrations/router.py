"""Routing integration: ComplexityRouter.

ComplexityRouter scores an input by its scale-invariant complexity (L2
norm combined with a normalised entropy proxy) and routes it to the
cheapest expert whose capacity covers that score.

What it does NOT prove: that the complexity score predicts how hard a
request is for a real model, or that the routing saves anything.  The
score is a statistic of the vector it is given - in this integration, a
synthetic one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from venom.integrations._base import guard, ok, require

SOURCE_COMPLEXITY_ROUTER = "ComplexityRouter"


def complexity_router_available() -> bool:
    """Is the ComplexityRouter sibling importable?"""
    from venom.integrations._loader import load
    return load("complexityrouter") is not None


def complexity_router_run(vector: Optional[Any] = None,
                          experts: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Score an input and route it to the cheapest sufficient expert.

    What it does: calls ComplexityRouter's ``complexity_score`` and
    ``route``, then ``handle`` so the chosen expert's call counter moves;
    the usage report is returned alongside the decision.

    What it does NOT prove: that the complexity score reflects the real
    difficulty of a request, that the cheapest expert is good enough, or
    that any compute was saved.  The routing decision is only as good as
    the capacities that were configured.
    """
    def _run():
        module, failure = require("complexityrouter", SOURCE_COMPLEXITY_ROUTER)
        if failure is not None:
            return failure
        import numpy as np

        data = np.asarray(vector, dtype=float) if vector is not None else np.array(
            [0.3, 0.9, 0.2, 0.8, 0.5, 0.1, 0.7, 0.4]
        )
        router = module.ComplexityRouter()
        if experts:
            for name, capacity, cost in experts:
                router.add_expert(name, float(capacity), float(cost))
        else:
            router.add_expert("small", 2.0, 0.1)
            router.add_expert("medium", 20.0, 0.5)
            router.add_expert("large", 1000.0, 1.0)

        score = module.complexity_score(data)
        expert = router.route(data)
        handled = router.route_and_handle(data)
        return ok(
            SOURCE_COMPLEXITY_ROUTER,
            complexity_score=score,
            expert=expert.name,
            expert_cost=expert.cost,
            handled=handled,
            usage=router.usage_report(),
            note=(
                "The routing decision follows from the configured "
                "capacities and the score of the vector supplied. It is not "
                "evidence that the request was genuinely easy, and no "
                "compute was saved by this call."
            ),
        )
    return guard(SOURCE_COMPLEXITY_ROUTER, _run)
