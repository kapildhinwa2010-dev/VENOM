"""Tests for the gated self-surgery (plan phase P5).

The contract under test:
  * a prune is never applied without a verifier;
  * a prune whose drift exceeds the threshold is cancelled and the weights
    are restored EXACTLY;
  * a prune within the threshold survives, but only with a verifier;
  * the verifier must be able to report a failure, and that failure must
    also cancel the prune.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from venom.surgeon.prune import PruneManager, PrunePlan, build_plan  # noqa: E402


def _model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.manual_seed(0)
    return Model().eval()


def test_build_plan_finds_units_that_never_fired():
    m = _model()
    counts = {"fc1": [0] * 4 + [100] * 12}
    out = build_plan(m, counts, total_samples=100, threshold=0.001)
    assert out["plan_count"] == 1
    assert out["total_units_to_prune"] == 4
    assert "NOTHING has been modified" in out["warning"] or "PLAN" in out["warning"]


def test_build_plan_ignores_a_module_with_no_samples():
    m = _model()
    out = build_plan(m, {"fc1": [100] * 16}, total_samples=100)
    assert out["plan_count"] == 0


def test_build_plan_needs_samples():
    m = _model()
    assert "error" in build_plan(m, {"fc1": [0]}, total_samples=0)


def test_prune_applies_and_restores_bit_exactly():
    m = _model()
    pm = PruneManager(m)
    before = m.fc1.weight.detach().clone()

    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[0, 1, 2], total_units=16)
    out = pm.apply(plan)
    assert out["ok"] is True
    assert not torch.equal(before, m.fc1.weight)
    assert torch.count_nonzero(m.fc1.weight[0]) == 0

    pm.restore()
    assert torch.equal(before, m.fc1.weight)


def test_a_prune_without_a_verifier_is_refused_and_restored():
    m = _model()
    pm = PruneManager(m)
    before = m.fc1.weight.detach().clone()
    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[0], total_units=16)
    pm.apply(plan)
    out = pm.verify(None)
    assert out["ok"] is False
    assert out["error"] == "no_verifier"
    assert torch.equal(before, m.fc1.weight), "the weights must be restored"


def test_drift_above_the_threshold_cancels_the_prune():
    m = _model()
    pm = PruneManager(m, drift_threshold=0.0)
    before = m.fc1.weight.detach().clone()
    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[0, 1], total_units=16)
    out = pm.prune_with_gate(plan, lambda: {"drift": 0.5})
    assert out["ok"] is False
    assert out["error"] == "drift_exceeded"
    assert out["rolled_back"] is True
    assert torch.equal(before, m.fc1.weight), "the weights must be restored exactly"


def test_drift_within_the_threshold_keeps_the_prune():
    m = _model()
    pm = PruneManager(m, drift_threshold=0.01)
    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[0], total_units=16)
    out = pm.prune_with_gate(plan, lambda: {"drift": 0.0})
    assert out["ok"] is True
    assert out["rolled_back"] is False
    assert torch.count_nonzero(m.fc1.weight[0]) == 0
    pm.restore()


def test_a_verifier_that_cannot_measure_cancels_the_prune():
    """A verifier with no drift key must not be read as 'zero drift'."""
    m = _model()
    pm = PruneManager(m, drift_threshold=1.0)
    before = m.fc1.weight.detach().clone()
    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[0], total_units=16)
    out = pm.prune_with_gate(plan, lambda: {"note": "cannot measure"})
    assert out["ok"] is False
    assert out["error"] == "verifier_must_report_drift"
    assert torch.equal(before, m.fc1.weight)


def test_a_raising_verifier_cancels_the_prune():
    m = _model()
    pm = PruneManager(m)

    def boom():
        raise RuntimeError("no")

    before = m.fc1.weight.detach().clone()
    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[0], total_units=16)
    out = pm.prune_with_gate(plan, boom)
    assert out["ok"] is False
    assert out["error"] == "verifier_raised"
    assert torch.equal(before, m.fc1.weight)


def test_is_pristine_tracks_the_snapshot():
    m = _model()
    pm = PruneManager(m)
    assert pm.is_pristine() is True
    pm.apply(PrunePlan(target_path="fc1", layer_name="Linear",
                       unit_indices=[0], total_units=16))
    assert pm.is_pristine() is False
    pm.restore()
    assert pm.is_pristine() is True


def test_pruning_an_unknown_target_is_refused():
    m = _model()
    pm = PruneManager(m)
    out = pm.apply(PrunePlan(target_path="does.not.exist", layer_name="x",
                              unit_indices=[0], total_units=1))
    assert out["ok"] is False and out["error"] == "target_not_found"


def test_a_zero_dead_unit_really_is_zero():
    """The end-to-end point: a unit that never fired is actually removable."""
    torch.manual_seed(0)
    m = _model()
    pm = PruneManager(m)
    # fc1 rows 12..15 are the ones we will call dead
    plan = PrunePlan(target_path="fc1", layer_name="Linear",
                     unit_indices=[12, 13], total_units=16)
    pm.apply(plan)
    assert torch.count_nonzero(m.fc1.weight[12]) == 0
    assert torch.count_nonzero(m.fc1.weight[13]) == 0
    pm.restore()
    assert torch.count_nonzero(m.fc1.weight[12]) > 0
