"""Tests for the bounded self-edit dials (plan phase P4).

The contract under test:
  * a dial changes the host's behaviour;
  * a dial that moves the host too far is rolled back AUTOMATICALLY;
  * removing a dial restores the host BIT-EXACTLY;
  * a dial can never be applied without a verifier being able to check it.
"""
import pytest

torch = pytest.importorskip("torch")

from venom.core.dials import DialManager, logit_drift  # noqa: E402


def _model(layers=4, dim=8):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(dim, dim)

        def forward(self, x):
            return torch.relu(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    torch.manual_seed(0)
    return Model().eval()


def _ids():
    return torch.ones(1, 4, 8)


def test_logit_drift_is_zero_for_identical_tensors():
    a = torch.randn(2, 5)
    assert logit_drift(a, a) == pytest.approx(0.0, abs=1e-9)


def test_logit_drift_grows_with_the_change():
    a = torch.randn(2, 5)
    assert logit_drift(a, a * 1.1) > 0.05


def test_logit_drift_of_mismatched_shapes_is_infinite():
    assert logit_drift(torch.randn(2, 5), torch.randn(3, 5)) == float("inf")


def test_a_dial_changes_the_output():
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.5)
    ids = _ids()
    with torch.no_grad():
        before = m(ids).clone()
    out = dm.apply(0, 0.2)
    assert out["ok"] is True
    with torch.no_grad():
        after = m(ids)
    assert not torch.equal(before, after)


def test_revert_restores_the_host_bit_exactly():
    """THE contract: after revert_all the host is bit-exact identical."""
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.5)
    ids = _ids()
    with torch.no_grad():
        before = m(ids).clone()
    dm.apply(1, 0.2)
    dm.revert_all()
    with torch.no_grad():
        after = m(ids)
    assert torch.equal(before, after)


def test_revert_of_a_single_layer_restores_bit_exactly():
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.5)
    ids = _ids()
    with torch.no_grad():
        before = m(ids).clone()
    dm.apply(0, 0.3)
    dm.apply(1, 0.3)
    out = dm.revert(0)
    assert out["ok"] is True
    with torch.no_grad():
        after = m(ids)
    assert not torch.equal(before, after), "layer 1 must still be dialled"
    dm.revert_all()
    with torch.no_grad():
        final = m(ids)
    assert torch.equal(before, final)


def test_delta_out_of_range_is_refused_without_touching_the_host():
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.1)
    ids = _ids()
    with torch.no_grad():
        before = m(ids).clone()
    out = dm.apply(0, 0.5)
    assert out["ok"] is False and out["error"] == "delta_out_of_range"
    with torch.no_grad():
        after = m(ids)
    assert torch.equal(before, after)


def test_the_default_max_delta_is_0_1():
    m = _model()
    dm = DialManager(m, list(m.blocks))
    out = dm.apply(0, 0.2)
    assert out["ok"] is False and out["error"] == "delta_out_of_range"
    assert dm.state()["active_dials"] == {}


def test_invalid_layer_is_refused():
    m = _model(layers=2)
    dm = DialManager(m, list(m.blocks))
    out = dm.apply(99, 0.05)
    assert out["ok"] is False and out["error"] == "invalid_layer"


def test_too_many_dials_is_refused():
    m = _model(layers=4)
    dm = DialManager(m, list(m.blocks), max_active_dials=2, max_delta=0.5)
    assert dm.apply(0, 0.01)["ok"]
    assert dm.apply(1, 0.01)["ok"]
    out = dm.apply(2, 0.01)
    assert out["ok"] is False and out["error"] == "too_many_dials"


def test_reapplying_to_a_layer_replaces_it_and_does_not_count_against_the_cap():
    m = _model(layers=4)
    dm = DialManager(m, list(m.blocks), max_active_dials=2, max_delta=0.5)
    dm.apply(0, 0.01)
    dm.apply(0, 0.02)
    assert dm.state()["active_dials"] == {0: 0.02}


def test_drift_exceeded_triggers_an_automatic_rollback():
    m = _model()
    ids = _ids()
    with torch.no_grad():
        before = m(ids).clone()
    dm = DialManager(m, list(m.blocks), max_delta=1.0,
                     drift_threshold=0.01, verifier=lambda: {"drift": 0.5})
    out = dm.apply(0, 0.5)
    assert out["ok"] is False
    assert out["error"] == "drift_exceeded"
    assert out["rolled_back"] is True
    with torch.no_grad():
        after = m(ids)
    assert torch.equal(before, after), "the host must be restored bit-exactly"


def test_a_verifier_without_a_drift_key_is_treated_as_a_failure():
    """A verifier that cannot measure must NOT be read as 'no drift'."""
    m = _model()
    ids = _ids()
    with torch.no_grad():
        before = m(ids).clone()
    dm = DialManager(m, list(m.blocks), max_delta=1.0,
                     drift_threshold=0.01, verifier=lambda: {"note": "cannot measure"})
    out = dm.apply(0, 0.5)
    assert out["ok"] is False
    assert out["error"] == "drift_exceeded"
    assert out.get("verifier_error")
    with torch.no_grad():
        after = m(ids)
    assert torch.equal(before, after)


def test_a_raising_verifier_is_treated_as_a_failure():
    m = _model()

    def boom():
        raise RuntimeError("no")

    dm = DialManager(m, list(m.blocks), max_delta=1.0,
                     drift_threshold=0.01, verifier=boom)
    out = dm.apply(0, 0.5)
    assert out["ok"] is False and out["error"] == "drift_exceeded"


def test_without_a_verifier_the_dial_is_applied_but_marked_unverified():
    m = _model()
    dm = DialManager(m, list(m.blocks))
    out = dm.apply(0, 0.05)
    assert out["ok"] is True
    assert out["verified"] is False


def test_state_and_is_clean():
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.5)
    assert dm.is_clean() is True
    dm.apply(0, 0.05)
    assert dm.is_clean() is False
    assert dm.state()["active_dials"] == {0: 0.05}
    dm.revert_all()
    assert dm.is_clean() is True


def test_snapshot_and_restore_of_the_dial_state():
    """The snapshot taken BEFORE an apply must restore the pre-dial state."""
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.5)
    before_apply = dm.apply(0, 0.05)
    snapshot_id = before_apply["snapshot_id"]
    assert dm.state()["active_dials"] == {0: 0.05}
    out = dm.restore(snapshot_id)
    assert out["ok"] is True
    assert dm.state()["active_dials"] == {}


def test_restore_state_accepts_a_plain_dict():
    m = _model()
    dm = DialManager(m, list(m.blocks), max_delta=0.5)
    dm.apply(0, 0.05)
    state = dm.state()
    dm.revert_all()
    out = dm.restore_state(state)
    assert out["ok"] is True
    assert dm.state()["active_dials"] == {0: 0.05}
