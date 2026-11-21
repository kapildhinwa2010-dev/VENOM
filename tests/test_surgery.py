"""Tests for the surgery stack: census, the write surface, and the killer.

These use a tiny synthetic model so they run in a second and need no
downloads. The real-model behaviour is covered by
`experiments/e6_kill_dead.py`.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from venom.install import (install, is_installed, load_installed,  # noqa: E402
                            read_manifest, uninstall)
from venom.surgeon.census import census, dead_units  # noqa: E402
from venom.surgeon.kill import (capture_logits, compare_logits,  # noqa: E402
                                kill_dead_neurons)
from venom.surgeon.write import (WriteAccess, input_dim,  # noqa: E402
                                 output_dim, prunable_axes)


# --------------------------------------------------------------- fixtures
class TinyFFN(nn.Module):
    """Stands in for an FFN block: an expansion and its consumer."""

    def __init__(self, d_in=8, d_hidden=16, d_out=8):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden)
        self.up_proj = nn.Linear(d_in, d_hidden)
        self.down_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.block = TinyFFN()
        self.head = nn.Linear(8, 32)

    def forward(self, input_ids):
        return self.head(self.block(self.embed(input_ids)))


class TinyTokenizer:
    """A tokenizer good enough for the census and the drift check."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt"):
        ids = [(abs(hash(tok)) % 30) + 1 for tok in text.split()][:6] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


PROMPTS = ["alpha beta gamma", "delta epsilon", "zeta eta theta", "iota kappa"]
HELDOUT = ["lambda mu", "nu xi omicron"]


def make_all_units_live(model):
    """Force every FFN unit to fire, so 'nothing is dead' is a real result
    and not an accident of the random initialisation."""
    with torch.no_grad():
        model.block.gate_proj.weight.normal_(0.0, 1.0)
        model.block.gate_proj.bias.fill_(1.0)
        model.block.up_proj.weight.normal_(0.0, 1.0)
        model.block.up_proj.bias.fill_(1.0)
    return model


# ------------------------------------------------------------------ axes
def test_prunable_axes_resolves_output_axis():
    lin = nn.Linear(4, 6)
    weight, axis = prunable_axes(lin)
    assert axis == 0
    assert output_dim(lin) == 6
    assert input_dim(lin) == 4


def test_conv1d_output_axis_is_one():
    from transformers.pytorch_utils import Conv1D

    conv = Conv1D(6, 4)          # nf=6, nx=4 -> weight [4, 6]
    weight, axis = prunable_axes(conv)
    assert axis == 1
    assert output_dim(conv) == 6
    assert input_dim(conv) == 4


def test_write_access_describes_the_surface():
    w = WriteAccess(TinyModel())
    described = w.describe()
    assert described["writable_modules"] >= 3
    assert "block.gate_proj" in [m["path"] for m in described["modules"]]
    assert "not prevention" in described["warning"]


# --------------------------------------------------------------- write
def test_zero_units_changes_only_the_named_units():
    model = TinyModel()
    w = WriteAccess(model)
    before = model.block.gate_proj.weight.detach().clone()
    out = w.zero_units("block.gate_proj", [0, 3])
    assert out["ok"] and out["zeroed_units"] == 2
    after = model.block.gate_proj.weight.detach()
    assert torch.all(after[0] == 0) and torch.all(after[3] == 0)
    assert torch.equal(before[1], after[1])


def test_zero_units_is_reversible_bit_exactly():
    model = TinyModel()
    w = WriteAccess(model)
    before = model.block.gate_proj.weight.detach().clone()
    w.zero_units("block.gate_proj", [0, 3, 5])
    w.restore()
    assert torch.equal(before, model.block.gate_proj.weight.detach())
    assert w.is_pristine()


def test_journal_records_every_mutation():
    model = TinyModel()
    w = WriteAccess(model)
    w.zero_units("block.gate_proj", [1])
    w.set_attr("block", "scale", 2.0)
    actions = [e["action"] for e in w.journal()]
    assert "zero_units" in actions
    assert "set_attr" in actions


def test_remove_units_rewrites_the_shape_and_the_consumers():
    model = TinyModel()
    w = WriteAccess(model)
    out = w.remove_units("block.gate_proj", [0, 1, 2])
    assert out["ok"]
    assert out["removed_units"] == 3
    assert model.block.gate_proj.weight.shape[0] == 13
    # the consumer (down_proj) and the sibling (up_proj) must have shrunk too
    assert model.block.down_proj.weight.shape[1] == 13
    assert model.block.up_proj.weight.shape[0] == 13
    assert model.block.gate_proj.bias.shape[0] == 13


def test_remove_units_only_touches_the_targets_own_parent():
    """The bug that broke GPT-2: a global search for consumers with a
    matching input dim reshaped every block, not just the target's."""
    model = TinyModel()
    model.other = TinyFFN()
    w = WriteAccess(model)
    w.remove_units("block.gate_proj", [0])
    assert model.other.down_proj.weight.shape[1] == 16, (
        "an unrelated block was reshaped - the consumer search is not scoped "
        "to the target's parent")


def test_remove_units_refuses_when_nothing_consumes_the_output():
    model = TinyModel()
    w = WriteAccess(model)
    out = w.remove_units("head", [0])
    assert out["ok"] is False
    assert out["error"] == "no_consumer_in_parent"


def test_remove_units_can_be_undone_bit_exactly():
    model = TinyModel()
    w = WriteAccess(model)
    before = model.block.gate_proj.weight.detach().clone()
    w.remove_units("block.gate_proj", [0, 1])
    w.restore()
    assert model.block.gate_proj.weight.shape == before.shape
    assert torch.equal(before, model.block.gate_proj.weight.detach())


def test_load_state_dict_is_snapshot_first():
    model = TinyModel()
    w = WriteAccess(model)
    other = TinyModel()
    out = w.load_state_dict(other.state_dict())
    assert out["ok"]
    assert w.is_pristine() is False
    w.restore()
    assert w.is_pristine()


# -------------------------------------------------------------- census
def test_census_counts_per_unit():
    model = TinyModel()
    out = census(model, TinyTokenizer(), PROMPTS, ffn_only=False)
    assert out["ok"]
    assert out["samples"] == len(PROMPTS)
    info = out["modules"]["block.gate_proj"]
    assert info["out_features"] == 16
    assert len(info["fired_count"]) == 16
    assert sum(info["fired_count"]) > 0


def test_census_finds_units_that_never_fired():
    model = TinyModel()
    with torch.no_grad():
        model.block.gate_proj.weight[0] = 0.0
        model.block.gate_proj.bias[0] = 0.0
    out = census(model, TinyTokenizer(), PROMPTS, ffn_only=False)
    dead = dead_units(out, 0.0)
    assert "block.gate_proj" in dead
    assert 0 in dead["block.gate_proj"]


def test_dead_units_threshold_zero_means_exactly_zero():
    """A strict `<` comparison against 0.0 never matches, which silently
    turns the killer into a no-op. This is the regression test for it."""
    model = TinyModel()
    with torch.no_grad():
        model.block.gate_proj.weight[0] = 0.0
        model.block.gate_proj.bias[0] = 0.0
    out = census(model, TinyTokenizer(), PROMPTS, ffn_only=False)
    assert dead_units(out, 0.0), "threshold 0 must select exactly-zero units"
    assert dead_units(out, 0.001), "a nonzero threshold is a superset"


# --------------------------------------------------------------- killer
def test_kill_refuses_without_a_heldout_set():
    model = TinyModel()
    out = kill_dead_neurons(model, TinyTokenizer(), PROMPTS, [])
    assert out["ok"] is False
    assert out["error"] == "no_heldout_set"


def test_kill_refuses_when_nothing_is_dead():
    model = make_all_units_live(TinyModel())
    out = kill_dead_neurons(model, TinyTokenizer(), PROMPTS, HELDOUT)
    assert out["ok"] is False
    assert out["error"] == "nothing_dead"


def test_kill_removes_dead_units_and_keeps_the_model_running():
    model = make_all_units_live(TinyModel())
    with torch.no_grad():
        model.block.gate_proj.weight[0] = 0.0
        model.block.gate_proj.bias[0] = 0.0
    out = kill_dead_neurons(model, TinyTokenizer(), PROMPTS, HELDOUT,
                            mode="zero", threshold=0.0)
    assert out["ok"] is True
    assert out["units_removed"] == 1
    assert out["drift"]["max_abs_logit_delta"] == 0.0
    assert out["rolled_back"] is False


def test_kill_rolls_back_when_the_drift_is_too_large():
    model = make_all_units_live(TinyModel())
    with torch.no_grad():
        model.block.gate_proj.weight[0] = 0.0
        model.block.gate_proj.bias[0] = 0.0
    before = model.block.gate_proj.weight.detach().clone()
    out = kill_dead_neurons(model, TinyTokenizer(), PROMPTS, HELDOUT,
                            mode="zero", threshold=0.0,
                            drift_threshold=-1.0)   # impossible threshold
    assert out["ok"] is False
    assert out["error"] == "drift_exceeded"
    assert out["rolled_back"] is True
    assert torch.equal(before, model.block.gate_proj.weight.detach())


def test_kill_in_remove_mode_reshapes_the_model():
    model = make_all_units_live(TinyModel())
    with torch.no_grad():
        model.block.gate_proj.weight[0] = 0.0
        model.block.gate_proj.bias[0] = 0.0
    out = kill_dead_neurons(model, TinyTokenizer(), PROMPTS, HELDOUT,
                            mode="remove", threshold=0.0)
    assert out["ok"] is True
    assert model.block.gate_proj.weight.shape[0] == 15
    assert model.block.down_proj.weight.shape[1] == 15


def test_compare_logits_reports_agreement():
    a = [torch.tensor([1.0, 2.0, 3.0])]
    b = [torch.tensor([1.0, 2.0, 3.5])]
    out = compare_logits(a, b)
    assert out["ok"]
    assert out["max_abs_logit_delta"] == pytest.approx(0.5)
    assert out["top1_agreement"] == 1.0


# -------------------------------------------------------------- install
def _tiny_hf_model():
    """A real HuggingFace model, built locally - no download."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=64,
                     n_positions=32, n_inner=64)
    return GPT2LMHeadModel(cfg)


def test_install_writes_a_manifest_and_changes_no_weight(tmp_path):
    model = _tiny_hf_model()
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    model.save_pretrained(model_dir)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = install(str(model_dir))
    assert out["ok"]
    assert out["note"].startswith("no weight was changed")
    # re-load and compare: installing must not have touched a single tensor
    from transformers import AutoModelForCausalLM
    again = AutoModelForCausalLM.from_pretrained(str(model_dir))
    for k, v in again.state_dict().items():
        assert torch.equal(before[k], v), f"install changed {k}"
    assert out["ok"]
    assert is_installed(str(model_dir))
    manifest = read_manifest(str(model_dir))
    assert manifest["guarantees"]["install_changes_no_weights"] is True
    assert manifest["writable_module_count"] > 0
    uninstall(str(model_dir))
    assert not is_installed(str(model_dir))


def test_load_installed_returns_a_write_surface(tmp_path):
    model = _tiny_hf_model()
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    model.save_pretrained(model_dir)
    install(str(model_dir))

    loaded, tokenizer, manifest, write = load_installed(str(model_dir))
    assert manifest["writable_module_count"] > 0
    described = write.describe()
    assert described["writable_modules"] == manifest["writable_module_count"]
    # the write surface must actually reach the loaded model
    target = manifest["writable_surface"][0]["path"]
    out = write.zero_units(target, [0])
    assert out["ok"]
    assert write.journal()
