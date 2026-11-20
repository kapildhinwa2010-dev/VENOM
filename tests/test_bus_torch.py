"""Torch-level checks for the Bus trust contract.

Skipped automatically when torch is not installed, so `pytest -q` still works
in a torch-free environment.

The contract under test: attaching the bus must not change the host's output,
and detaching must restore the host bit-exactly.
"""
import pytest

torch = pytest.importorskip("torch")

from venom.core.bus import Bus, logits_hash  # noqa: E402


class TinyBlock(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class TinyModel(torch.nn.Module):
    """Mimics the GPT-2 shape the bus is written against."""

    def __init__(self, dim=32, layers=4, vocab=32):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([TinyBlock(dim) for _ in range(layers)])
        self.transformer.ln_f = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab)
        self.vocab = vocab

    def forward(self, ids):
        x = self.transformer.h[0](torch.nn.functional.one_hot(ids, self.vocab).float())
        for blk in self.transformer.h[1:]:
            x = blk(x)
        return self.lm_head(self.transformer.ln_f(x))


def _fixture():
    torch.manual_seed(0)
    model = TinyModel().eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        base = model(ids)
    return model, ids, base


def test_bus_attach_does_not_change_output():
    model, ids, base = _fixture()
    bus = Bus(model, list(model.transformer.h))
    with torch.no_grad():
        with bus:
            during = model(ids)
    assert torch.equal(base, during), "attaching the bus changed the output"


def test_bus_detach_restores_bit_exact_output():
    model, ids, base = _fixture()
    bus = Bus(model, list(model.transformer.h))
    with torch.no_grad():
        with bus:
            model(ids)
        after = model(ids)
    assert torch.equal(base, after), "detach did not restore bit-exact output"
    assert logits_hash(base) == logits_hash(after)


def test_bus_detach_clears_handles_but_keeps_taps():
    """v0.1.2: detach() must NOT wipe collected taps.

    Wiping on detach destroyed the data the bus was attached to collect.
    """
    model, ids, _ = _fixture()
    bus = Bus(model, list(model.transformer.h))
    bus.attach()
    assert bus.attached and len(bus._handles) == len(model.transformer.h)
    with torch.no_grad():
        model(ids)
    captured = dict(bus.taps)
    bus.detach()
    assert not bus.attached
    assert bus._handles == []
    assert set(bus.taps) == set(captured), "detach wiped the collected taps"
    bus.clear_taps()
    assert bus.taps == {}


def test_bus_double_attach_raises():
    model, _, _ = _fixture()
    bus = Bus(model, list(model.transformer.h))
    bus.attach()
    with pytest.raises(RuntimeError):
        bus.attach()
    bus.detach()


def test_bus_taps_every_block():
    model, ids, _ = _fixture()
    bus = Bus(model, list(model.transformer.h))
    with torch.no_grad():
        with bus:
            model(ids)
    assert bus.calls == len(model.transformer.h)
    assert set(bus.taps) == {f"block_{i}" for i in range(len(model.transformer.h))}


def test_bus_taps_are_snapshots_not_live_views():
    """A tap must be a snapshot, not a live alias of a tensor that gets mutated.

    The two forward passes must both run *inside* the bus context - outside it
    the hooks are already detached, so no new tap would be recorded.
    """
    model, ids, _ = _fixture()
    bus = Bus(model, list(model.transformer.h))
    with torch.no_grad():
        with bus:
            model(ids)
            first = bus.taps["block_0"]
            snapshot = first.clone()
            model(ids)
            second = bus.taps["block_0"]
    assert second is not first, "tap was reused instead of replaced"
    assert torch.equal(first, snapshot), "first tap was mutated by the second pass"


def test_bus_repeated_attach_detach_cycles_are_stable():
    model, ids, base = _fixture()
    for _ in range(3):
        bus = Bus(model, list(model.transformer.h))
        with torch.no_grad():
            with bus:
                model(ids)
            after = model(ids)
        assert torch.equal(base, after)


def test_bus_hook_on_tuple_output_module():
    """Some blocks return tuples; the hook must not corrupt them."""
    class TupleBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(32, 32)

        def forward(self, x):
            return (self.fc(x), None)

    class TupleModel(TinyModel):
        def forward(self, ids):
            x = torch.nn.functional.one_hot(ids, self.vocab).float()
            for blk in self.transformer.h:
                out = blk(x)
                x = out[0] if isinstance(out, tuple) else out
            return self.lm_head(self.transformer.ln_f(x))

    torch.manual_seed(0)
    model = TupleModel().eval()
    model.transformer.h = torch.nn.ModuleList([TupleBlock() for _ in range(4)])
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        base = model(ids)
    bus = Bus(model, list(model.transformer.h))
    with torch.no_grad():
        with bus:
            during = model(ids)
        after = model(ids)
    assert torch.equal(base, during)
    assert torch.equal(base, after)
