"""E6: the dead-neuron killer on real models, and the write surface, end to end.

Four things are demonstrated, and each is reported separately because they
say different things:

1. **The census on a real model.** How many units in GPT-2 / Qwen2.5 never
   fire? The honest answer is expected to be "almost none", and if that is
   what comes out, that is what gets reported.

2. **The killer on a model that genuinely has dead units.** GPT-2 does not
   have them, so a controlled fixture is used: units are zeroed on purpose,
   the census then finds them, the killer removes them, and the drift is
   measured. This is a test of the MECHANISM, and it is labelled as such -
   it is not a claim that GPT-2 ships with dead neurons.

3. **The real shape rewrite.** `mode="remove"` shrinks the tensors and the
   downstream consumers, the model is checked that it still runs, the drift is
   measured, and the weights are restored bit-exactly.

4. **Install and the full write surface.** A model directory gets a VENOM
   manifest, is reloaded through `load_installed`, and the write surface is
   exercised on the reloaded model.

    python experiments/e6_kill_dead.py --model gpt2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.surgeon.census import census, dead_units
from venom.surgeon.kill import capture_logits, compare_logits, kill_dead_neurons
from venom.surgeon.write import WriteAccess

PLAN_PROMPTS = [
    "The capital of France is", "Hello world", "In 1969 humans landed on",
    "The largest ocean is", "Photosynthesis converts light into",
    "My favourite colour is", "Water boils at", "Shakespeare wrote",
    "In the morning I drink", "The second planet from the sun is",
    "The tallest mountain on Earth is", "Water is made of",
]
HELDOUT_PROMPTS = [
    "Rome is the capital of", "The Red Sea is in", "Venus is the",
    "Hamlet was written by", "Mount Everest is in",
]


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    return model, tok


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="e6_kill_dead.json")
    args = ap.parse_args()

    report = {"model": args.model, "sections": {}}

    # ---------------------------------------------------------------- 1
    section("1. THE CENSUS ON A REAL MODEL")
    model, tok = load(args.model)
    cen = census(model, tok, PLAN_PROMPTS)
    dead = dead_units(cen, 0.0)
    rare = {p: sum(1 for r in m["fired_rate"] if 0 < r < 0.01)
            for p, m in cen["modules"].items()}
    print(f"  calibration samples : {cen['samples']}")
    print(f"  modules scanned     : {cen['summary']['modules_scanned']}")
    print(f"  total units         : {cen['summary']['total_units']}")
    print(f"  exactly-dead units  : {cen['summary']['never_fired_units']}")
    print(f"  rarely-active (<1%) : {sum(rare.values())}")
    print(f"  modules with a dead unit: {list(dead.keys()) or 'none'}")
    report["sections"]["census_real_model"] = {
        "samples": cen["samples"],
        "summary": cen["summary"],
        "modules_with_dead_units": {k: len(v) for k, v in dead.items()},
        "rarely_active_units": rare,
        "verdict": ("no unit fired on exactly zero samples, so there is "
                    "nothing to kill" if not dead else
                    f"{sum(len(v) for v in dead.values())} candidates"),
        "caveat": cen["caveat"],
    }

    # ---------------------------------------------------------------- 2
    section("2. THE KILL PATH ON A MODEL THAT REALLY HAS DEAD UNITS")
    print("  (a controlled fixture: units are zeroed on purpose, so the")
    print("   census has something genuinely dead to find)")
    fixture, ftok = load(args.model)
    write = WriteAccess(fixture, label="fixture")
    target = None
    for path, module in fixture.named_modules():
        from venom.surgeon.write import prunable_axes
        if prunable_axes(module) is not None and "mlp" in path:
            target = path
            break
    if target is None:
        print("  no FFN module found in this architecture - skipped")
        report["sections"]["kill_fixture"] = {"skipped": "no FFN module"}
    else:
        killed_units = 12
        zeroed = write.zero_units(target, list(range(killed_units)))
        print(f"  fixture: zeroed {zeroed['zeroed_units']} units of {target}")
        cen2 = census(fixture, ftok, PLAN_PROMPTS)
        dead2 = dead_units(cen2, 0.0)
        print(f"  census now finds    : { {k: len(v) for k, v in dead2.items()} }")
        res = kill_dead_neurons(fixture, ftok, PLAN_PROMPTS, HELDOUT_PROMPTS,
                                mode="zero", threshold=0.0)
        print(f"  kill ok             : {res.get('ok')}  units: {res.get('units_removed')}")
        print(f"  drift (held-out)    : {res.get('drift')}")
        print(f"  verdict             : {res.get('verdict')}")
        report["sections"]["kill_fixture"] = {
            "target": target,
            "zeroed_for_fixture": zeroed["zeroed_units"],
            "census_found": {k: len(v) for k, v in dead2.items()},
            "kill": res,
            "note": ("the dead units here were created deliberately, so this "
                     "demonstrates the KILL MECHANISM, not a property of the "
                     "model"),
        }

    # ---------------------------------------------------------------- 3
    section("3. THE REAL SHAPE REWRITE (mode='remove')")
    model2, tok2 = load(args.model)
    write2 = WriteAccess(model2, label="rewrite")
    before = capture_logits(model2, tok2, HELDOUT_PROMPTS)
    target = None
    for path, module in model2.named_modules():
        from venom.surgeon.write import prunable_axes
        if prunable_axes(module) is not None and "mlp" in path:
            target = path
            break
    rm = write2.remove_units(target, list(range(8)))
    print(f"  target              : {rm.get('target')}")
    print(f"  removed             : {rm.get('removed_units')} units")
    print(f"  out_features        : {rm.get('out_features_before')} -> {rm.get('out_features_after')}")
    print(f"  consumers reshaped  : {rm.get('consumers_reshaped')}")
    print(f"  siblings reshaped   : {rm.get('siblings_reshaped')}")
    still_runs = True
    try:
        model2(**tok2("The capital of France is", return_tensors="pt"))
        print("  the model still runs: yes")
    except Exception as exc:                            # noqa: BLE001
        still_runs = False
        print(f"  the model still runs: NO -> {exc}")
    drift = compare_logits(before, capture_logits(model2, tok2, HELDOUT_PROMPTS))
    print(f"  drift on held-out   : {drift}")
    write2.restore()
    restored = compare_logits(before, capture_logits(model2, tok2, HELDOUT_PROMPTS))
    print(f"  after restore       : {restored}")
    report["sections"]["shape_rewrite"] = {
        "target": rm.get("target"),
        "removed_units": rm.get("removed_units"),
        "out_features_before": rm.get("out_features_before"),
        "out_features_after": rm.get("out_features_after"),
        "consumers_reshaped": rm.get("consumers_reshaped"),
        "siblings_reshaped": rm.get("siblings_reshaped"),
        "model_still_runs": still_runs,
        "drift": drift,
        "drift_after_restore": restored,
        "bit_exact_after_restore": restored["max_abs_logit_delta"] == 0.0,
        "note": ("these 8 units were NOT dead, so a real drift is expected "
                 "and correct - the point is that the rewrite is consistent, "
                 "the model still runs, and the restore is bit-exact"),
    }

    # ---------------------------------------------------------------- 4
    section("4. INSTALL + THE FULL WRITE SURFACE")
    tmp = tempfile.mkdtemp(prefix="venom-e6-")
    try:
        model3, tok3 = load(args.model)
        model3.save_pretrained(tmp)
        tok3.save_pretrained(tmp)
        from venom.install import install, is_installed, load_installed

        inst = install(tmp, label="e6")
        print(f"  installed           : {inst.get('installed')}")
        print(f"  manifest            : {os.path.basename(inst.get('manifest',''))}")
        print(f"  writable modules    : {inst.get('writable_modules')}")
        print(f"  note                : {inst.get('note')}")
        loaded, ltok, manifest, lwrite = load_installed(tmp)
        print(f"  reloaded via manifest: family={manifest['fingerprint']['family']} "
              f"layers={manifest['fingerprint']['num_layers']}")
        surface = lwrite.describe()
        print(f"  write surface       : {surface['writable_modules']} modules, "
              f"journal entries={surface['journal_entries']}")
        wpath = manifest["writable_surface"][0]["path"]
        mut = lwrite.zero_units(wpath, [0])
        print(f"  mutation            : {mut.get('target')} zeroed "
              f"{mut.get('zeroed_units')} unit(s)")
        print(f"  journal             : {[e['action'] for e in lwrite.journal()]}")
        report["sections"]["install"] = {
            "installed": inst.get("installed"),
            "writable_modules": inst.get("writable_modules"),
            "fingerprint": manifest["fingerprint"],
            "guarantees": manifest["guarantees"],
            "non_guarantees": manifest["non_guarantees"],
            "mutation": mut,
            "journal": lwrite.journal(),
            "note": "installing changed no weight; the model became addressable",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
