"""Install VENOM into a saved model directory.

The request this answers: *"put VENOM into the model, and give it access to
change anything inside it."*

What "install" means here, precisely
------------------------------------
A model on disk is a directory of tensors. Installing VENOM writes a
**manifest** into that directory declaring:

* the model's fingerprint (family, where the blocks live, how many layers);
* the **writable surface** - every projection VENOM is allowed to write to,
  with its shape, so the surface can be audited before anything is touched;
* the capabilities the installed host has;
* the guarantees that hold, and the ones that do not.

Loading an installed model then hands back a `WriteAccess` object - the
unrestricted, journaled, reversible write surface - so the caller can change
anything, and every change is recorded and can be rolled back.

What this is NOT
----------------
It does not modify the model's behaviour by itself. Installing does not
change a single weight. It makes the model **addressable**: from then on,
VENOM can find the model, fingerprint it, census it, and write to it.

The guarantees are honest:

* attach/detach is bit-exact - installing changes nothing about the tensors;
* every mutation is journaled and snapshotted;
* a shape rewrite is undoable because the pre-rewrite modules are kept;
* nothing here prevents you from damaging the model. It makes damage
  *reversible and visible*, not impossible.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

MANIFEST_NAME = "venom.manifest.json"
MANIFEST_VERSION = 1


def manifest_path(model_path: str) -> str:
    return os.path.join(model_path, MANIFEST_NAME)


def is_installed(model_path: str) -> bool:
    return os.path.exists(manifest_path(model_path))


def read_manifest(model_path: str) -> Optional[Dict[str, Any]]:
    path = manifest_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def install(model_path: str, *, label: str = "host",
            overwrite: bool = True) -> Dict[str, Any]:
    """Install VENOM into a saved model directory. Writes the manifest."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from venom import __version__
    from venom.core.discovery import fingerprint_universal
    from venom.surgeon.write import WriteAccess

    if not os.path.isdir(model_path):
        return {"ok": False, "error": "not_a_directory", "path": model_path}
    if is_installed(model_path) and not overwrite:
        return {"ok": False, "error": "already_installed",
                "path": manifest_path(model_path)}

    model = AutoModelForCausalLM.from_pretrained(model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:                                    # noqa: BLE001
        tokenizer = None

    fp = fingerprint_universal(model, model_path)
    write = WriteAccess(model, label=label)
    surface = write.describe()

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "venom_version": __version__,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": os.path.abspath(model_path),
        "label": label,
        "fingerprint": {
            "family": fp.get("family"),
            "model_type": fp.get("model_type"),
            "blocks_path": fp.get("blocks_path"),
            "num_layers": fp.get("num_layers"),
            "hidden_size": fp.get("hidden_size"),
            "lm_head_path": fp.get("lm_head_path"),
            "norm_path": fp.get("norm_path"),
            "discovery_confidence": fp.get("confidence"),
        },
        "writable_surface": surface["modules"],
        "writable_module_count": surface["writable_modules"],
        "capabilities": [
            "census", "kill", "write", "dial", "prune", "audit",
            "uncertainty", "check_myself", "why", "where",
            "memory", "verify", "snapshot", "restore",
        ],
        "guarantees": {
            "install_changes_no_weights": True,
            "attach_detach_bit_exact": True,
            "all_mutations_journaled": True,
            "all_mutations_snapshotted": True,
            "shape_rewrite_reversible": True,
        },
        "non_guarantees": [
            "VENOM does not prevent a caller from damaging the model.",
            "A drift within the threshold on a held-out set is evidence, not proof.",
            "Pruning buys memory and compute, not intelligence.",
            "The census is sample-dependent: a unit that never fired may still fire.",
        ],
    }

    with open(manifest_path(model_path), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "ok": True,
        "installed": True,
        "manifest": manifest_path(model_path),
        "fingerprint": manifest["fingerprint"],
        "writable_modules": manifest["writable_module_count"],
        "note": "no weight was changed; the model is now addressable by VENOM",
    }


def load_installed(model_path: str, *, label: str = "host"
                   ) -> Tuple[Any, Any, Dict[str, Any], Any]:
    """Load an installed model and return the full write surface with it.

    Returns (model, tokenizer, manifest, write_access).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from venom.surgeon.write import WriteAccess

    manifest = read_manifest(model_path)
    if manifest is None:
        raise FileNotFoundError(
            f"{model_path} is not VENOM-installed. Run `venom install {model_path}`.")

    model = AutoModelForCausalLM.from_pretrained(model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:                                    # noqa: BLE001
        tokenizer = None
    write = WriteAccess(model, label=label)
    return model, tokenizer, manifest, write


def uninstall(model_path: str) -> Dict[str, Any]:
    """Remove the manifest. The weights were never touched, so nothing to undo."""
    path = manifest_path(model_path)
    if not os.path.exists(path):
        return {"ok": False, "error": "not_installed", "path": path}
    os.remove(path)
    return {"ok": True, "removed": path,
            "note": "only the manifest was removed; no weight was ever changed"}
