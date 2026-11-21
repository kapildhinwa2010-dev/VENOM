"""Memory integrations: NEVER_LIE, FactBase, KNOW-VAULT, fluidtech.

These four siblings are all *memory* implementations, but they are not
interchangeable and the adapters do not pretend otherwise:

* NEVER_LIE  - a slot memory whose reads are digest-guarded (abstain on
  corruption).
* FactBase   - a (subject, predicate, object) fact store with indexed
  queries.
* KNOW-VAULT - an estimator of how many mutually non-overlapping "slots"
  a key matrix holds.
* fluidtech  - the fluidlayer FrozenRegister: an append-only, 20-byte,
  write-once slot table with a fail-loud read path.

VENOM's own memory (`venom.memory.store.MemoryStore`) remains the default
and the fallback.  Nothing here replaces it; these are optional
alternatives that a caller can reach for explicitly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from venom.integrations._base import fail, guard, ok, require

SOURCE_NEVER_LIE = "NEVER_LIE"
SOURCE_FACTBASE = "FactBase"
SOURCE_KNOW_VAULT = "KNOW-VAULT"
SOURCE_FLUIDTECH = "fluidtech"


# --------------------------------------------------------------------------- #
# NEVER_LIE
# --------------------------------------------------------------------------- #

def never_lie_available() -> bool:
    """Is the NEVER_LIE sibling importable?"""
    from venom.integrations._loader import load
    return load("never_lie") is not None


def never_lie_run(facts: Optional[Dict[str, Any]] = None,
                  n_corruptions: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Run the sibling's own fuzz routine over a set of facts.

    What it does: hands the facts to NEVER_LIE's ``fuzz()``, which stores
    them, corrupts random slots and counts how many reads came back
    correct, abstained or wrong.

    What it does NOT prove: that the digest guard stops a *deliberate*
    attacker - the sibling says so itself ("use an HMAC for adversarial
    settings"), and this is corruption detection, not cryptographic
    security.  It also says nothing about VENOM's own ``MemoryStore``,
    which is a different implementation with a different (shorter) digest.
    """
    def _run():
        module, failure = require("never_lie", SOURCE_NEVER_LIE)
        if failure is not None:
            return failure
        payload = dict(facts) if facts else {
            "capital_of_France": "Paris",
            "capital_of_Japan": "Tokyo",
            "boiling_point_water_c": 100,
        }
        report = module.fuzz(payload, n_corruptions=int(n_corruptions),
                             seed=int(seed))
        return ok(
            SOURCE_NEVER_LIE,
            facts=len(payload),
            corruptions=int(n_corruptions),
            report=report,
            note=(
                "The sibling's fuzz routine counted correct / abstain / wrong "
                "reads after random bit flips. 'wrong: 0' means the digest "
                "guard caught these corruptions; it is not a security "
                "guarantee against an attacker who rewrites fact and digest "
                "together."
            ),
        )
    return guard(SOURCE_NEVER_LIE, _run)


# --------------------------------------------------------------------------- #
# FactBase
# --------------------------------------------------------------------------- #

def factbase_available() -> bool:
    """Is the FactBase sibling importable?"""
    from venom.integrations._loader import load
    return load("factbase") is not None


def factbase_run(triples: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Store triples in the sibling's KnowledgeNet and query them back.

    What it does: adds (subject, predicate, object) triples to FactBase's
    ``KnowledgeNet`` and reads them back through its indexed query.

    What it does NOT prove: that the facts are *true*, that the query
    answers are entailed, or that FactBase reasons - it is a storage
    layer.  A round-trip through it only shows the store keeps and returns
    what it was given.
    """
    def _run():
        module, failure = require("factbase", SOURCE_FACTBASE)
        if failure is not None:
            return failure
        data = list(triples) if triples else [
            ("France", "capital", "Paris"),
            ("Japan", "capital", "Tokyo"),
            ("Paris", "country", "France"),
        ]
        net = module.KnowledgeNet()
        added = 0
        for item in data:
            subject, predicate, obj = item
            if net.add(subject, predicate, obj):
                added += 1
        stored = net.facts()
        first = stored[0] if stored else None
        query = []
        if first is not None:
            query = [f.as_tuple() for f in net.query(subject=first.subject)]
        return ok(
            SOURCE_FACTBASE,
            added=added,
            total=len(stored),
            triples=[list(f.as_tuple()) for f in stored],
            subjects=net.subjects(),
            predicates=net.predicates(),
            query_by_subject=query,
            note=(
                "A storage round-trip only: it shows the sibling's indexed "
                "query returns the triples that were put in. It is not a "
                "truth or entailment check."
            ),
        )
    return guard(SOURCE_FACTBASE, _run)


# --------------------------------------------------------------------------- #
# KNOW-VAULT
# --------------------------------------------------------------------------- #

def know_vault_available() -> bool:
    """Is the KNOW-VAULT sibling importable?"""
    from venom.integrations._loader import load
    return load("know_vault") is not None


def know_vault_run(keys: Optional[Any] = None,
                   activations: Optional[Any] = None,
                   cos_tol: float = 0.5) -> Dict[str, Any]:
    """Ask the sibling how many distinct slots a key matrix can hold.

    What it does: calls KNOW-VAULT's ``vault_report`` on a key matrix
    (rows = neurons, columns = embedding dimension) and, when activations
    are supplied, also reports which neurons never fire.

    What it does NOT prove: that a model can store that many facts, or
    that the matrix handed in is the model's actual memory.  The sibling's
    slot count is a *greedy, order-dependent* estimate of how many rows
    are not near-duplicates of each other - a capacity estimate, not a
    measurement of recall.
    """
    def _run():
        module, failure = require("know_vault", SOURCE_KNOW_VAULT)
        if failure is not None:
            return failure
        import numpy as np

        if keys is None:
            base = np.eye(6, dtype=float)
            extra = np.stack([base[0] * 0.99, base[1] * 0.98])
            matrix = np.concatenate([base, extra], axis=0)
        else:
            matrix = np.asarray(keys, dtype=float)
        acts = None if activations is None else np.asarray(activations, dtype=float)
        report = module.vault_report(matrix, acts, cos_tol=float(cos_tol))
        return ok(
            SOURCE_KNOW_VAULT,
            cos_tol=float(cos_tol),
            report=report,
            note=(
                "The sibling's slot estimate is greedy and depends on row "
                "order. It estimates capacity of the matrix it was given; "
                "it does not measure whether a model stores facts in it."
            ),
        )
    return guard(SOURCE_KNOW_VAULT, _run)


# --------------------------------------------------------------------------- #
# fluidtech
# --------------------------------------------------------------------------- #

def fluidtech_available() -> bool:
    """Is the fluidtech sibling importable?"""
    from venom.integrations._loader import load
    return load("fluidlayer") is not None


def fluidtech_run(facts: Optional[List[Any]] = None,
                  capacity: int = 64) -> Dict[str, Any]:
    """Use the sibling's FrozenRegister as an exact, append-only fact table.

    What it does: promotes facts into fluidtech's ``FrozenRegister``
    (20-byte write-once slots) and reads them back through its verifying
    ``recall``, including the fail-loud path where a mismatched embedding
    signature returns nothing instead of another fact's value.

    What it does NOT prove: that the facts are correct, and nothing at all
    about fluidtech's *fluid* tier - only the L1 register is exercised
    here.  The register stores 32-bit integer ids, so the adapter maps
    text facts to ids and back; the round-trip is exact for the id, not
    for arbitrary text.
    """
    def _run():
        module, failure = require("fluidlayer", SOURCE_FLUIDTECH)
        if failure is not None:
            return failure
        import numpy as np

        data = list(facts) if facts else [
            ("capital_of_france", "Paris"),
            ("capital_of_japan", "Tokyo"),
        ]
        register = module.FrozenRegister(capacity=int(capacity))
        index = {}
        slots = []
        for i, item in enumerate(data):
            key_text = str(item[0])
            key = module.ctx_key([ord(c) for c in key_text])
            sig = module.embedding_signature(
                np.array([ord(c) / 255.0 for c in key_text[:4]], dtype=float)
            )
            slot = register.promote(val=i, key=key, sig=sig)
            index[key] = {"text": key_text, "value": item[1], "sig": sig}
            slots.append(slot)

        recalled = []
        for key, meta in index.items():
            got = register.recall(key, meta["sig"])
            recalled.append({
                "key": meta["text"],
                "found": got is not None,
                "value": meta["value"] if got is not None else None,
                "slot": got[0] if got is not None else None,
            })

        # fail-loud path: a wrong signature must NOT return a value.
        first_key = next(iter(index))
        loud = register.recall(first_key, sig=(int(index[first_key]["sig"]) + 1) & 0xFFFFFFFF)
        return ok(
            SOURCE_FLUIDTECH,
            capacity=int(capacity),
            facts=len(data),
            recalled=recalled,
            fail_loud_missed=loud is None,
            stats=register.stats(),
            note=(
                "Only the L1 FrozenRegister is exercised. 'fail_loud_missed' "
                "shows the register refuses a slot whose signature does not "
                "match instead of returning a wrong value."
            ),
        )
    return guard(SOURCE_FLUIDTECH, _run)
