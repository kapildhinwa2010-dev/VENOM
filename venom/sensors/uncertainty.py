"""Pure-python per-token uncertainty. No torch required - works on raw logit rows."""
from __future__ import annotations

import math
from typing import Dict, List

ENTROPY_THRESHOLD = 2.0
MARGIN_THRESHOLD = 0.1


def token_uncertainty(logits_row: List[float], k: int = 5) -> Dict[str, float]:
    max_l = max(logits_row)
    exps = [math.exp(l - max_l) for l in logits_row]
    sum_exp = sum(exps)
    probs = [e / sum_exp for e in exps]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
    ordered = sorted(probs, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = top1 - top2
    spread = 1.0 - sum(ordered[:k])
    return {
        "entropy": round(entropy, 4),
        "margin": round(margin, 4),
        "top_k_spread": round(spread, 4),
        "confidence": round(top1, 4),
        "is_uncertain": entropy > ENTROPY_THRESHOLD or margin < MARGIN_THRESHOLD,
        "top1_prob": round(top1, 4),
        "top2_prob": round(top2, 4),
    }


def sequence_uncertainty(all_logits: List[List[float]], k: int = 5) -> List[Dict[str, float]]:
    return [token_uncertainty(row, k) for row in all_logits]
