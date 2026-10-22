"""Hardening the fraud model.

monotone      XGBoost with monotonic constraints: the score can never drop when a risk
              signal gets worse. Removes the non-monotone pockets greedy attacks find.
adversarial   Iterative adversarial training: attack the current model with training
              frauds, add the successful evasions back as fraud, retrain.
probing       Block a card after a few declined attempts. Turns the black-box attack
              from 'unlimited queries' into 'a handful'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .attacks import greedy
from .data import featurize
from .models import Scorer, xgb
from .threat import ThreatModel

PROBING_LIMIT = 5  # declined attempts before the card is frozen


def train_baseline(raw, y, seed=0) -> Scorer:
    return Scorer("baseline", xgb(featurize(raw).to_numpy(np.float32), y, monotone=False, seed=seed))


def train_monotone(raw, y, seed=0) -> Scorer:
    return Scorer("monotone", xgb(featurize(raw).to_numpy(np.float32), y, monotone=True, seed=seed))


def adversarial_training(raw: pd.DataFrame, y: np.ndarray, calib: tuple, threat: ThreatModel,
                         rounds: int = 3, per_round: int = 300, seed: int = 0, log=print) -> Scorer:
    rng = np.random.default_rng(seed)
    x_raw, labels = raw.copy(), y.copy()
    scorer = train_monotone(x_raw, labels, seed).calibrate(*calib)
    frauds = raw[(y == 1) & (raw["amount"] >= threat.min_amount).to_numpy()]
    for r in range(rounds):
        sample = frauds.iloc[rng.choice(len(frauds), min(per_round, len(frauds)), replace=False)]
        found = []
        for _, orig in sample.iterrows():
            res = greedy(scorer, orig, threat, max_queries=300)
            if res.success:
                found.append(res.adv)
        log(f"  adversarial training round {r + 1}: {len(found)}/{len(sample)} training frauds evaded, adding them as fraud")
        if not found:
            break
        adv = pd.DataFrame(found)
        x_raw = pd.concat([x_raw, adv], ignore_index=True)
        labels = np.r_[labels, np.ones(len(adv), int)]
        scorer = Scorer("adv-trained", xgb(featurize(x_raw).to_numpy(np.float32), labels, monotone=True, seed=seed)).calibrate(*calib)
    scorer.name = "adv-trained"
    return scorer
