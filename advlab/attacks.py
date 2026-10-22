"""Evasion attacks under a realistic threat model.

greedy     Query the target directly (black box). Each query is a real attempted
           transaction, so the query count matters: it is what a probing detector sees.
transfer   Optimise against the attacker's own surrogate model offline, then submit
           once to the bank. No probing footprint.
profit     The economic attacker: try amounts from high to low and keep the highest
           amount the surrogate believes will pass. Maximises stolen value, not evasion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import Scorer
from .threat import ThreatModel, _with


@dataclass
class AttackResult:
    success: bool
    adv: pd.Series
    score: float
    queries: int
    cost: float
    path: list = field(default_factory=list)  # (field, value, score) steps


def greedy(scorer: Scorer, orig: pd.Series, threat: ThreatModel, max_queries: int = 400) -> AttackResult:
    """Best score drop per unit of effort, one field at a time, until approved."""
    cur = orig.copy()
    s = float(scorer.score_raw(cur.to_frame().T)[0])
    queries, path = 1, [("start", None, s)]
    while s >= scorer.threshold and queries < max_queries:
        moves = threat.candidates(orig, cur)[: max_queries - queries]  # every probe counts against the limit
        if not moves:
            break
        cands = pd.DataFrame([_with(cur, k, v) for k, v in moves])
        scores = scorer.score_raw(cands)
        queries += len(moves)
        base_cost = threat.cost(orig, cur)
        gain = [(s - sc) / (0.25 + threat.cost(orig, cands.iloc[i]) - base_cost) for i, sc in enumerate(scores)]
        i = int(np.argmax(gain))
        if scores[i] >= s - 1e-6:
            break  # local minimum
        cur, s = cands.iloc[i].copy(), float(scores[i])
        path.append((moves[i][0], moves[i][1], s))
    return AttackResult(s < scorer.threshold and threat.valid(orig, cur), cur, s, queries, threat.cost(orig, cur), path)


def transfer(target: Scorer, surrogate: Scorer, orig: pd.Series, threat: ThreatModel) -> AttackResult:
    """Craft against the surrogate with a safety margin, then one real attempt."""
    margin = copy.copy(surrogate)  # keep the surrogate's own scoring (subclasses featurize differently)
    margin.threshold = surrogate.threshold * 0.5
    r = greedy(margin, orig, threat, max_queries=2000)
    s = float(target.score_raw(r.adv.to_frame().T)[0])
    return AttackResult(s < target.threshold and threat.valid(orig, r.adv), r.adv, s, 1, r.cost, r.path)


def profit(target: Scorer, surrogate: Scorer, orig: pd.Series, threat: ThreatModel) -> tuple[AttackResult, float]:
    """Highest amount the attacker expects to get through; returns (result, amount stolen)."""
    hi = float(orig["amount"])
    for amount in np.geomspace(hi, threat.min_amount, 8):
        start = _with(orig, "amount", round(float(amount), 2))
        local = ThreatModel(**{**threat.__dict__, "min_amount": float(amount)})
        r = transfer(target, surrogate, start, local)
        if r.success:
            return r, float(r.adv["amount"])
    return r, 0.0


def run_attacks(target: Scorer, surrogate: Scorer, frauds: pd.DataFrame, threat: ThreatModel,
                query_limit: int | None = None) -> pd.DataFrame:
    """Attack every fraud the target currently catches. With query_limit the attacker
    is cut off after that many declined attempts (a probing detector)."""
    worth_it = frauds[frauds["amount"] >= threat.min_amount]
    caught = worth_it[target.score_raw(worth_it) >= target.threshold]
    rows = []
    for _, orig in caught.iterrows():
        g = greedy(target, orig, threat, max_queries=query_limit or 400)
        t = transfer(target, surrogate, orig, threat)
        _, stolen = profit(target, surrogate, orig, threat)
        rows.append({"greedy_success": g.success, "greedy_queries": g.queries, "greedy_cost": g.cost,
                     "transfer_success": t.success, "transfer_cost": t.cost,
                     "profit_stolen": stolen, "original_amount": float(orig["amount"])})
    return pd.DataFrame(rows)
