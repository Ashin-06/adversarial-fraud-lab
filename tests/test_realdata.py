import numpy as np
import pandas as pd

from advlab.attacks import greedy
from advlab.realdata import V, AmountTimeThreat, ULBScorer, featurize


class _AlwaysHigh:
    """A model that declines everything: probing can never get through."""

    def predict_proba(self, x):
        return np.c_[np.zeros(len(x)), np.ones(len(x))]


def _row(amount=250.0, t=40_000.0):
    return pd.Series({"time": t, "amount": amount, **{v: 0.1 for v in V}})


def test_threat_only_moves_amount_and_time():
    th = AmountTimeThreat()
    orig = _row()
    moves = th.candidates(orig, orig)
    assert {k for k, _ in moves} == {"amount", "time"}
    assert all(v < orig["amount"] for k, v in moves if k == "amount")
    changed = orig.copy()
    changed["V3"] = 9.0
    assert th.valid(orig, orig) and not th.valid(orig, changed)


def test_context_only_model_ignores_attacker_fields():
    raw = pd.DataFrame([_row(), _row(amount=1.0, t=3_000.0)])
    a, b = featurize(raw, controllable=False)
    assert np.array_equal(a, b) and featurize(raw).shape[1] == len(V) + 3


def test_lockout_caps_every_probe():
    s = ULBScorer("x", _AlwaysHigh(), threshold=0.5)
    r = greedy(s, _row(), AmountTimeThreat(), max_queries=5)
    assert not r.success and r.queries <= 5
