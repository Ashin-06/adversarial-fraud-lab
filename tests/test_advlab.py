import numpy as np
import pandas as pd
import pytest

from advlab.attacks import greedy
from advlab.data import FEATURES, featurize, generate, split
from advlab.defenses import train_baseline, train_monotone
from advlab.models import MONOTONE
from advlab.threat import ThreatModel


@pytest.fixture(scope="module")
def lab():
    raw, y = generate(8000, seed=1)
    parts = split(raw, y, 1)
    base = train_baseline(*parts["train"], 1).calibrate(*parts["calib"])
    mono = train_monotone(*parts["train"], 1).calibrate(*parts["calib"])
    return parts, base, mono


def test_features_stay_consistent_with_raw_fields():
    raw, _ = generate(200, seed=0)
    f = featurize(raw)
    assert list(f.columns) == FEATURES
    assert np.allclose(f["amount_ratio_log"], np.log(raw["amount"] / raw["avg_amount_30d"]))
    one = featurize(raw.iloc[0].to_frame().T)  # object-dtype single row from an attack
    assert np.allclose(one.to_numpy(float), f.iloc[[0]].to_numpy(float))


def test_threat_model_never_touches_victim_profile(lab):
    parts, base, _ = lab
    raw, y = parts["test"]
    t = ThreatModel()
    fraud = raw[(y == 1) & (raw["amount"] >= t.min_amount).to_numpy()].iloc[0]
    r = greedy(base, fraud, t)
    for k in ("account_age_days", "avg_amount_30d", "usual_hour"):
        assert r.adv[k] == fraud[k]
    assert r.adv["amount"] >= t.min_amount and r.cost <= t.budget


def test_cost_accounting():
    t = ThreatModel()
    o = pd.Series({"new_device": True, "ip_country_mismatch": True, "ship_bill_match": False,
                   "email_age_days": 3.0, "velocity_1h": 3, "secs_since_last": 60.0})
    a = o.copy()
    a["new_device"], a["ip_country_mismatch"] = False, False
    assert t.cost(o, a) == pytest.approx(2.5)


def test_models_detect_fraud_and_respect_monotonicity(lab):
    parts, base, mono = lab
    raw, y = parts["test"]
    assert (base.score_raw(raw)[y == 1] >= base.threshold).mean() > 0.8
    row = raw[y == 1].iloc[[0]].copy()
    for col, sign in (("new_device", 1), ("ship_bill_match", -1)):
        lo, hi = row.copy(), row.copy()
        lo[col], hi[col] = False, True
        d = mono.score_raw(hi)[0] - mono.score_raw(lo)[0]
        assert d * sign >= -1e-6, col
    assert sum(1 for v in MONOTONE.values() if v) >= 8


def test_lockout_limits_black_box_attacks(lab):
    parts, base, _ = lab
    raw, y = parts["test"]
    t = ThreatModel()
    frauds = raw[(y == 1) & (raw["amount"] >= t.min_amount).to_numpy()]
    caught = frauds[base.score_raw(frauds) >= base.threshold].iloc[:15]
    free = np.mean([greedy(base, r, t).success for _, r in caught.iterrows()])
    locked = np.mean([greedy(base, r, t, max_queries=5).success for _, r in caught.iterrows()])
    assert free > locked
