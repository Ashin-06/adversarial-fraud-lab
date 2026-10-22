"""Card-not-present transactions with raw fields, so attacks happen in the attacker's
world (amount, time, merchant, device, VPN...) rather than on abstract feature vectors.

Features are always *derived* from raw fields through `featurize`, which keeps every
adversarial example internally consistent: lowering the amount also lowers the
amount-to-average ratio, shopping at 2 pm changes the hour deviation, and so on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MCC = {"grocery": 0.05, "fuel": 0.10, "restaurants": 0.10, "travel": 0.35, "electronics": 0.60,
       "digital_goods": 0.70, "gift_cards": 0.90}
MCC_NAMES = list(MCC)

RAW_COLUMNS = ["account_age_days", "avg_amount_30d", "usual_hour", "amount", "hour", "mcc", "velocity_1h",
               "secs_since_last", "new_device", "ip_country_mismatch", "ship_bill_match", "email_age_days"]

FEATURES = ["amount_log", "amount_ratio_log", "hour_sin", "hour_cos", "hour_deviation", "merchant_risk",
            "velocity_1h", "secs_since_last_log", "new_device", "ip_country_mismatch", "ship_bill_match",
            "email_age_log", "account_age_log", "avg_amount_log"] + [f"mcc_{m}" for m in MCC_NAMES]


def featurize(raw: pd.DataFrame) -> pd.DataFrame:
    # rows built one at a time (attacks) arrive as object dtype; normalise first
    raw = raw.astype({c: float for c in RAW_COLUMNS if c != "mcc"})
    f = pd.DataFrame(index=raw.index)
    f["amount_log"] = np.log1p(raw["amount"])
    f["amount_ratio_log"] = np.log(raw["amount"] / raw["avg_amount_30d"])
    ang = raw["hour"] / 24 * 2 * np.pi
    f["hour_sin"], f["hour_cos"] = np.sin(ang), np.cos(ang)
    d = np.abs(raw["hour"] - raw["usual_hour"]) % 24
    f["hour_deviation"] = np.minimum(d, 24 - d)
    f["merchant_risk"] = raw["mcc"].map(MCC).astype(float)
    f["velocity_1h"] = raw["velocity_1h"].astype(float)
    f["secs_since_last_log"] = np.log1p(raw["secs_since_last"])
    for c in ("new_device", "ip_country_mismatch", "ship_bill_match"):
        f[c] = raw[c].astype(float)
    f["email_age_log"] = np.log1p(raw["email_age_days"])
    f["account_age_log"] = np.log1p(raw["account_age_days"])
    f["avg_amount_log"] = np.log1p(raw["avg_amount_30d"])
    for m in MCC_NAMES:
        f[f"mcc_{m}"] = (raw["mcc"] == m).astype(float)
    return f[FEATURES]


def _customers(rng, n):
    return pd.DataFrame({
        "account_age_days": rng.gamma(2.0, 500, n).clip(5, 6000).round(),
        "avg_amount_30d": np.exp(rng.normal(7.2, 0.7, n)).round(2),  # ~₹1,300 typical
        "usual_hour": np.clip(rng.normal(15, 3, n), 7, 22).round(),
    })


def generate(n: int = 40_000, fraud_rate: float = 0.03, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_f = int(n * fraud_rate)
    n_l = n - n_f

    legit = _customers(rng, n_l)
    legit["amount"] = (legit["avg_amount_30d"] * np.exp(rng.normal(0, 0.6, n_l))).round(2)
    legit["hour"] = ((legit["usual_hour"] + rng.normal(0, 2.5, n_l)).round() % 24)
    legit["mcc"] = rng.choice(MCC_NAMES, n_l, p=[0.30, 0.15, 0.25, 0.08, 0.12, 0.07, 0.03])
    legit["velocity_1h"] = rng.poisson(0.4, n_l)
    legit["secs_since_last"] = np.exp(rng.normal(10.5, 1.4, n_l)).round()
    legit["new_device"] = rng.random(n_l) < 0.04
    legit["ip_country_mismatch"] = rng.random(n_l) < 0.03
    legit["ship_bill_match"] = rng.random(n_l) < 0.92
    legit["email_age_days"] = rng.gamma(3, 600, n_l).round()

    fraud = _customers(rng, n_f)  # stolen cards belong to ordinary customers
    fraud["amount"] = (fraud["avg_amount_30d"] * np.exp(rng.normal(1.4, 0.7, n_f))).round(2)
    fraud["hour"] = rng.integers(0, 24, n_f)
    fraud["mcc"] = rng.choice(MCC_NAMES, n_f, p=[0.05, 0.03, 0.05, 0.12, 0.30, 0.20, 0.25])
    fraud["velocity_1h"] = rng.poisson(2.2, n_f)
    fraud["secs_since_last"] = np.exp(rng.normal(7.0, 1.6, n_f)).round()
    fraud["new_device"] = rng.random(n_f) < 0.75
    fraud["ip_country_mismatch"] = rng.random(n_f) < 0.55
    fraud["ship_bill_match"] = rng.random(n_f) < 0.30
    fraud["email_age_days"] = np.where(rng.random(n_f) < 0.6, rng.integers(0, 30, n_f), rng.gamma(2, 400, n_f).round())

    raw = pd.concat([legit, fraud], ignore_index=True)[RAW_COLUMNS]
    y = np.r_[np.zeros(n_l), np.ones(n_f)].astype(int)
    order = rng.permutation(len(raw))
    return raw.iloc[order].reset_index(drop=True), y[order]


def split(raw: pd.DataFrame, y: np.ndarray, seed: int = 0):
    """train (defender) / calibration / test, plus a separate attacker dataset used to
    train the surrogate: the attacker never sees the defender's training data."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(raw))
    a, b, c = int(0.5 * len(idx)), int(0.6 * len(idx)), int(0.8 * len(idx))
    parts = {"train": idx[:a], "calib": idx[a:b], "test": idx[b:c], "attacker": idx[c:]}
    return {k: (raw.iloc[v].reset_index(drop=True), y[v]) for k, v in parts.items()}
