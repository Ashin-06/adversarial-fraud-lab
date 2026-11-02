"""Evasion on real card fraud.

ULB / Worldline credit-card fraud data (OpenML #1597): 284,807 real card transactions,
492 frauds. V1-V28 are PCA components of the confidential raw fields, so nobody - neither
the bank's analysts nor an attacker - knows which real-world lever moves them. The only
fields the attacker demonstrably controls are the amount (split the theft into smaller
charges) and when the charge is made. That is the threat model here: amount down to
EUR 1 and any hour of the day, everything else fixed. It is an approximation (in the raw
data a different amount may also shift some components), stated rather than hidden.

Defences compared:
  baseline         XGBoost on every feature, threshold at 1% false-positive rate
  adv-trained      evasions found on training frauds are added back as fraud, retrained
  context-only     the same model without the two attacker-controlled fields
and the probing lockout (card frozen after 5 declines) applied to the black-box attack.

    python -m advlab realdata --data creditcard.pq
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .attacks import greedy, transfer
from .defenses import PROBING_LIMIT
from .models import Scorer

V = [f"V{i}" for i in range(1, 29)]
DAY = 86_400


def load(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path) if str(path).endswith((".pq", ".parquet")) else pd.read_csv(path)
    raw = df[["Time", "Amount", *V]].rename(columns={"Time": "time", "Amount": "amount"}).astype(float)
    return raw, df["Class"].astype(int).to_numpy()


def featurize(raw: pd.DataFrame, controllable: bool = True) -> np.ndarray:
    cols = [raw[V].to_numpy(float)]
    if controllable:
        hour = (raw["time"].to_numpy(float) % DAY) / 3600
        cols.append(np.c_[np.log1p(raw["amount"].to_numpy(float)), np.sin(hour / 24 * 2 * np.pi), np.cos(hour / 24 * 2 * np.pi)])
    return np.hstack(cols).astype(np.float32)


class ULBScorer(Scorer):
    def __init__(self, name, model, threshold=0.5, controllable=True):
        super().__init__(name, model, threshold)
        self.controllable = controllable

    def score_raw(self, raw: pd.DataFrame) -> np.ndarray:
        return self.score_features(featurize(raw.astype({c: float for c in raw.columns}), self.controllable))


@dataclass
class AmountTimeThreat:
    """The attacker picks the amount (>= min_amount) and the hour; nothing else."""
    min_amount: float = 1.0
    budget: float = 0.0

    def cost(self, orig: pd.Series, adv: pd.Series) -> float:
        return 0.0  # both levers are free; lowering the amount costs the attacker money, measured separately

    def candidates(self, orig: pd.Series, cur: pd.Series) -> list[tuple[str, float]]:
        out = [("amount", round(float(a), 2)) for a in np.geomspace(self.min_amount, max(cur["amount"], self.min_amount), 6)[:-1]]
        day_start = cur["time"] - cur["time"] % DAY
        for h in range(0, 24, 3):
            t = day_start + h * 3600 + cur["time"] % 3600
            if abs(t - cur["time"]) >= 1800:
                out.append(("time", float(t)))
        return out

    def valid(self, orig: pd.Series, adv: pd.Series) -> bool:
        return bool((adv[V] == orig[V]).all() and adv["amount"] >= self.min_amount - 1e-6)


def _xgb(x, y, seed=0, **kw) -> XGBClassifier:
    params = dict(n_estimators=350, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
                  scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1), eval_metric="aucpr", random_state=seed, n_jobs=4)
    return XGBClassifier(**{**params, **kw}).fit(x, y)


def _calibrate(s: ULBScorer, raw, y, fpr=0.01) -> ULBScorer:
    p = s.score_raw(raw)
    s.threshold = float(np.quantile(p[y == 0], 1 - fpr))
    return s


def _profit(target, surrogate, orig, threat) -> float:
    """Highest amount the attacker, guided by their own model, gets approved."""
    if float(orig["amount"]) < threat.min_amount:  # zero-value card checks: nothing to steal
        return 0.0
    for amount in np.geomspace(float(orig["amount"]), threat.min_amount, 8):
        start = orig.copy()
        start["amount"] = round(float(amount), 2)
        r = transfer(target, surrogate, start, AmountTimeThreat(min_amount=float(amount)))
        if r.success:
            return float(r.adv["amount"])
    return 0.0


def run(path: str | Path, seed: int = 0, log=print) -> dict:
    raw, y = load(path)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train, other = train_test_split(rest, test_size=0.375, stratify=y[rest], random_state=seed)   # 50% / 30% of all
    calib, attacker = train_test_split(other, test_size=2 / 3, stratify=y[other], random_state=seed)  # 10% / 20%
    def part(ix):
        return raw.iloc[ix].reset_index(drop=True), y[ix]

    (tr_raw, tr_y), cal, (te_raw, te_y), (at_raw, at_y) = part(train), part(calib), part(test), part(attacker)
    threat = AmountTimeThreat()

    log("training defender models")
    base = _calibrate(ULBScorer("baseline", _xgb(featurize(tr_raw), tr_y, seed)), *cal)
    context = _calibrate(ULBScorer("context-only", _xgb(featurize(tr_raw, False), tr_y, seed), controllable=False), *cal)
    # adversarial training: evade the baseline with training frauds, add the evasions as fraud
    found = [greedy(base, orig, threat, max_queries=300) for _, orig in tr_raw[tr_y == 1].iterrows()]
    adv = pd.DataFrame([r.adv for r in found if r.success])
    log(f"  adversarial training: {len(adv)}/{len(found)} training frauds evaded the baseline")
    aug_raw, aug_y = pd.concat([tr_raw, adv], ignore_index=True), np.r_[tr_y, np.ones(len(adv), int)]
    hardened = _calibrate(ULBScorer("adv-trained", _xgb(featurize(aug_raw), aug_y, seed)), *cal)
    surrogate = _calibrate(ULBScorer("surrogate", _xgb(featurize(at_raw), at_y, seed + 1, max_depth=4, n_estimators=250)), at_raw, at_y)

    frauds = te_raw[te_y == 1]
    res = {"dataset": "ULB / Worldline card fraud (OpenML #1597)", "test_frauds": int(len(frauds)),
           "threat_model": "amount (down to EUR 1) and hour of day; V1-V28 fixed", "models": {}}
    for m in (base, hardened, context):
        p = m.score_raw(te_raw)
        caught = frauds[m.score_raw(frauds) >= m.threshold]
        log(f"attacking {m.name}: {len(caught)} test frauds caught")
        g = [greedy(m, o, threat, max_queries=400) for _, o in caught.iterrows()]
        lim = [greedy(m, o, threat, max_queries=PROBING_LIMIT) for _, o in caught.iterrows()]
        t = [transfer(m, surrogate, o, threat) for _, o in caught.iterrows()]
        stolen = [_profit(m, surrogate, o, threat) for _, o in caught.iterrows()]
        n = max(len(caught), 1)
        res["models"][m.name] = {
            "pr_auc": round(float(average_precision_score(te_y, p)), 4),
            "recall_at_1pct_fpr": round(float((p[te_y == 1] >= m.threshold).mean()), 4),
            "frauds_caught": int(len(caught)),
            "greedy_evasion": round(sum(r.success for r in g) / n, 4),
            "greedy_median_queries": float(np.median([r.queries for r in g if r.success])) if any(r.success for r in g) else None,
            "greedy_evasion_with_probing_lockout": round(sum(r.success for r in lim) / n, 4),
            "transfer_evasion": round(sum(r.success for r in t) / n, 4),
            "value_stolen_share": round(sum(stolen) / max(float(caught["amount"].sum()), 1e-9), 4),
        }
    return res


def main(data: str, out: str = "results/real_data.json", seed: int = 0) -> dict:
    res = run(data, seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res
