"""Target models (the bank's) and the attacker's surrogate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBClassifier

from .data import FEATURES, featurize

# Direction in which risk may only go up. Without these constraints a tree ensemble
# can learn pockets where *more* suspicious values lower the score; attackers find them.
MONOTONE = {
    "amount_ratio_log": 1, "hour_deviation": 1, "merchant_risk": 1, "velocity_1h": 1,
    "secs_since_last_log": -1, "new_device": 1, "ip_country_mismatch": 1, "ship_bill_match": -1,
    "email_age_log": -1, "account_age_log": -1,
}


@dataclass
class Scorer:
    """Wraps any model so attacks and evaluation speak raw transactions."""
    name: str
    model: object
    threshold: float = 0.5

    def score_raw(self, raw: pd.DataFrame) -> np.ndarray:
        return self.score_features(featurize(raw).to_numpy(dtype=np.float32))

    def score_features(self, x: np.ndarray) -> np.ndarray:
        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                return torch.sigmoid(self.model(torch.as_tensor(x, dtype=torch.float32))).numpy().ravel()
        return self.model.predict_proba(x)[:, 1]

    def calibrate(self, raw: pd.DataFrame, y: np.ndarray, fpr: float = 0.01) -> "Scorer":
        """Set the decline threshold so `fpr` of genuine transactions are declined."""
        s = self.score_raw(raw)
        self.threshold = float(np.quantile(s[y == 0], 1 - fpr))
        return self


def xgb(x: np.ndarray, y: np.ndarray, monotone: bool = False, seed: int = 0, weights=None) -> XGBClassifier:
    params = dict(n_estimators=350, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
                  scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1), eval_metric="aucpr",
                  random_state=seed, n_jobs=4)
    if monotone:
        params["monotone_constraints"] = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in FEATURES) + ")"
    return XGBClassifier(**params).fit(x, y, sample_weight=weights)


class MLP(nn.Module):
    def __init__(self, d: int, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def surrogate(x: np.ndarray, y: np.ndarray, epochs: int = 40, seed: int = 0) -> MLP:
    """The attacker's own model, trained on their data (e.g. which stolen-card attempts
    went through on earlier campaigns). It never sees the bank's model or data."""
    torch.manual_seed(seed)
    m = MLP(x.shape[1], x.mean(0), x.std(0) + 1e-6)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    pw = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1))
    for _ in range(epochs):
        for idx in torch.randperm(len(xt)).split(256):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(xt[idx]).squeeze(-1), yt[idx], pos_weight=pw).backward()
            opt.step()
    return m.eval()
