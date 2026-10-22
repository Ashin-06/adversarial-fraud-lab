"""End-to-end evaluation: train defender models, attack them, write a report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .attacks import run_attacks
from .data import featurize, generate, split
from .defenses import PROBING_LIMIT, adversarial_training, train_baseline, train_monotone
from .models import Scorer, surrogate
from .threat import ThreatModel


def clean_metrics(s: Scorer, raw, y) -> dict:
    p = s.score_raw(raw)
    return {"pr_auc": float(average_precision_score(y, p)), "recall_at_1pct_fpr": float((p[y == 1] >= s.threshold).mean())}


def build(seed: int = 0, n: int = 40_000, log=print):
    raw, y = generate(n, seed=seed)
    parts = split(raw, y, seed)
    threat = ThreatModel()
    calib = parts["calib"]
    log("training defender models")
    models = [train_baseline(*parts["train"], seed).calibrate(*calib), train_monotone(*parts["train"], seed).calibrate(*calib)]
    models.append(adversarial_training(*parts["train"], calib, threat, seed=seed, log=log))
    log("training the attacker's surrogate on its own data")
    xa, ya = featurize(parts["attacker"][0]).to_numpy(np.float32), parts["attacker"][1]
    sur = Scorer("surrogate", surrogate(xa, ya, seed=seed)).calibrate(*parts["attacker"])
    return parts, threat, models, sur


def main(out: str = "results", n_attacks: int = 120, seed: int = 0, quick: bool = False):
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts, threat, models, sur = build(seed, 16_000 if quick else 40_000)
    test_raw, test_y = parts["test"]
    frauds = test_raw[test_y == 1]
    frauds = frauds.iloc[: (40 if quick else n_attacks)]

    rows, per_model = [], {}
    for m in models:
        cm = clean_metrics(m, test_raw, test_y)
        print(f"attacking {m.name} ...")
        res = run_attacks(m, sur, frauds, threat)
        lim = run_attacks(m, sur, frauds, threat, query_limit=PROBING_LIMIT)
        per_model[m.name] = res
        stolen = res["profit_stolen"].sum()
        rows.append({
            "model": m.name, "pr_auc": cm["pr_auc"], "recall@1%FPR": cm["recall_at_1pct_fpr"],
            "frauds attacked": len(res),
            "black-box evasion": res["greedy_success"].mean() if len(res) else 0.0,
            "median queries": float(res.loc[res["greedy_success"], "greedy_queries"].median()) if res["greedy_success"].any() else float("nan"),
            f"evasion with {PROBING_LIMIT}-attempt lockout": lim["greedy_success"].mean() if len(lim) else 0.0,
            "transfer evasion": res["transfer_success"].mean() if len(res) else 0.0,
            "money stolen / attempted": stolen / max(res["original_amount"].sum(), 1),
            "mean attacker effort": float(res.loc[res["greedy_success"], "greedy_cost"].mean()) if res["greedy_success"].any() else float("nan"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)
    print(table.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    cols = ["black-box evasion", f"evasion with {PROBING_LIMIT}-attempt lockout", "transfer evasion", "money stolen / attempted"]
    x = np.arange(len(table))
    for i, c in enumerate(cols):
        ax[0].bar(x + (i - 1.5) * 0.2, table[c], 0.2, label=c)
    ax[0].set_xticks(x, table["model"])
    ax[0].set_ylabel("share")
    ax[0].set_title("Attack success by defence")
    ax[0].legend(fontsize=8)
    for name, res in per_model.items():
        q = res.loc[res["greedy_success"], "greedy_queries"]
        if len(q):
            ax[1].hist(q, bins=20, alpha=0.5, label=name)
    ax[1].axvline(PROBING_LIMIT, ls="--", c="k", label="lockout")
    ax[1].set_xlabel("queries (attempted transactions) to evade")
    ax[1].set_title("How many tries a black-box attacker needs")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    json.dump({"seed": seed, "threat_model": threat.__dict__, "seconds": round(time.time() - t0, 1)},
              open(out / "run.json", "w"), indent=2, default=str)
    print(f"\nwritten to {out}/ in {time.time() - t0:.0f}s")
    return table


if __name__ == "__main__":
    main()
