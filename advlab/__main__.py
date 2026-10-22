"""Command line.

    python -m advlab train                 # build defender models + attacker surrogate -> artifacts/
    python -m advlab attack --index 3      # walk one attack step by step against each model
    python -m advlab evaluate [--quick]    # full benchmark -> results/
    python -m advlab realdata --data creditcard.pq   # attacks on real card fraud -> results/real_data.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from .attacks import greedy, profit, transfer
from .evaluate import build, main as evaluate

ART = Path("artifacts/lab.joblib")


def load_or_build(seed: int = 0):
    if ART.exists():
        return joblib.load(ART)
    lab = build(seed)
    ART.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lab, ART)
    return lab


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="advlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("attack")
    a.add_argument("--index", type=int, default=0, help="which caught test fraud to attack")
    e = sub.add_parser("evaluate")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--attacks", type=int, default=120)
    e.add_argument("--out", default="results")
    r = sub.add_parser("realdata", help="evasion on the real ULB card-fraud data")
    r.add_argument("--data", required=True, help="OpenML #1597 parquet or Kaggle creditcard.csv")
    r.add_argument("--out", default="results/real_data.json")
    args = ap.parse_args(argv)

    if args.cmd == "train":
        ART.unlink(missing_ok=True)
        parts, threat, models, sur = load_or_build(args.seed)
        for m in models:
            print(f"{m.name:12s} decline threshold {m.threshold:.3f} (1% false-positive rate)")
        print(f"saved {ART}")
    elif args.cmd == "attack":
        parts, threat, models, sur = load_or_build()
        raw, y = parts["test"]
        frauds = raw[(y == 1) & (raw["amount"] >= threat.min_amount).to_numpy()]
        caught = frauds[models[0].score_raw(frauds) >= models[0].threshold]
        orig = caught.iloc[args.index % len(caught)]
        shown = ("amount", "hour", "mcc", "velocity_1h", "new_device", "ip_country_mismatch", "ship_bill_match", "email_age_days")
        print("original fraud:", ", ".join(f"{k}={getattr(orig[k], 'item', lambda: orig[k])()}" for k in shown))
        for m in models:
            r = greedy(m, orig, threat)
            print(f"\n[{m.name}] black-box greedy: {'EVADED' if r.success else 'caught'} after {r.queries} attempts, effort {r.cost:.1f}")
            for f, v, s in r.path:
                print(f"   {f:22s} -> {str(v):12s} score {s:.3f}  (decline at {m.threshold:.3f})")
            tr = transfer(m, sur, orig, threat)
            _, stolen = profit(m, sur, orig, threat)
            print(f"   transfer attack: {'EVADED' if tr.success else 'caught'};  profit-seeking attacker gets ₹{stolen:,.0f} of ₹{orig['amount']:,.0f}")
    elif args.cmd == "realdata":
        import json

        from .realdata import main as realdata

        print(json.dumps(realdata(args.data, args.out), indent=2))
    else:
        evaluate(args.out, args.attacks, quick=args.quick)


if __name__ == "__main__":
    main()
