# Model card: adversarially trained fraud scorer

XGBoost with monotone constraints, retrained on evasions found by attacking it (adversarial
training), with the decline threshold set at a 1% false-positive rate. Compared against a plain
XGBoost baseline and a monotone-only model. Numbers come from `docs/evaluation.md` and `results/`.

## Intended use

Scoring card-not-present transactions where fraudsters adapt to the model, behind a payment flow
that also locks cards after repeated declines. It is a research and demonstration model trained on
simulated data; it needs retraining on the bank's own transactions before real use.

## Evaluation

**Simulated card transactions** (40,000 rows; 105 caught stolen-card frauds attacked per model):

| Model | PR-AUC | Black-box evasion | With 5-decline lockout | Transfer evasion | Money stolen / attempted |
|---|---|---|---|---|---|
| Baseline | 0.999 | 100% | 2% | 42% | 40% |
| Monotone | 0.999 | 100% | 0% | 50% | 41% |
| Adversarially trained | 0.990 | 90% | 1% | 2% | 12% |

**Real card fraud** (ULB / Worldline, amount and time are the only attackable fields):

| Model | PR-AUC | Black-box evasion | With lockout | Transfer evasion |
|---|---|---|---|---|
| Baseline XGBoost | 0.838 | 3.5% | 1.2% | 1.2% |
| Adversarially trained | 0.838 | 3.5% | 1.2% | 1.2% |
| Without amount and time | 0.825 | 0% | 0% | 0% |

## Known failure modes

- With unlimited probing every model is beaten; robustness depends on a lockout outside the model.
- The threat model is hand-written: levers it doesn't include (for example a compromised merchant)
  aren't defended.
- On the real data adversarial training had nothing to learn (no training fraud could be pushed under
  the threshold), so it is identical to the baseline there.
- The main benchmark is simulated, so it compares defences rather than predicting real rates.

## Before relying on it

Retrain on the bank's own labelled transactions, write the threat model from real fraud cases, and
re-run `evaluate` and `realdata`-style tests on held-out data.
