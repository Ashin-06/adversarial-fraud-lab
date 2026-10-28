# Adversarial Fraud Lab

Measures how easily a fraudster with a stolen card can get past a bank's fraud model, and which defences actually help.

Most adversarial machine-learning work attacks images by changing pixels. A card fraudster can't edit a model's features directly. They control a few real-world levers, each with a cost, and the victim's profile (account age, usual spend, usual shopping hour) stays fixed:

| Lever | Cost to the attacker |
|---|---|
| Steal a smaller amount | lower payout |
| Shop at a normal hour, at a low-risk merchant type | free |
| Slow down: fewer attempts, longer gaps | time |
| Use a VPN exit in the victim's country | small |
| Use a drop address in the billing city | medium |
| Use an aged email account | medium |
| Spoof the victim's device fingerprint | high |

Features are always recomputed from the raw fields, so every attack transaction stays internally consistent.

## Attacks and defences

| Attack | What the attacker knows |
|---|---|
| Black-box probing | only approve or decline answers; each probe is a real attempted payment |
| Transfer | trains their own model on past results, optimises offline and tries once |
| Profit-seeking | a transfer attack that steps the amount down until it expects approval |

| Defence | Idea |
|---|---|
| Monotone model | risk can never fall when a risk signal gets worse |
| Adversarial training | attack the model with its own training frauds, add the evasions back and retrain |
| Card lockout | freeze the card after 5 declines |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python -m advlab train              # defender models and the attacker's surrogate
python -m advlab attack --index 3   # step-by-step attack on one transaction
python -m advlab evaluate           # full benchmark, written to results/
streamlit run app.py                # attack studio
```

In the attack studio you pick a stolen-card transaction, set the attacker's budget, switch the lockout on or off and watch each model's fraud score fall step by step.

## Results

| Test | Result |
|---|---|
| Simulated data, unlimited probing | every model is beaten (90–100%), after about 88 tries |
| Simulated data, with a 5-decline lockout | evasion drops to 0–1% |
| Simulated data, offline transfer attack | 43% evade the baseline, 1% the adversarially trained model |
| Real card fraud (ULB), amount and time only | 3.5% evade with probing, 1.2% with lockout; 0% for a model that ignores those two fields |

Full tables and an example attack trace are in [docs/evaluation.md](docs/evaluation.md).

## Using it on your own model

```python
from advlab.models import Scorer
from advlab.threat import ThreatModel
from advlab.attacks import greedy

scorer = Scorer("prod", your_model).calibrate(holdout_raw, holdout_labels, fpr=0.01)
result = greedy(scorer, raw_transaction_row, ThreatModel(min_amount=2000, budget=4))
```

Your model needs `predict_proba` over the features from `advlab.data.featurize`, or you can wrap your own feature code. Then set `ThreatModel.costs` and `candidates` to the levers your fraudsters really have.

## Project layout

```
advlab/
  data.py      transactions and feature computation
  threat.py    which fields the attacker can change and at what cost
  models.py    XGBoost models, the attacker's surrogate, threshold calibration
  attacks.py   probing, transfer and profit-seeking attacks
  defenses.py  monotone model, adversarial training, lockout
  evaluate.py  benchmark and report
app.py         Streamlit attack studio
```

## Limitations

- The main benchmark uses simulated transactions, so its numbers compare defences rather than predict real rates.
- In the real dataset most fields are anonymised, so only amount and time could be attacked; real attackers usually control more.
- There is no API service yet; it runs as a CLI and a dashboard.
