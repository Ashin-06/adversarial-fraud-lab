# Threat model

What the scoring service protects, from whom, and what is still open. Written for the 1.0.0 code;
nobody independent has reviewed it.

## What it protects

| Asset | Where it lives |
|---|---|
| The fraud decision on each transaction | returned to the caller; logged in `decisions` |
| The trained model bundle | a signed `lab.joblib` at `ADVLAB_MODEL_PATH` |
| The model signing key | `ADVLAB_MODEL_KEY` |
| API keys | callers hold the key; the service holds only SHA-256 hashes |

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| A fraudster probes the model until a transaction passes | Probing needs many attempts (median 84–95); lockout after 5 declines belongs in the payment flow | **The API does not lock cards.** With unlimited probing, 90% of attacks still succeed against the hardened model in simulation. Enforce a lockout in front of it. |
| A transfer attack (attacker's own surrogate, one try) | Adversarial training: 2% evasion against 42% for the baseline (simulated data) | Measured on synthetic data with one threat model |
| A malicious model bundle runs code when loaded (pickle) | Only a bundle whose HMAC matches the model key is loaded, from the verified bytes | Anyone holding the key can sign a bundle |
| Scores leak model internals to an attacker | Scores are returned to authenticated `scorer` keys only | A scorer key held by a compromised merchant integration gives an attacker a probing oracle |
| Errors leak data | Validation errors don't echo input; unhandled errors return only a request ID | — |
| Flooding | 500 transactions per call, 256 KB body cap, per-client rate limit | Rate limit is per process |

## Not covered

- Card lockout, step-up authentication and velocity limits in the payment flow.
- TLS (terminate it at the ingress).
