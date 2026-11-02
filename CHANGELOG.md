# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-11-02

### Added
- A scoring API (`python -m advlab serve`): `POST /v1/score` returns approve or decline from the
  adversarially trained model, `GET /v1/decisions` and `GET /v1/model`; every decision is logged in
  SQLite or Postgres with numbered migrations (`python -m advlab migrate`).
- Signed model bundles: with `ADVLAB_MODEL_KEY` set, `train` writes `lab.joblib.sig`, and the API only
  unpickles a bundle whose HMAC matches, from the bytes it verified. Production refuses to start
  without hashed API keys, a database, the model key and a signed bundle.
- Roles on API keys, rate limit, body cap, security headers, JSON logs with request IDs, `/healthz`,
  `/readyz` and Prometheus `/metrics`.
- `pyproject.toml`, pinned lock files (CPU-only torch from PyTorch's index, `xgboost-cpu`), a container
  image (slim Python 3.12, non-root, health check), CI with ruff, bandit, pip-audit and coverage.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`, MIT licence.

### Changed
- The simulated benchmark was re-run on the pinned versions (xgboost 3.4, torch 2.15) and
  `results/`, the README, `docs/evaluation.md` and the model card now show those numbers. They moved
  slightly (transfer evasion 43% → 42% for the baseline and 1% → 2% for the adversarially trained
  model; money stolen 14% → 12%); the conclusions are the same. The real-data results were not re-run.
- Three lint fixes (a bound loop variable, a lambda turned into a function, fixed SQL strings) with no
  change in behaviour.
