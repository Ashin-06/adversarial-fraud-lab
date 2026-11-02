# Runbook

How to train, sign, deploy and operate the scoring service. Commands assume the repository root.

## Train and sign

```bash
export ADVLAB_MODEL_KEY=$(python -c "import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())")
python -m advlab train        # artifacts/lab.joblib and artifacts/lab.joblib.sig
```

Replace the generated training data with your own labelled transactions (see "Using it on your own
model" in the README) before using it for real decisions.

## Deploy

```bash
docker build -t adversarial-fraud-lab:1.0.0 .
python -m advlab.ops keygen          # API keys per scoring client, viewer and monitor
docker run -d --name advlab -p 8005:8005 -v advlab-data:/data -v $PWD/artifacts:/models:ro \
  -e ADVLAB_KEYS_FILE=/run/secrets/advlab-keys.json -e ADVLAB_MODEL_KEY=... \
  adversarial-fraud-lab:1.0.0
curl -fsS http://127.0.0.1:8005/readyz
```

The image runs in production mode as a non-root user. It refuses to start, and logs why, without
keys, the model key or a correctly signed bundle at `/models/lab.joblib`.

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `ADVLAB_ENV` | — | `development` (default) or `production` |
| `ADVLAB_API_KEYS` / `ADVLAB_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `scorer`, `viewer`, `analyst` (score and read), `monitor`, `admin` |
| `ADVLAB_MODEL_PATH` | yes | the signed bundle from `train` |
| `ADVLAB_MODEL_KEY` | yes | base64, at least 32 bytes; `ADVLAB_MODEL_KEY_PREVIOUS` for rotation |
| `ADVLAB_DATABASE_URL` | yes | the decision log: `sqlite:////data/advlab.db` or `postgresql://...` |
| `ADVLAB_HOST`, `ADVLAB_PORT` | — | default `127.0.0.1:8005` |
| `ADVLAB_RATE_LIMIT`, `ADVLAB_MAX_BODY_BYTES` | — | default 120/min per client (raise it for the payment switch), 256,000 bytes |

## Watch

- `GET /healthz`; `GET /readyz` (database and model); `GET /metrics` (role `monitor`): requests,
  latency, errors, `advlab_decisions_total{decision="approve|decline"}`.

Suggested alerts: the decline share moving sharply (a model problem or an attack wave); many declines
for one card in a short window (probing: lock the card upstream).

## Rotate keys

**Model key.** Set `ADVLAB_MODEL_KEY_PREVIOUS` to the old key and `ADVLAB_MODEL_KEY` to a new one,
re-sign the bundle, restart, then drop `_PREVIOUS`. **API keys.** Add the new hash, restart, hand over
the key, remove the old hash, restart.

## Back up and restore

Keep the bundle and its signature with your model artefacts; back up the decision log like any SQLite
or Postgres database.

## Incidents

1. **An evasion pattern is seen in production.** Find the transactions in `/v1/decisions`, add the
   lever to `ThreatModel`, retrain with adversarial training, re-run `evaluate` and redeploy.
2. **Leaked scorer key.** Remove its hash and restart: it is a probing oracle for attackers.
