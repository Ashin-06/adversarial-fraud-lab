"""The scoring API: settings, the signed model bundle, auth, limits, headers, metrics, health,
the decision log and migrations."""
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from advlab import db, ops
from advlab.evaluate import build
from advlab.signing import load_verified, sign

MODEL_KEY = b"m" * 32
MK = base64.b64encode(MODEL_KEY).decode()
KEY = "advlab_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "switch", "role": "scorer"}}
TXN = {"txn_ref": "T-1", "account_age_days": 900, "avg_amount_30d": 1800, "usual_hour": 14, "amount": 1500, "hour": 15,
       "mcc": "grocery", "velocity_1h": 1, "secs_since_last": 86400, "new_device": False, "ip_country_mismatch": False,
       "ship_bill_match": True, "email_age_days": 2000}
RISKY = {**TXN, "txn_ref": "T-2", "account_age_days": 3, "amount": 48000, "hour": 3, "mcc": "gift_cards", "velocity_1h": 9,
         "secs_since_last": 40, "new_device": True, "ip_country_mismatch": True, "ship_bill_match": False, "email_age_days": 1}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("lab")
    lab = build(0, n=8000, log=lambda *_: None)  # small, so the test stays quick
    joblib.dump(lab, d / "lab.joblib")
    sign(d / "lab.joblib", MODEL_KEY)
    joblib.dump(lab, d / "unsigned.joblib")
    return d


def boot(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith("ADVLAB_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(f"ADVLAB_{k}", v)
    import advlab.api as api

    return importlib.reload(api)


def signed(bundle, **env):
    return {"MODEL_PATH": str(bundle / "lab.joblib"), "MODEL_KEY": MK, **env}


def test_production_needs_a_signed_bundle(monkeypatch, bundle):
    prod = {"ENV": "production", "API_KEYS": json.dumps(HASHED), "DATABASE_URL": "sqlite://"}
    with pytest.raises(ops.ConfigError, match="MODEL_KEY"):
        boot(monkeypatch, **prod, MODEL_PATH=str(bundle / "lab.joblib"))
    with pytest.raises(ops.ConfigError, match="MODEL_PATH"):
        boot(monkeypatch, **prod, MODEL_KEY=MK)
    with pytest.raises(ops.ConfigError, match="no signature"):
        boot(monkeypatch, **prod, MODEL_KEY=MK, MODEL_PATH=str(bundle / "unsigned.joblib"))
    with pytest.raises(ops.ConfigError, match="not signed"):
        boot(monkeypatch, **prod, MODEL_KEY=base64.b64encode(b"k" * 32).decode(), MODEL_PATH=str(bundle / "lab.joblib"))
    api = boot(monkeypatch, **prod, **signed(bundle))
    assert api.MODEL.name == "adv-trained" and len(api.MODEL_SHA) == 64
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"ADVLAB_HOST": "0.0.0.0"})
    with pytest.raises(ValueError):
        load_verified(bundle / "unsigned.joblib", [MODEL_KEY])


def test_scores_are_decided_logged_and_kept_across_a_restart(monkeypatch, bundle, tmp_path):
    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'd.db'}"
    c = TestClient(boot(monkeypatch, **signed(bundle, DATABASE_URL=url)).app)
    r = c.post("/v1/score", json={"transactions": [TXN, RISKY]}).json()
    assert [x["txn_ref"] for x in r["results"]] == ["T-1", "T-2"] and r["model"] == "adv-trained"
    assert r["results"][0]["decision"] == "approve" and r["results"][1]["decision"] == "decline"
    assert c.post("/v1/score", json={"transactions": [{**TXN, "amount": -5}]}).status_code == 422
    assert c.post("/v1/score", json={"transactions": [{**TXN, "mcc": "casino"}]}).status_code == 422

    c = TestClient(boot(monkeypatch, **signed(bundle, DATABASE_URL=url)).app)  # restart
    got = c.get("/v1/decisions", params={"txn_ref": "T-2"}).json()["decisions"]
    assert got and got[0]["decision"] == "decline" and got[0]["requested_by"] == "demo"
    assert c.get("/v1/model").json()["model"] == "adv-trained"


def test_roles_and_metrics(monkeypatch, bundle):
    keys = {"k-sw": {"user": "switch", "role": "scorer"}, "k-view": {"user": "vic", "role": "viewer"},
            "k-mon": {"user": "prom", "role": "monitor"}}
    c = TestClient(boot(monkeypatch, **signed(bundle, API_KEYS=json.dumps(keys))).app)
    body = {"transactions": [TXN]}
    assert c.post("/v1/score", json=body).status_code == 401
    assert c.post("/v1/score", json=body, headers={"x-api-key": "k-view"}).status_code == 403
    assert c.post("/v1/score", json=body, headers={"x-api-key": "k-sw"}).status_code == 200
    assert c.get("/v1/decisions", headers={"x-api-key": "k-sw"}).status_code == 403
    assert c.get("/v1/decisions", headers={"x-api-key": "k-view"}).status_code == 200
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'advlab_decisions_total{decision="approve"}' in m.text


def test_limits_headers_errors_and_health(monkeypatch, bundle):
    c = TestClient(boot(monkeypatch, **signed(bundle, RATE_LIMIT="3")).app)
    codes = [c.get("/v1/model").status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]
    api = boot(monkeypatch, **signed(bundle, MAX_BODY_BYTES="400"))

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: card 4111111111111111")

    c = TestClient(api.app)
    assert c.post("/v1/score", json={"transactions": [TXN, RISKY]}).status_code == 413
    crash = c.get("/boom")
    assert crash.status_code == 500 and "4111" not in crash.text and crash.json()["request_id"]
    bad = c.post("/v1/score", json={"transactions": [{**TXN, "txn_ref": "SECRET-REF", "hour": "late"}]})
    assert bad.status_code == 422 and "late" not in bad.text
    r = c.get("/v1/model", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.headers["content-security-policy"] == ops.API_CSP
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/readyz").json()["checks"] == {"database": True, "model": True}


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"] and db.migrate(eng) == []
    with eng.connect() as c:
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert {"decisions", "schema_migrations"} <= tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("ADVLAB_")}, "ADVLAB_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "advlab", "migrate"], env=env, capture_output=True, text=True,
                         timeout=300, check=False)
    assert out.returncode == 0 and "up to date" in out.stdout
