"""Scoring service: decisions from the adversarially trained fraud model.

    POST /v1/score           score up to 500 raw transactions; decline or approve each [score]
    GET  /v1/decisions       recent decisions [read]
    GET  /v1/model           which model is loaded, its threshold and the defenders' clean metrics [read]
    GET  /healthz, /readyz, /metrics

The decision comes from the adversarially trained XGBoost model at the threshold calibrated
for a 1% false-decline rate. The model bundle is the one `python -m advlab train` writes; in
production ADVLAB_MODEL_PATH must point at it and its .sig must match ADVLAB_MODEL_KEY,
because the bundle is a pickle. Settings: ops.py and docs/runbook.md.

    python -m advlab serve
"""
from __future__ import annotations

import math
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import pandas as pd
from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from . import db, ops
from .data import MCC_NAMES
from .signing import load_verified

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
MODEL_PATH = os.environ.get("ADVLAB_MODEL_PATH", "").strip()
if settings.production and not MODEL_PATH:
    raise ops.ConfigError("production needs ADVLAB_MODEL_PATH: the signed bundle from `python -m advlab train`")


def _load():
    if MODEL_PATH:
        try:
            lab, digest = load_verified(MODEL_PATH, [settings.secret["MODEL_KEY"], *settings.previous["MODEL_KEY"]])
        except (OSError, ValueError) as e:
            raise ops.ConfigError(f"cannot load ADVLAB_MODEL_PATH: {e}") from None
    else:  # development: the CLI's own artifact, built on first use
        from .__main__ import load_or_build

        lab, digest = load_or_build(), "development (unsigned)"
        ops.log.warning("no ADVLAB_MODEL_PATH: serving artifacts/lab.joblib, which is not signature-checked")
    _, _, models, _ = lab
    hardened = next(m for m in models if m.name == "adv-trained")
    return hardened, digest


MODEL, MODEL_SHA = _load()
engine = db.connect(settings.database_url)
_lock = threading.Lock()

app = FastAPI(title="Adversarially robust fraud scoring", version="1.0.0")
ops.install(app, settings, {"database": lambda: engine.connect().close(), "model": lambda: MODEL is not None})


class Transaction(BaseModel):
    txn_ref: str = Field(..., min_length=1, max_length=64)
    account_age_days: float = Field(..., ge=0, le=36_500, allow_inf_nan=False)
    avg_amount_30d: float = Field(..., gt=0, le=1e9, allow_inf_nan=False)
    usual_hour: float = Field(..., ge=0, lt=24, allow_inf_nan=False)
    amount: float = Field(..., gt=0, le=1e9, allow_inf_nan=False)
    hour: float = Field(..., ge=0, lt=24, allow_inf_nan=False)
    mcc: Literal[tuple(MCC_NAMES)]  # type: ignore[valid-type]
    velocity_1h: float = Field(..., ge=0, le=10_000, allow_inf_nan=False)
    secs_since_last: float = Field(..., ge=0, le=1e9, allow_inf_nan=False)
    new_device: bool
    ip_country_mismatch: bool
    ship_bill_match: bool
    email_age_days: float = Field(..., ge=0, le=36_500, allow_inf_nan=False)


class ScoreRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=500)


@app.post("/v1/score")
def score(req: ScoreRequest, ident: dict = Depends(ops.require("score"))) -> Dict[str, Any]:
    raw = pd.DataFrame([t.model_dump(exclude={"txn_ref"}) for t in req.transactions])
    scores = MODEL.score_raw(raw)
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    out = []
    with _lock, engine.begin() as c:
        for t, s in zip(req.transactions, scores):
            decision = "decline" if s >= MODEL.threshold else "approve"
            ops.DECISIONS.labels(decision=decision).inc()
            c.execute(text("INSERT INTO decisions VALUES (:i, :t, :u, :r, :m, :h, :s, :th, :d)"),
                      {"i": uuid.uuid4().hex, "t": now, "u": ident["user"] or "demo", "r": t.txn_ref, "m": MODEL.name,
                       "h": MODEL_SHA, "s": f"{float(s):.6f}", "th": f"{MODEL.threshold:.6f}", "d": decision})
            out.append({"txn_ref": t.txn_ref, "score": round(float(s), 6), "decision": decision})
    return {"model": MODEL.name, "threshold": round(MODEL.threshold, 6), "results": out}


@app.get("/v1/decisions", dependencies=[Depends(ops.require("read"))])
def decisions(txn_ref: str | None = Query(None, max_length=64), limit: int = Query(100, ge=1, le=1000)) -> Dict[str, Any]:
    sql = ("SELECT * FROM decisions WHERE txn_ref = :r ORDER BY created_at DESC LIMIT :n" if txn_ref
           else "SELECT * FROM decisions ORDER BY created_at DESC LIMIT :n")
    with engine.connect() as c:
        rows = c.execute(text(sql), {"r": txn_ref, "n": limit}).mappings().all()
    return {"decisions": [dict(r) for r in rows]}


@app.get("/v1/model", dependencies=[Depends(ops.require("read"))])
def model_info() -> Dict[str, Any]:
    t = MODEL.threshold
    return {"model": MODEL.name, "threshold": t if math.isfinite(t) else None, "sha256": MODEL_SHA,
            "false_decline_target": "1% of genuine transactions (calibration split)"}
