-- One row per scored transaction: the decision, the score and threshold behind it, and who asked.
-- txn_ref is the caller's own reference; raw transaction fields are not stored.
CREATE TABLE decisions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    txn_ref TEXT NOT NULL,
    model TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    score TEXT NOT NULL,
    threshold TEXT NOT NULL,
    decision TEXT NOT NULL
);
CREATE INDEX decisions_created ON decisions (created_at);
CREATE INDEX decisions_txn ON decisions (txn_ref);
