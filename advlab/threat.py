"""Threat model: what a card fraudster can actually change, what it costs them, and
what they need to stay profitable.

Immutable: everything about the victim's profile (account age, average spend, usual
hour). Free: amount (down to what is worth stealing), time of day, merchant type,
and pacing (fewer attempts per hour, longer gaps). Costly: spoofing a known device,
routing through a VPN exit in the victim's country, using a drop address that matches
billing, buying an aged email account.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import MCC_NAMES


@dataclass
class ThreatModel:
    min_amount: float = 2000.0           # below this the theft is not worth the risk
    budget: float = 4.0                  # total effort the attacker will spend per card
    costs: dict = field(default_factory=lambda: {
        "new_device": 2.0,               # buy/spoof a device fingerprint
        "ip_country_mismatch": 0.5,      # residential VPN in the victim's country
        "ship_bill_match": 1.5,          # drop address near the billing address
        "email_age_days": 1.5,           # aged email account from a marketplace
        "pacing": 0.5,                   # waiting between attempts (slower cash-out)
    })
    mutable: tuple = ("amount", "hour", "mcc", "velocity_1h", "secs_since_last", "new_device",
                      "ip_country_mismatch", "ship_bill_match", "email_age_days")

    def cost(self, orig: pd.Series, adv: pd.Series) -> float:
        c = 0.0
        if adv["new_device"] < orig["new_device"]:
            c += self.costs["new_device"]
        if adv["ip_country_mismatch"] < orig["ip_country_mismatch"]:
            c += self.costs["ip_country_mismatch"]
        if adv["ship_bill_match"] > orig["ship_bill_match"]:
            c += self.costs["ship_bill_match"]
        if adv["email_age_days"] > orig["email_age_days"] + 1:
            c += self.costs["email_age_days"]
        if adv["velocity_1h"] < orig["velocity_1h"] or adv["secs_since_last"] > orig["secs_since_last"]:
            c += self.costs["pacing"]
        return c

    def candidates(self, orig: pd.Series, cur: pd.Series) -> list[tuple[str, object]]:
        """Single-field moves from the current point that stay inside the threat model."""
        out = []
        for a in np.geomspace(self.min_amount, max(cur["amount"], self.min_amount), 6)[:-1]:
            out.append(("amount", round(float(a), 2)))
        for h in (cur["usual_hour"] if "usual_hour" in cur else 14, 10, 13, 16, 19):
            out.append(("hour", float(h)))
        for m in MCC_NAMES:
            if m != cur["mcc"]:
                out.append(("mcc", m))
        if cur["velocity_1h"] > 0:
            out.append(("velocity_1h", 0))
        if cur["secs_since_last"] < 86400:
            out.append(("secs_since_last", float(max(cur["secs_since_last"] * 10, 36000))))
        for f, benign in (("new_device", False), ("ip_country_mismatch", False), ("ship_bill_match", True)):
            if cur[f] != benign:
                out.append((f, benign))
        if cur["email_age_days"] < 730:
            out.append(("email_age_days", 1500.0))
        return [(k, v) for k, v in out if self.cost(orig, _with(cur, k, v)) <= self.budget]

    def valid(self, orig: pd.Series, adv: pd.Series) -> bool:
        immutable = ("account_age_days", "avg_amount_30d", "usual_hour")
        return (all(adv[k] == orig[k] for k in immutable) and adv["amount"] >= self.min_amount - 1e-6
                and 0 <= adv["hour"] <= 23 and self.cost(orig, adv) <= self.budget + 1e-9)


def _with(row: pd.Series, k: str, v) -> pd.Series:
    r = row.copy()
    r[k] = v
    return r
