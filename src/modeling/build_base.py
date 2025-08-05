# src/modeling/build_base.py

from __future__ import annotations

from datetime import timezone
from typing import List

import numpy as np
import pandas as pd

from src.runtime.artifacts import load_artifact, save_artifact


def _zscore_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            mu = out[c].astype(float).mean()
            sd = out[c].astype(float).std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                # skip degenerate columns
                continue
            out[c + "_z"] = (out[c].astype(float) - mu) / sd
    return out


def build() -> None:
    # Load inputs
    geo_wx, _, s1 = load_artifact("nyc_boro_wx")
    acs,    _, s2 = load_artifact("nyc_geo_acs2")
    raw,    _, s3 = load_artifact("nyc_raw")
    if not all("LOADED" in s for s in (s1, s2, s3)):
        raise SystemExit("❌ Need nyc_boro_wx, nyc_geo_acs2, nyc_raw first.")

    # Merge outcomes + descriptor (by unique_key)
    base = geo_wx.merge(
        raw[["unique_key", "closed_date", "descriptor"]], on="unique_key", how="left"
    )

    # Merge ACS by tract_geoid (present from spatialize step)
    # Keep only tract-level columns from ACS (avoid reintroducing request columns)
    acs_cols = ["tract_geoid", "median_hh_income", "total_pop", "median_age",
                "median_gross_rent", "share_65_plus"]
    base = base.merge(acs[acs_cols].drop_duplicates("tract_geoid"),
                      on="tract_geoid", how="left")

    # Outcomes: event + time-to-close (hours)
    now = pd.Timestamp.now(tz=timezone.utc)
    base["event"] = base["closed_date"].notna().astype(int)
    base["ttc_hours"] = np.where(
        base["event"].eq(1),
        (base["closed_date"] - base["created_date"]).dt.total_seconds() / 3600.0,
        (now - base["created_date"]).dt.total_seconds() / 3600.0,
    )
    base = base[base["ttc_hours"] >= 0].copy()

    # Z-score continuous weather covariates
    wx_like = [c for c in base.columns if c.startswith(("prcp_", "snowpx_", "freeze_"))]
    # include raw temp/precip if present
    for maybe in ["temp", "prcp_mm", "snow_proxy_mm"]:
        if maybe in base.columns:
            wx_like.append(maybe)
    base = _zscore_cols(base, sorted(set(wx_like)))

    event_rate = float(base["event"].mean() * 100.0)

    save_artifact(
        "model_base",
        base,
        {
            "event_rate_pct": event_rate,
            "n_rows": int(len(base)),
            "n_features": int(len(base.columns)),
        },
    )
    print(
        f"💾 Saved model_base: {len(base):,} rows  |  event rate = {event_rate:.1f}%"
    )


if __name__ == "__main__":
    build()
