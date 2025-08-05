"""
Create modelling base table:
nyc_boro_wx  ⟂  outcomes + descriptor  ⟂  ACS metrics
→ artifacts/model_base.parquet
"""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from src.runtime.artifacts import load_artifact, save_artifact


def build() -> None:
    geo_wx, _, s1 = load_artifact("nyc_boro_wx")
    acs,    _, s2 = load_artifact("nyc_geo_acs2")
    raw,    _, s3 = load_artifact("nyc_raw")

    if not all("LOADED" in s for s in (s1, s2, s3)):
        raise SystemExit("❌  Need nyc_boro_wx, nyc_geo_acs2, nyc_raw first.")

    base = (
        geo_wx.merge(
            raw[["unique_key", "closed_date", "descriptor"]],
            on="unique_key",
            how="left",
        )
        .merge(
            acs.drop(columns=["created_date", "latitude", "longitude", "borough"], errors="ignore"),
            on="unique_key",
            how="left",
        )
    )

    now = pd.Timestamp.now(tz=timezone.utc)
    base["event"] = base["closed_date"].notna().astype(int)
    base["ttc_hours"] = np.where(
        base["event"].eq(1),
        (base["closed_date"] - base["created_date"]).dt.total_seconds() / 3600.0,
        (now - base["created_date"]).dt.total_seconds() / 3600.0,
    )
    base = base[base["ttc_hours"] >= 0].copy()

    save_artifact(
        "model_base",
        base,
        {"event_rate_pct": float(base["event"].mean() * 100)},
    )

    print(f"✓  model_base saved  |  rows={len(base):,}   event rate={base['event'].mean():.2%}")


if __name__ == "__main__":
    build()
