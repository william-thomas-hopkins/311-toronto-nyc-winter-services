"""
Join NYC SRs (spatialized) with borough-hour Meteostat features
→ artifacts/nyc_boro_wx.parquet
"""

from __future__ import annotations

import pandas as pd

from src.runtime.artifacts import load_artifact, save_artifact


def build() -> None:
    # Load dependencies
    geo, _, s_geo = load_artifact("nyc_geo")      # from spatial.py
    wx,  _, s_wx  = load_artifact("wx_boro")      # from weather.py
    if "LOADED" not in s_geo or "LOADED" not in s_wx:
        raise SystemExit("❌ Need nyc_geo and wx_boro first.")

    # Ensure join keys are aligned
    if "created_hour_utc" not in geo.columns:
        raise SystemExit("nyc_geo missing created_hour_utc.")
    if "borough" not in geo.columns:
        raise SystemExit("nyc_geo missing borough.")
    if "created_hour_utc" not in wx.columns or "borough" not in wx.columns:
        raise SystemExit("wx_boro missing required keys.")

    # Normalize to UTC hourly
    geo["created_hour_utc"] = pd.to_datetime(geo["created_hour_utc"], utc=True).dt.floor("h")
    wx["created_hour_utc"]  = pd.to_datetime(wx["created_hour_utc"],  utc=True).dt.floor("h")

    # Keep the weather features we care about (wx_boro already is hourly per borough)
    wx_cols_keep = [
        "created_hour_utc", "borough", "temp", "prcp_mm", "snow_proxy_mm", "freeze_flag",
        "prcp_3h", "prcp_6h", "prcp_12h", "prcp_24h",
        "snowpx_3h", "snowpx_6h", "snowpx_12h", "snowpx_24h",
        "freeze_3h_min", "freeze_6h_min", "freeze_12h_min", "freeze_24h_min",
    ]
    wx = wx[[c for c in wx_cols_keep if c in wx.columns]].drop_duplicates(["borough", "created_hour_utc"])

    # Join on (borough, created_hour_utc)
    base = geo.merge(wx, on=["borough", "created_hour_utc"], how="left")

    # Quick attach rate
    attach = base["temp"].notna().mean() * 100.0
    print(f"✓  Weather attach rate: {attach:.1f}%  (rows={len(base):,})")

    save_artifact("nyc_boro_wx", base)


if __name__ == "__main__":
    build()
