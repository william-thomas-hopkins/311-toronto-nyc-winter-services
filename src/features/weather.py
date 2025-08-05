"""
Fetch Meteostat hourly observations, aggregate to borough level
(and add rolling windows / proxies) → artifacts/wx_boro.parquet
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from meteostat import Hourly, Stations

from src.runtime.artifacts import load_artifact, save_artifact
from src.runtime.config import get_config

CFG = get_config()

#  Hard-coded centroids for the five NYC boroughs
BORO_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "BRONX":         (40.8448, -73.8648),
    "BROOKLYN":      (40.6500, -73.9496),
    "MANHATTAN":     (40.7831, -73.9712),
    "QUEENS":        (40.7282, -73.7949),
    "STATEN ISLAND": (40.5795, -74.1502),
}


#  Helpers
def _nearest_station(lat: float, lon: float) -> str | None:
    """Return the Meteostat station id closest to (lat, lon) with data."""
    cand = Stations().nearby(lat, lon).fetch(10)
    return str(cand.index[0]) if len(cand) else None



#  Builder
def build() -> None:
    nyc, _, st = load_artifact("nyc_raw")
    if "LOADED" not in st:
        raise SystemExit("❌  Run ingest first to build nyc_raw.")

    # pad ±3 days so rolling windows at edges have values
    t0 = nyc["created_date"].min().floor("h") - timedelta(days=3)
    t1 = nyc["created_date"].max().ceil("h") + timedelta(days=3)
    # Meteostat expects naive UTC
    start = t0.tz_convert("UTC").tz_localize(None)  
    end   = t1.tz_convert("UTC").tz_localize(None)

    parts, meta_stations = [], {}

    for boro, (lat, lon) in BORO_CENTROIDS.items():
        sid = _nearest_station(lat, lon)
        if sid is None:
            print(f"⚠️  No station for {boro} – skipping")
            continue

        df = Hourly(sid, start, end).fetch()
        if df.empty:
            print(f"⚠️  Station {sid} returned 0 rows for {boro}")
            continue

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.reset_index().rename(columns={"time": "created_hour_utc"})
        df["borough"] = boro

        
        df["prcp_mm"] = df["prcp"].fillna(0.0)
        df["snow_proxy_mm"] = np.where(
            df["snow"].notna(),
            df["snow"],
            np.where(df["temp"] <= 0, df["prcp_mm"], 0.0),
        )
        df["freeze_flag"] = (df["temp"] <= 0).astype(int)

        
        for w in (3, 6, 12, 24):
            df[f"prcp_{w}h"]       = df["prcp_mm"].rolling(w, min_periods=1).sum()
            df[f"snowpx_{w}h"]     = df["snow_proxy_mm"].rolling(w, min_periods=1).sum()
            df[f"freeze_{w}h_min"] = df["temp"].rolling(w, min_periods=1).min()

        parts.append(df)
        meta_stations[boro] = sid
        print(f"✓  {boro:<13} → station {sid}  rows={len(df):,}")

    if not parts:
        raise SystemExit("❌  No boroughs fetched – aborting.")

    wx = pd.concat(parts, ignore_index=True).sort_values(
        ["borough", "created_hour_utc"]
    )

    save_artifact("wx_boro", wx, {"stations": meta_stations})


if __name__ == "__main__":
    build()
