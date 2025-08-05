"""
NYC Survival analysis with proxy closures.

Outputs
- artifacts/nyc_with_proxy.parquet
- artifacts/nyc_km_summary_by_descriptor.csv
- artifacts/nyc_km_summary_by_descriptor.csv.meta.json
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])


def _write_meta(target_path: Path, meta: dict) -> None:
    sidecar = Path(f"{target_path}.meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, default=str))


def _with_proxy(nyc: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetimes are tz-aware UTC
    for col in ("created_date", "closed_date", "resolution_action_updated_date"):
        if col in nyc.columns:
            nyc[col] = pd.to_datetime(nyc[col], utc=True, errors="coerce")

    # Proxy close: resolution_action_updated_date if it exists and is >= created_date
    cond_proxy_ok = (
        nyc["resolution_action_updated_date"].notna()
        & nyc["created_date"].notna()
        & (nyc["resolution_action_updated_date"] >= nyc["created_date"])
    )
    closed_or_proxy = nyc["closed_date"].where(
        nyc["closed_date"].notna(), nyc["resolution_action_updated_date"].where(cond_proxy_ok)
    )

    out = nyc.copy()
    out["closed_or_proxy"] = closed_or_proxy

    # Build event + duration for survival (time-to-close, hours)
    now = pd.Timestamp.now(tz=timezone.utc)
    event = out["closed_or_proxy"].notna().astype(int)
    ttc_hours = np.where(
        event == 1,
        (out["closed_or_proxy"] - out["created_date"]).dt.total_seconds() / 3600.0,
        (now - out["created_date"]).dt.total_seconds() / 3600.0,
    )
    out["event_proxy"] = event
    out["ttc_hours_proxy"] = ttc_hours
    out = out.loc[out["ttc_hours_proxy"] >= 0].copy()

    return out


def _km_summary(df: pd.DataFrame, desc_col: str = "descriptor", top_n: int = 20) -> pd.DataFrame:
    # Pick top descriptors by count (to keep the run fast/stable)
    top_desc = (
        df[desc_col].fillna("UNKNOWN").value_counts().head(top_n).index.to_list()
    )
    df = df.copy()
    df[desc_col] = df[desc_col].fillna("UNKNOWN")
    df = df[df[desc_col].isin(top_desc)]

    rows = []
    for desc, g in df.groupby(desc_col, sort=False):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=g["ttc_hours_proxy"], event_observed=g["event_proxy"])
        N = int(len(g))
        median_hrs = float(kmf.median_survival_time_) if kmf.median_survival_time_ is not None else np.nan
        pct_cens = 100.0 * (1.0 - float(g["event_proxy"].mean()))  # percent (0..100)
        rows.append(
            {
                "descriptor": desc,
                "N": N,
                "median_hours": median_hrs,
                "pct_censored_percent": round(pct_cens, 2),
            }
        )
    out = pd.DataFrame(rows).sort_values(["N", "median_hours"], ascending=[False, True], ignore_index=True)
    return out


def analyze() -> None:
    nyc, _, st = load_artifact("nyc_raw")
    if "LOADED" not in st:
        raise SystemExit("❌ Need nyc_raw first.")

    with_proxy = _with_proxy(nyc)

    # Save the proxy-augmented dataset
    p_parq = ART / "nyc_with_proxy.parquet"
    with_proxy.to_parquet(p_parq, index=False)
    print(f"💾 Saved nyc_with_proxy: {len(with_proxy):,} rows → {p_parq}")

    # KM summary by descriptor
    km = _km_summary(with_proxy)
    p_csv = ART / "nyc_km_summary_by_descriptor.csv"
    km.to_csv(p_csv, index=False)
    _write_meta(
        p_csv,
        {
            "n_total": int(len(with_proxy)),
            "censored_percent_overall": round(100.0 * (1.0 - float(with_proxy["event_proxy"].mean())), 2),
            "top_descriptor": km.iloc[0]["descriptor"] if len(km) else None,
            "top_descriptor_N": int(km.iloc[0]["N"]) if len(km) else None,
        },
    )
    print(f"💾 Saved KM summary → {p_csv}")


if __name__ == "__main__":
    analyze()
