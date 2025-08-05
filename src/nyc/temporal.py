"""
NYC temporal: arrivals/closures per hour, backlog, Little's Law fit.

Outputs
- artifacts/nyc_temporal_series.csv
- artifacts/nyc_temporal_series.csv.meta.json
- artifacts/nyc_temporal_backlog.png
- artifacts/nyc_temporal_backlog.png.meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])


def _write_meta(target_path: Path, meta: dict) -> None:
    sidecar = Path(f"{target_path}.meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, default=str))


def _hourly_count(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    s = s.dt.floor("h")
    return (
        s.value_counts()
        .sort_index()
        .asfreq("h", fill_value=0)  # lower-case h to avoid deprecation warning
    )


def analyze() -> None:
    nyc, _, st = load_artifact("nyc_raw")
    if "LOADED" not in st:
        raise SystemExit("❌ Need nyc_raw first.")

    # Arrivals & closures per hour
    arrivals_h = _hourly_count(nyc["created_date"])
    closures_h = _hourly_count(nyc["closed_date"])

    # Build a full, continuous hourly index to avoid holes
    if len(arrivals_h) and len(closures_h):
        idx_start = min(arrivals_h.index.min(), closures_h.index.min())
        idx_end   = max(arrivals_h.index.max(), closures_h.index.max())
        full_idx = pd.date_range(idx_start, idx_end, freq="h", tz="UTC")
    else:
        full_idx = arrivals_h.index

    arrivals_h = arrivals_h.reindex(full_idx, fill_value=0)
    closures_h = closures_h.reindex(full_idx, fill_value=0)

    # Observed backlog (queue length)
    L_obs = arrivals_h.cumsum() - closures_h.cumsum()

    hourly = pd.DataFrame(
        {"arrivals": arrivals_h, "closures": closures_h, "backlog": L_obs}
    )
    hourly.index.name = "ts"

    # --- Little's Law: L ≈ λ * W ---
    # W (service time): get time-to-close for closed items, reduce to ONE value per hour, then reindex
    closed = nyc.dropna(subset=["closed_date"]).copy()
    closed["created_date"] = pd.to_datetime(closed["created_date"], utc=True, errors="coerce")
    closed["closed_date"]  = pd.to_datetime(closed["closed_date"],  utc=True, errors="coerce")
    closed = closed.dropna(subset=["created_date", "closed_date"])
    closed["ttc_hours"] = (closed["closed_date"] - closed["created_date"]).dt.total_seconds() / 3600.0
    closed["closed_hour"] = closed["closed_date"].dt.floor("h")

    # Aggregate to guarantee unique hourly index BEFORE upsampling
    # (median tends to be robust; mean is also fine)
    service_time_hourly = (
        closed.groupby("closed_hour")["ttc_hours"]
        .median()
        .sort_index()
        .reindex(full_idx)               # align to full hourly timeline
        .ffill()                         # carry forward last-known value
    )

    # Rolling λ (arrivals per hour) and rolling W (service time)
    metrics = {}
    for window_h, lab in [(24 * 7, "7d"), (24 * 14, "14d")]:
        lam = hourly["arrivals"].rolling(window_h, min_periods=1).mean()
        W   = service_time_hourly.rolling(window_h, min_periods=1).mean()
        hourly[f"L_pred_{lab}"] = lam * W

    # Evaluate scores
    out = hourly.dropna(subset=["backlog"]).copy()
    for lab in ["7d", "14d"]:
        y = out["backlog"].values
        yhat = out[f"L_pred_{lab}"].fillna(0).values
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) if len(y) > 1 else 0.0
        r2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = float(np.mean(np.abs(y - yhat)))
        metrics[f"little_law_r2_{lab}"]  = round(r2, 3)
        metrics[f"little_law_mae_{lab}"] = round(mae, 1)

    # Save CSV
    p_csv = ART / "nyc_temporal_series.csv"
    hourly.reset_index().to_csv(p_csv, index=False)
    _write_meta(
        p_csv,
        {
            "peak_backlog": int(out["backlog"].max()) if len(out) else None,
            "peak_backlog_ts": str(out["backlog"].idxmax()) if len(out) else None,
            **metrics,
        },
    )
    print(f"💾 Saved temporal series → {p_csv}")

    # Figure
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(out.index, out["backlog"], label="Backlog (L_obs)")
    for lab in ["7d", "14d"]:
        ax.plot(out.index, out[f"L_pred_{lab}"], label=f"L_pred_{lab}")
    ax.set_title("NYC Backlog vs. Little’s Law Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Requests in backlog")
    ax.legend()
    fig.tight_layout()
    p_png = ART / "nyc_temporal_backlog.png"
    fig.savefig(p_png, dpi=150)
    plt.close(fig)
    _write_meta(p_png, metrics)
    print(f"🖼️  Saved figure → {p_png}")

    # Log scores
    print(f"Little’s Law fit (7d):  R²={metrics['little_law_r2_7d']}  MAE={metrics['little_law_mae_7d']}")
    print(f"Little’s Law fit (14d): R²={metrics['little_law_r2_14d']}  MAE={metrics['little_law_mae_14d']}")


if __name__ == "__main__":
    analyze()
