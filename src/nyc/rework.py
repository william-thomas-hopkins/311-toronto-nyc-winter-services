"""
NYC rework detection via spatiotemporal proximity.

- Canonical: next request within D=50m and W=48h
- Placebo: shift window back by 168h
- Sensitivity grid across D∈{25,50,75,100,150}, W∈{24,36,48,72,96}

Outputs
- artifacts/nyc_rework_pairs_canonical.csv
- artifacts/nyc_rework_pairs_canonical.csv.meta.json
- artifacts/nyc_rework_pairs_placebo.csv
- artifacts/nyc_rework_pairs_placebo.csv.meta.json
- artifacts/nyc_rework_sensitivity.csv
- artifacts/nyc_rework_sensitivity.csv.meta.json
- artifacts/nyc_rework_sensitivity_heatmap.png
- artifacts/nyc_rework_sensitivity_heatmap.png.meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])
R_EARTH_M = 6371000.0


def _write_meta(target_path: Path, meta: dict) -> None:
    sidecar = Path(f"{target_path}.meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, default=str))


def _balltree(df: pd.DataFrame) -> BallTree:
    coords = np.deg2rad(np.c_[df["latitude"].values, df["longitude"].values])
    return BallTree(coords, metric="haversine")


def _pairs_within(df: pd.DataFrame, dist_m: float, window_h: float, placebo_shift_h: float = 0.0) -> pd.DataFrame:
    df = df.sort_values("created_date").reset_index(drop=True).copy()
    df["created_date"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")
    tree = _balltree(df)

    rad_threshold = dist_m / R_EARTH_M
    dt = pd.to_timedelta(window_h, unit="h")
    shift = pd.to_timedelta(placebo_shift_h, unit="h")

    idx_src = []
    idx_dst = []

    for i, row in df.iterrows():
        t0 = row["created_date"]
        t1 = t0 + dt
        mask_time = (df["created_date"] > (t0 + shift)) & (df["created_date"] <= (t1 + shift))
        if not mask_time.any():
            continue

        cand_idx = np.flatnonzero(mask_time.values)
        if len(cand_idx) == 0:
            continue

        # spatial filter using BallTree query radius
        dists, inds = tree.query_radius(
            np.deg2rad([[row["latitude"], row["longitude"]]]), r=rad_threshold, return_distance=True
        )
        inds = set(inds[0].tolist())
        cand = [j for j in cand_idx if j in inds and j != i]

        if cand:
            idx_src.extend([i] * len(cand))
            idx_dst.extend(cand)

    if not idx_src:
        return pd.DataFrame(columns=["src_idx", "dst_idx"])

    pairs = pd.DataFrame({"src_idx": idx_src, "dst_idx": idx_dst})
    pairs = pairs.drop_duplicates()
    pairs["distance_m"] = _haversine_m(
        df.loc[pairs["src_idx"], "latitude"].values,
        df.loc[pairs["src_idx"], "longitude"].values,
        df.loc[pairs["dst_idx"], "latitude"].values,
        df.loc[pairs["dst_idx"], "longitude"].values,
    )
    pairs["dt_hours"] = (
        (df.loc[pairs["dst_idx"], "created_date"].values - df.loc[pairs["src_idx"], "created_date"].values)
        / np.timedelta64(1, "h")
    )
    pairs["src_unique_key"] = df.loc[pairs["src_idx"], "unique_key"].values
    pairs["dst_unique_key"] = df.loc[pairs["dst_idx"], "unique_key"].values
    return pairs


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    a = np.deg2rad(lat1)
    b = np.deg2rad(lon1)
    c = np.deg2rad(lat2)
    d = np.deg2rad(lon2)
    dlat = c - a
    dlon = d - b
    sin2 = np.sin(dlat / 2.0) ** 2 + np.cos(a) * np.cos(c) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_M * np.arcsin(np.sqrt(sin2))


def analyze() -> None:
    geo, _, st = load_artifact("nyc_geo")
    if "LOADED" not in st:
        raise SystemExit("❌ Need nyc_geo first.")

    geo = geo.dropna(subset=["latitude", "longitude", "created_date"]).copy()
    geo["created_date"] = pd.to_datetime(geo["created_date"], utc=True, errors="coerce")
    geo = geo.sort_values("created_date").reset_index(drop=True)

    # Canonical and placebo
    canonical = _pairs_within(geo, dist_m=50.0, window_h=48.0)
    placebo = _pairs_within(geo, dist_m=50.0, window_h=48.0, placebo_shift_h=-168.0)

    p_can = ART / "nyc_rework_pairs_canonical.csv"
    canonical.to_csv(p_can, index=False)
    _write_meta(p_can, {"n_pairs_canonical": int(len(canonical))})
    print(f"💾 Saved canonical rework pairs → {p_can}  (n={len(canonical):,})")

    p_pl = ART / "nyc_rework_pairs_placebo.csv"
    placebo.to_csv(p_pl, index=False)
    _write_meta(p_pl, {"n_pairs_placebo": int(len(placebo))})
    print(f"💾 Saved placebo rework pairs → {p_pl}  (n={len(placebo):,})")

    # Sensitivity grid
    distances = [25, 50, 75, 100, 150]
    windows = [24, 36, 48, 72, 96]
    rows = []
    for d in distances:
        for w in windows:
            n = len(_pairs_within(geo, dist_m=float(d), window_h=float(w)))
            rows.append({"distance_m": d, "window_h": w, "n_pairs": n})
    sens = pd.DataFrame(rows)

    p_sens = ART / "nyc_rework_sensitivity.csv"
    sens.to_csv(p_sens, index=False)
    _write_meta(
        p_sens,
        {
            "best_cell": sens.sort_values("n_pairs", ascending=False).head(1).to_dict(orient="records")[0]
            if len(sens)
            else None
        },
    )
    print(f"💾 Saved sensitivity grid → {p_sens}")

    # Heatmap
    pivot = sens.pivot(index="distance_m", columns="window_h", values="n_pairs")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    ax.set_xlabel("Window (hours)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Rework sensitivity (pairs count)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("# pairs")
    fig.tight_layout()
    p_png = ART / "nyc_rework_sensitivity_heatmap.png"
    fig.savefig(p_png, dpi=150)
    plt.close(fig)
    _write_meta(
        p_png,
        {
            "min_pairs": int(pivot.values.min()) if pivot.size else None,
            "max_pairs": int(pivot.values.max()) if pivot.size else None,
        },
    )
    print(f"🖼️  Saved heatmap → {p_png}")


if __name__ == "__main__":
    analyze()
