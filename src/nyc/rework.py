# src/nyc/rework.py
"""
NYC rework detection via spatiotemporal proximity.

- Canonical: next request within D=50 m and W=48 h
- Placebo: shift the time window back by 168 h (−7 days)
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
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])
R_EARTH_M = 6_371_000.0

# Canonical default (matches notebook)
CANON_DIST_M = 50.0
CANON_WIN_H = 48.0

DIST_GRID = [25, 50, 75, 100, 150]
WIN_GRID = [24, 36, 48, 72, 96]


def _write_meta(target_path: Path, meta: dict) -> None:
    Path(f"{target_path}.meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _to_radians_latlon(df: pd.DataFrame) -> np.ndarray:
    """Return [[lat_rad, lon_rad], ...] as required by BallTree(metric='haversine')."""
    lat = np.deg2rad(df["latitude"].to_numpy())
    lon = np.deg2rad(df["longitude"].to_numpy())
    return np.column_stack([lat, lon])


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine in meters."""
    a = np.deg2rad(lat1)
    b = np.deg2rad(lon1)
    c = np.deg2rad(lat2)
    d = np.deg2rad(lon2)
    dlat = c - a
    dlon = d - b
    sin2 = np.sin(dlat / 2.0) ** 2 + np.cos(a) * np.cos(c) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_M * np.arcsin(np.sqrt(sin2))


def _prep_geo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["latitude", "longitude", "created_date"]).copy()
    out["created_date"] = pd.to_datetime(out["created_date"], utc=True, errors="coerce")
    out = out.dropna(subset=["created_date"])
    out = out.sort_values("created_date").reset_index(drop=True)
    # Useful for debugging/joins
    if "unique_key" not in out.columns:
        out["unique_key"] = np.arange(len(out))
    return out


def _precompute_neighbors(tree: BallTree, coords_rad: np.ndarray, max_dist_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query neighbors for each point within max_dist_m (in meters), return (indices, distances_rad).
    distances are in radians (BallTree convention for haversine).
    """
    r_max = max_dist_m / R_EARTH_M  # meters -> radians
    inds, dists = tree.query_radius(coords_rad, r=r_max, return_distance=True, sort_results=True)
    return inds, dists  # arrays of arrays


def _pairs_within(df: pd.DataFrame,
                  pre_inds: np.ndarray,
                  pre_dists_rad: np.ndarray,
                  dist_m: float,
                  window_h: float,
                  placebo_shift_h: float = 0.0) -> pd.DataFrame:
    """
    Build pairs from precomputed spatial neighbors, filtering by tighter dist/time constraints.
    """
    # tighter spatial threshold (meters -> radians)
    r_thr = dist_m / R_EARTH_M
    dt = pd.to_timedelta(window_h, unit="h")
    shift = pd.to_timedelta(placebo_shift_h, unit="h")

    created = df["created_date"].to_numpy()  # tz-aware datetime64[ns, UTC]

    src_idx = []
    dst_idx = []

    for i in range(len(df)):
        t0 = created[i]
        t1 = t0 + dt

        # candidate neighbors from precomputed sets (includes self and past/future)
        neigh_idx = pre_inds[i]
        neigh_dist = pre_dists_rad[i]

        if len(neigh_idx) == 0:
            continue

        # spatial cut
        keep_spatial = neigh_dist <= r_thr
        if not np.any(keep_spatial):
            continue

        cand = neigh_idx[keep_spatial]

        # exclude self
        cand = cand[cand != i]
        if cand.size == 0:
            continue

        # temporal cut (future within window; with placebo shift if provided)
        # created is a numpy array of datetimes; comparisons are vectorized
        c_times = created[cand]
        mask_time = (c_times > (t0 + shift)) & (c_times <= (t1 + shift))
        cand = cand[mask_time]

        if cand.size:
            src_idx.extend([i] * cand.size)
            dst_idx.extend(cand.tolist())

    if not src_idx:
        return pd.DataFrame(columns=[
            "src_idx", "dst_idx", "distance_m", "dt_hours", "src_unique_key", "dst_unique_key"
        ])

    pairs = pd.DataFrame({"src_idx": src_idx, "dst_idx": dst_idx}).drop_duplicates()

    # distances
    pairs["distance_m"] = _haversine_m(
        df.loc[pairs["src_idx"], "latitude"].to_numpy(),
        df.loc[pairs["src_idx"], "longitude"].to_numpy(),
        df.loc[pairs["dst_idx"], "latitude"].to_numpy(),
        df.loc[pairs["dst_idx"], "longitude"].to_numpy(),
    )

    # time deltas (hours)
    dt_vals = (
        df.loc[pairs["dst_idx"], "created_date"].to_numpy()
        - df.loc[pairs["src_idx"], "created_date"].to_numpy()
    )
    pairs["dt_hours"] = dt_vals.astype("timedelta64[h]").astype(float)

    pairs["src_unique_key"] = df.loc[pairs["src_idx"], "unique_key"].to_numpy()
    pairs["dst_unique_key"] = df.loc[pairs["dst_idx"], "unique_key"].to_numpy()

    # enforce the thresholds again (safety)
    pairs = pairs[
        (pairs["distance_m"] <= dist_m + 1e-6) &
        (pairs["dt_hours"] > 0) &
        (pairs["dt_hours"] <= window_h + 1e-9)
    ].reset_index(drop=True)

    return pairs


def analyze() -> None:
    geo, _, st = load_artifact("nyc_geo")
    if "LOADED" not in st:
        raise SystemExit("❌ Need nyc_geo first.")

    df = _prep_geo(geo)
    if df.empty:
        print("⚠️  No rows after prep.")
        return

    # Build tree once at the max search radius we will ever need
    coords_rad = _to_radians_latlon(df)
    tree = BallTree(coords_rad, metric="haversine")
    pre_inds, pre_dists = _precompute_neighbors(tree, coords_rad, max_dist_m=max(DIST_GRID))

    # Quick sanity: % of rows with at least one spatial neighbor within 150 m
    has_neighbor_150m = np.array([np.any(d <= DIST_GRID[-1] / R_EARTH_M) and len(i) > 1 for i, d in zip(pre_inds, pre_dists)])
    print(f"🔎 Spatial sanity: {has_neighbor_150m.mean()*100:.1f}% of points have a neighbor within {DIST_GRID[-1]} m")

    # Canonical & placebo
    canonical = _pairs_within(df, pre_inds, pre_dists, dist_m=CANON_DIST_M, window_h=CANON_WIN_H, placebo_shift_h=0.0)
    placebo   = _pairs_within(df, pre_inds, pre_dists, dist_m=CANON_DIST_M, window_h=CANON_WIN_H, placebo_shift_h=-168.0)

    p_can = ART / "nyc_rework_pairs_canonical.csv"
    canonical.to_csv(p_can, index=False)
    _write_meta(p_can, {"n_pairs_canonical": int(len(canonical)), "dist_m": CANON_DIST_M, "window_h": CANON_WIN_H})
    print(f"💾 Saved canonical rework pairs → {p_can}  (n={len(canonical):,})")

    p_pl = ART / "nyc_rework_pairs_placebo.csv"
    placebo.to_csv(p_pl, index=False)
    _write_meta(p_pl, {"n_pairs_placebo": int(len(placebo)), "dist_m": CANON_DIST_M, "window_h": CANON_WIN_H, "shift_h": -168})
    print(f"💾 Saved placebo rework pairs → {p_pl}  (n={len(placebo):,})")

    # Sensitivity grid
    rows = []
    for d in DIST_GRID:
        for w in WIN_GRID:
            n = len(_pairs_within(df, pre_inds, pre_dists, dist_m=float(d), window_h=float(w)))
            rows.append({"distance_m": d, "window_h": w, "n_pairs": n})
    sens = pd.DataFrame(rows)
    p_sens = ART / "nyc_rework_sensitivity.csv"
    sens.to_csv(p_sens, index=False)
    best = sens.sort_values("n_pairs", ascending=False).head(1).to_dict("records")[0]
    _write_meta(p_sens, {"best_cell": best})
    print(f"💾 Saved sensitivity grid → {p_sens}")

    # Heatmap
    pivot = sens.pivot(index="distance_m", columns="window_h", values="n_pairs").sort_index().sort_index(axis=1)
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
    _write_meta(p_png, {"min_pairs": int(pivot.values.min()), "max_pairs": int(pivot.values.max())})
    print(f"🖼️  Saved heatmap → {p_png}")

    # Extra sanity logging
    print(f"🔎 Sanity: canonical={len(canonical)}, placebo={len(placebo)}, grid best={best}")
    if len(canonical) == 0:
        print("⚠️  No canonical pairs found. Check UTC timestamps, spatial density, and radius/time thresholds.")


if __name__ == "__main__":
    analyze()
