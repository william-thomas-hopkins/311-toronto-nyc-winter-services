"""
NYC spatial clustering: per-borough HDBSCAN on haversine distance.

Fixes:
- Tightened min_cluster_size and min_samples
- Added median/max radius (m), compactness, umbrella filter
- Rank score penalizes large radii
- Extra meta sidecars

Outputs
- artifacts/nyc_spatial_clusters.parquet
- artifacts/nyc_spatial_cluster_summary.csv
- artifacts/nyc_spatial_cluster_summary.csv.meta.json
- artifacts/hotspot_candidates_ranked.csv
- artifacts/hotspot_candidates_ranked.csv.meta.json
- artifacts/hotspot_umbrella_clusters.csv  (large, non-actionable blobs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import hdbscan
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])

# Umbrella filter threshold (median radius)
UMBRELLA_RADIUS_M = 1000.0  # 1 km
R_EARTH_M = 6371000.0


def _write_meta(target_path: Path, meta: dict) -> None:
    sidecar = Path(f"{target_path}.meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, default=str))


def _to_rad(lat: pd.Series, lon: pd.Series) -> np.ndarray:
    return np.deg2rad(np.c_[lat.values, lon.values])  # shape (n, 2) -> [lat, lon] in rad


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    a = np.deg2rad(lat1)
    b = np.deg2rad(lon1)
    c = np.deg2rad(lat2)
    d = np.deg2rad(lon2)
    dlat = c - a
    dlon = d - b
    sin2 = np.sin(dlat / 2.0) ** 2 + np.cos(a) * np.cos(c) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_M * np.arcsin(np.sqrt(sin2))


def _cluster_one_borough(df_b: pd.DataFrame) -> pd.DataFrame:
    coords_rad = _to_rad(df_b["latitude"], df_b["longitude"])[:, ::-1]  # BallTree expects [lon, lat] but HDBSCAN uses the same order if metric='haversine'; keep [lat, lon] for HDBSCAN
    # hdbscan with haversine expects [lat, lon] in radians
    coords_hdb = _to_rad(df_b["latitude"], df_b["longitude"])  # [lat, lon] in rad

    n = len(df_b)
    min_cluster_size = max(50, int(0.003 * n))  # ~0.3% of borough rows; floor 50
    min_samples = 15

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="haversine",
        core_dist_n_jobs=1,
        allow_single_cluster=False,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(coords_hdb)

    out = df_b.copy()
    out["spatial_cluster"] = labels
    return out


def _summarize_clusters(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Keep only real clusters (>=0); noise = -1
    cl = df[df["spatial_cluster"] >= 0].copy()
    if cl.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    # Compute centroid and radius stats
    gb = cl.groupby(["borough", "spatial_cluster"], as_index=False)
    centroids = gb[["latitude", "longitude"]].median().rename(
        columns={"latitude": "centroid_lat", "longitude": "centroid_lon"}
    )
    merged = cl.merge(centroids, on=["borough", "spatial_cluster"], how="left")
    merged["dist_m"] = _haversine_m(
        merged["latitude"], merged["longitude"], merged["centroid_lat"], merged["centroid_lon"]
    )

    s = (
        merged.groupby(["borough", "spatial_cluster"])
        .agg(
            size=("unique_key", "size"),
            centroid_lat=("centroid_lat", "first"),
            centroid_lon=("centroid_lon", "first"),
            radius_median_m=("dist_m", "median"),
            radius_max_m=("dist_m", "max"),
        )
        .reset_index()
    )
    s["compactness"] = s["radius_median_m"] / s["radius_max_m"].clip(lower=1.0)

    # Rank: penalize radius (150 m scale)
    s["rank_score"] = s["size"] / (1.0 + s["radius_median_m"] / 150.0)
    s = s.sort_values(["rank_score", "size"], ascending=[False, False], ignore_index=True)

    # Filter umbrella (too-big) clusters
    umbrella = s[s["radius_median_m"] > UMBRELLA_RADIUS_M].copy()
    hotspots = s[s["radius_median_m"] <= UMBRELLA_RADIUS_M].copy()

    return merged, s, hotspots if not hotspots.empty else pd.DataFrame(), umbrella


def analyze() -> None:
    geo, _, st = load_artifact("nyc_geo")
    if "LOADED" not in st:
        raise SystemExit("❌ Need nyc_geo first.")

    geo = geo.dropna(subset=["latitude", "longitude", "borough"]).copy()

    parts = []
    for bor, df_b in geo.groupby("borough"):
        parts.append(_cluster_one_borough(df_b))
    labeled = pd.concat(parts, ignore_index=True) if parts else geo.copy()

    # Save labeled points (all, including noise)
    p_parq = ART / "nyc_spatial_clusters.parquet"
    labeled.to_parquet(p_parq, index=False)
    print(f"💾 Saved nyc_spatial_clusters: {len(labeled):,} rows → {p_parq}")

    # Summaries + rankings
    merged, summary, hotspots, umbrella = _summarize_clusters(labeled)

    p_summary = ART / "nyc_spatial_cluster_summary.csv"
    summary.to_csv(p_summary, index=False)
    _write_meta(
        p_summary,
        {
            "n_clusters_total": int((summary.shape[0])) if summary is not None else 0,
            "umbrella_radius_m": UMBRELLA_RADIUS_M,
            "median_radius_median_m": float(summary["radius_median_m"].median()) if len(summary) else None,
            "p95_radius_median_m": float(summary["radius_median_m"].quantile(0.95)) if len(summary) else None,
        },
    )
    print(f"💾 Saved cluster summary → {p_summary}")

    p_hot = ART / "hotspot_candidates_ranked.csv"
    if hotspots is not None and len(hotspots):
        hotspots.to_csv(p_hot, index=False)
        _write_meta(
            p_hot,
            {
                "n_hotspots": int(len(hotspots)),
                "top_hotspot_radius_median_m": float(hotspots.iloc[0]["radius_median_m"]),
                "top_hotspot_size": int(hotspots.iloc[0]["size"]),
            },
        )
        print(f"💾 Saved hotspot ranks → {p_hot}")
    else:
        print("⚠️  No hotspots after umbrella filter; check parameters.")

    # Save umbrella clusters too (for context)
    p_umb = ART / "nyc_spatial_umbrella_clusters.csv"
    if umbrella is not None and len(umbrella):
        umbrella.to_csv(p_umb, index=False)
        _write_meta(p_umb, {"n_umbrella": int(len(umbrella)), "radius_threshold_m": UMBRELLA_RADIUS_M})
        print(f"💾 Saved umbrella clusters → {p_umb}")


if __name__ == "__main__":
    analyze()
