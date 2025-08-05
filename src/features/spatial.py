"""
Spatially attach NYC service requests to census tracts & borough
→ artifacts/nyc_geo.parquet
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.runtime.artifacts import load_artifact, save_artifact
from src.runtime.config import get_config

# TLS helpers
def _safe_download(url: str, dest: Path) -> None:
    """
    Download a file with robust TLS verification.
    Tries: requests+certifi → curl --fail --location → urllib with certifi context.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1) requests + certifi (preferred)
    try:
        import requests, certifi  # type: ignore
        with requests.get(url, stream=True, timeout=90, verify=certifi.where()) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return
    except Exception as e:
        print(f"⚠️  requests download failed: {e!r}")

    # 2) curl fallback (common on macOS)
    try:
        rc = subprocess.run(
            ["curl", "-fsSL", url, "-o", str(dest)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if dest.exists() and dest.stat().st_size > 0:
            return
    except Exception as e:
        print(f"⚠️  curl fallback failed: {e!r}")

    # 3) urllib + certifi SSL context
    try:
        import ssl, certifi  # type: ignore
        import urllib.request

        ctx = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        with opener.open(url, timeout=90) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        return
    except Exception as e:
        print(f"❌  urllib fallback failed: {e!r}")
        raise SystemExit(
            "Failed to download TIGER zip after multiple methods. "
            "Try installing `requests certifi` or run this once manually:\n"
            f"  curl -fsSL {url} -o '{dest}'"
        )


CFG = get_config()

NYC_COUNTIES: Dict[str, str] = {
    "005": "BRONX",
    "047": "BROOKLYN",
    "061": "MANHATTAN",
    "081": "QUEENS",
    "085": "STATEN ISLAND",
}

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_36_tract.zip"


def _ensure_tiger_zip() -> Path:
    tiger_dir = Path(CFG["paths"]["CACHE_DIR"]) / "tiger"
    tiger_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tiger_dir / "tl_2022_36_tract.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("⏬  Downloading TIGER tracts (NY)…")
        _safe_download(TIGER_URL, zip_path)
    return zip_path


def build() -> None:
    nyc, _, st = load_artifact("nyc_raw")
    if "LOADED" not in st:
        raise SystemExit("❌  Need nyc_raw first. Run: python -m cli.run build-nyc")

    # Create GeoDataFrame of SR points
    df = nyc[["unique_key", "created_date", "latitude", "longitude"]].copy()
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    # Load TIGER tracts → filter to NYC counties
    zip_path = _ensure_tiger_zip()
    tracts = gpd.read_file(f"zip://{zip_path}")
    tracts = tracts[tracts["COUNTYFP"].isin(NYC_COUNTIES)].to_crs("EPSG:4326")
    tracts["borough"] = tracts["COUNTYFP"].map(NYC_COUNTIES)
    tracts["tract_geoid"] = tracts["GEOID"]

    # Spatial join
    joined = gpd.sjoin(
        gdf,
        tracts[["tract_geoid", "borough", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns="index_right")

    joined["created_hour_utc"] = pd.to_datetime(
        joined["created_date"], utc=True
    ).dt.floor("h")

    keep_cols = [
        "unique_key",
        "created_date",
        "created_hour_utc",
        "latitude",
        "longitude",
        "tract_geoid",
        "borough",
    ]
    out = joined[keep_cols].copy()
    save_artifact("nyc_geo", out)

    attach_rate = out["tract_geoid"].notna().mean() * 100
    print(f"✓  nyc_geo saved  |  tract attach rate = {attach_rate:.1f}%  | rows={len(out):,}")


if __name__ == "__main__":
    build()
