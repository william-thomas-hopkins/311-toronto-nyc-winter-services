"""
Spatially attach NYC service requests to census tracts & borough
→ artifacts/nyc_geo.parquet
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.runtime.artifacts import load_artifact, save_artifact
from src.runtime.config import get_config

CFG = get_config()

NYC_COUNTIES: Dict[str, str] = {
    "005": "BRONX",
    "047": "BROOKLYN",
    "061": "MANHATTAN",
    "081": "QUEENS",
    "085": "STATEN ISLAND",
}


def _ensure_tiger_zip() -> Path:
    tiger_dir = Path(CFG["paths"]["CACHE_DIR"]) / "tiger"
    tiger_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tiger_dir / "tl_2022_36_tract.zip"
    if not zip_path.exists():
        print("⏬  Downloading TIGER tracts (NY)...")
        urllib.request.urlretrieve(
            "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_36_tract.zip",
            zip_path,
        )
    return zip_path


def build() -> None:
    nyc, _, st = load_artifact("nyc_raw")
    if "LOADED" not in st:
        raise SystemExit("❌  Need nyc_raw first.")

    #  Create GeoDataFrame of SR points
    df = nyc[["unique_key", "created_date", "latitude", "longitude"]].copy()
    df = df.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    #  Load TIGER tracts → filter to NYC counties
    zip_path = _ensure_tiger_zip()
    tracts = gpd.read_file(f"zip://{zip_path}")
    tracts = tracts[tracts["COUNTYFP"].isin(NYC_COUNTIES)].to_crs("EPSG:4326")
    tracts["borough"] = tracts["COUNTYFP"].map(NYC_COUNTIES)
    tracts["tract_geoid"] = tracts["GEOID"]

    #  Spatial join
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
    save_artifact("nyc_geo", joined[keep_cols])

    attach_rate = joined["tract_geoid"].notna().mean() * 100
    print(f"✓  nyc_geo saved  |  tract attach rate = {attach_rate:.1f} %")


if __name__ == "__main__":
    build()
