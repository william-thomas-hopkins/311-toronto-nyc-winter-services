"""
Fetch ACS-5 2022 tract-level variables and merge with nyc_geo
→ artifacts/nyc_geo_acs2.parquet
"""

from __future__ import annotations

from typing import List

import censusdata as cd
import pandas as pd

from src.runtime.artifacts import load_artifact, save_artifact

ACS_VARS = {
    "median_hh_income": "B19013_001E",
    "total_pop":        "B01003_001E",
    "median_age":       "B01002_001E",
    "median_gross_rent":"B25064_001E",
}

AGE_65_VARS: List[str] = (
    [f"B01001_{i:03d}E" for i in range(20, 26)] +
    [f"B01001_{i:03d}E" for i in range(44, 50)]
)


def _fetch_tracts(state: str, county: str) -> pd.DataFrame:
    geo_q = cd.censusgeo([("state", state), ("county", county), ("tract", "*")])
    cols  = list(ACS_VARS.values()) + AGE_65_VARS
    df    = cd.download("acs5", 2022, geo_q, cols)
    df = df.reset_index(drop=True)

    # build GEOID (= state + county + tract codes)
    df["tract_geoid"] = ["".join(map(str, g.params().values())) for g in geo_q]

    df = df.rename(columns={v: k for k, v in ACS_VARS.items()})
    df["age_65_plus"]   = df[AGE_65_VARS].sum(axis=1)
    df["share_65_plus"] = df["age_65_plus"] / df["total_pop"].replace(0, pd.NA)

    keep = ["tract_geoid"] + list(ACS_VARS.keys()) + ["share_65_plus"]
    return df[keep]


def build() -> None:
    geo, _, st = load_artifact("nyc_geo")
    if "LOADED" not in st:
        raise SystemExit("❌  Run spatial.py first.")
    # NYC FIPS
    counties = ["005", "047", "061", "081", "085"]  
    parts = [_fetch_tracts("36", c) for c in counties]
    acs = pd.concat(parts, ignore_index=True)

    merged = geo.merge(acs, on="tract_geoid", how="left")
    save_artifact("nyc_geo_acs2", merged)

    cov = merged["median_hh_income"].notna().mean() * 100
    print(f"✓  nyc_geo_acs2 saved  |  income coverage = {cov:.1f} %")


if __name__ == "__main__":
    build()
