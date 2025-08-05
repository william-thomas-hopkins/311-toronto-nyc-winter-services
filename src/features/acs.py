"""
Fetch ACS tract-level variables for NYC counties and join onto nyc_geo
→ artifacts/nyc_geo_acs2.parquet
"""

from __future__ import annotations

import censusdata as cd
import pandas as pd

from src.runtime.artifacts import load_artifact, save_artifact

# Friendly name -> ACS table code
ACS_VARS = {
    "median_hh_income": "B19013_001E",
    "total_pop":        "B01003_001E",
    "median_age":       "B01002_001E",
    "median_gross_rent":"B25064_001E",
}

# Age 65+ cells (male + female)
AGE_65_VARS = [f"B01001_{i:03d}E" for i in range(20, 26)] + [f"B01001_{i:03d}E" for i in range(44, 50)]

NYC_COUNTIES = ["005", "047", "061", "081", "085"]  # Bronx, Brooklyn, Manhattan, Queens, Staten Island


def _geoid_from_index_item(item) -> str:
    """
    Robustly extract the concatenated GEOID from either:
      - a censusgeo object (has .params()) OR
      - a tuple/list of pairs like (('state','36'), ('county','005'), ('tract','000101'))
    """
    # censusgeo branch
    if hasattr(item, "params"):
        params_obj = item.params()
        # params() can be an OrderedDict OR a list of (key, val) tuples depending on version
        try:
            vals = list(params_obj.values())
        except AttributeError:
            vals = [t[1] for t in params_obj]
        return "".join(vals)

    # tuple-of-tuples branch
    if isinstance(item, (tuple, list)):
        # expect [('state','36'), ('county','005'), ('tract','000101')]
        try:
            return "".join([pair[1] for pair in item])
        except Exception:
            return "".join(map(str, item))

    # fallback
    return str(item)


def _fetch_tracts(state_fips: str, county_fips: str) -> pd.DataFrame:
    geo = cd.censusgeo([("state", state_fips), ("county", county_fips), ("tract", "*")])
    var_list = list(ACS_VARS.values()) + AGE_65_VARS
    df = cd.download("acs5", 2022, geo, var_list)

    # Build tract_geoid from the DataFrame index (supports both index shapes)
    # Convert index to a Series of python objects first 
    idx_series = pd.Series(list(df.index), index=df.index)
    tract_geoid = idx_series.apply(_geoid_from_index_item)

    df = df.copy()
    df["tract_geoid"] = tract_geoid.values

    # Friendly column names
    df = df.rename(columns={v: k for k, v in ACS_VARS.items()})

    # Senior share (guard against divide-by-zero)
    df["age_65_plus"] = df[AGE_65_VARS].sum(axis=1, min_count=1)
    denom = df["total_pop"].replace({0: pd.NA})
    df["share_65_plus"] = df["age_65_plus"] / denom

    keep = ["tract_geoid"] + list(ACS_VARS.keys()) + ["share_65_plus"]
    return df[keep].reset_index(drop=True)


def build() -> None:
    geo, _, s_geo = load_artifact("nyc_geo")
    if "LOADED" not in s_geo:
        raise SystemExit("❌ Need nyc_geo first (run: python -m src.features.spatial)")

    # Pull ACS for each NYC county
    parts = [_fetch_tracts("36", c) for c in NYC_COUNTIES]
    acs = pd.concat(parts, ignore_index=True).drop_duplicates("tract_geoid")

    # Merge onto per-request rows from nyc_geo
    merged = geo.merge(acs, on="tract_geoid", how="left")

    attach = merged["median_hh_income"].notna().mean() * 100.0
    print(f"✓  ACS attach rate: {attach:.1f}%  (rows={len(merged):,})")

    save_artifact("nyc_geo_acs2", merged)


if __name__ == "__main__":
    build()
