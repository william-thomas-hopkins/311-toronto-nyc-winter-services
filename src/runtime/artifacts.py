"""
src/runtime/artifacts.py
-----------------------------------------------------------------------
Central registry (single source of truth) + Parquet/CSV I/O helpers.
Any new artifact must be declared in ARTIFACTS below.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd

from src.runtime.config import get_config, _make_serializable

#  Config & paths                                                    
CFG = get_config()
ARTIFACTS_DIR = Path(CFG["paths"]["ARTIFACTS_DIR"])
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

#  Registry                                                         
ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "nyc_raw": {
        "filename": "nyc_raw.parquet",
        "schema_min": [
            "unique_key", "created_date", "closed_date",
            "descriptor", "latitude", "longitude"
        ],
        "producer": "src.ingest.nyc",
        "desc": "NYC 311 snow/ice requests (API filtered).",
        "stale_on": ["config_hash", "code_tag"],
    },
    "tor_filtered": {
        "filename": "tor_filtered.parquet",
        "schema_min": [
            "created_date", "snapshot_date", "status",
            "service_request_type", "robust_pseudo_id"
        ],
        "producer": "src.ingest.toronto",
        "desc": "Toronto 311 snow/ice requests (CSV snapshots).",
        "stale_on": ["config_hash", "code_tag"],
    },
    "wx_boro": {
        "filename": "wx_boro.parquet",
        "schema_min": [
            "created_hour_utc", "borough",
            "temp", "prcp_mm", "snow_proxy_mm"
        ],
        "producer": "src.features.weather",
        "desc": "Hourly Meteostat features aggregated by borough.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "nyc_geo": {
        "filename": "nyc_geo.parquet",
        "schema_min": [
            "unique_key", "created_date", "created_hour_utc",
            "tract_geoid", "borough"
        ],
        "producer": "src.features.spatialize",
        "desc": "NYC SRs spatially joined to census tracts & borough.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "nyc_boro_wx": {
        "filename": "nyc_boro_wx.parquet",
        "schema_min": ["unique_key", "created_hour_utc", "borough", "prcp_mm"],
        "producer": "src.features.join_weather",
        "desc": "SRs joined to borough-hour weather.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "nyc_geo_acs2": {
        "filename": "nyc_geo_acs2.parquet",
        "schema_min": ["unique_key", "tract_geoid", "median_hh_income"],
        "producer": "src.features.acs",
        "desc": "SRs joined to ACS tract-level socio-economics.",
        "stale_on": ["config_hash", "code_tag"],
    },
    # ── Analytics & modelling ──────────────────────────────────────
    "nyc_spatial_clusters": {
        "filename": "nyc_spatial_clusters.parquet",
        "schema_min": ["unique_key", "spatial_cluster"],
        "producer": "src.nyc.spatial",
        "desc": "HDBSCAN cluster labels for NYC requests.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "model_base": {
        "filename": "model_base.parquet",
        "schema_min": ["unique_key", "ttc_hours", "event"],
        "producer": "src.modeling.build_base",
        "desc": "Unified modelling dataset (SR + WX + ACS).",
        "stale_on": ["config_hash", "code_tag"],
    },
}

#  Internal helpers                                                  
def _spec(name: str) -> Dict[str, Any]:
    if name not in ARTIFACTS:
        raise KeyError(f"Artifact '{name}' not registered.")
    return ARTIFACTS[name]


def artifact_path(name: str) -> Path:
    return ARTIFACTS_DIR / _spec(name)["filename"]


def meta_path(name: str) -> Path:
    p = artifact_path(name)
    return p.with_suffix(p.suffix + ".meta.json")


#  Save & load                                                       
def save_artifact(
    name: str,
    df: pd.DataFrame,
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """
    Write dataframe + JSON side-car atomically.
    Overwrites if file already exists.
    """
    spec = _spec(name)
    fpath = artifact_path(name)
    tmp = fpath.with_suffix(".tmp")

    df.to_parquet(tmp, index=False)
    os.replace(tmp, fpath)

    meta = {
        "artifact": name,
        "filename": spec["filename"],
        "rows": int(len(df)),
        "columns": list(df.columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": spec.get("producer"),
        "config_hash": CFG["config_hash"],
        "code_tag": CFG["code_tag"],
        "desc": spec.get("desc", ""),
    }
    if extra_meta:
        meta.update(_make_serializable(extra_meta))

    meta_path(name).write_text(json.dumps(meta, indent=2, default=str))
    print(f"💾  Saved {name}: {len(df):,} rows → {fpath}")


def load_artifact(
    name: str, *, expect_fresh: bool = True
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
    """
    Returns (df, meta, status)
    status ∈ {'LOADED','MISSING','STALE (…)','SCHEMA_WARN'}
    """
    spec = _spec(name)
    fpath, mpath = artifact_path(name), meta_path(name)

    if not fpath.exists() or not mpath.exists():
        return None, {}, "MISSING"

    try:
        df = pd.read_parquet(fpath)
    except Exception:
        return None, {}, "MISSING"

    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    status = "LOADED"

    # freshness
    if expect_fresh:
        for k in spec.get("stale_on", []):
            if meta.get(k) != CFG.get(k):
                status = f"STALE ({k})"
                break

    # schema
    if status == "LOADED":
        need = set(spec["schema_min"])
        if not need.issubset(df.columns):
            status = "SCHEMA_WARN"

    return df, meta, status


#  Dashboard helper (used by cli/run.py)                             
def list_artifacts_status(expect_fresh: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, spec in ARTIFACTS.items():
        _, meta, status = load_artifact(name, expect_fresh=expect_fresh)
        fpath = artifact_path(name)
        modified = (
            datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            if fpath.exists()
            else None
        )
        rows.append(
            {
                "name": name,
                "status": status,
                "rows": meta.get("rows"),
                "modified": modified,
                "desc": spec.get("desc", ""),
            }
        )
    return rows
