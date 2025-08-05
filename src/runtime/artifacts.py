#src/runtime/artifacts.py

"""
Artifact registry + save/load helpers.

Usage
-----
from src.runtime.artifacts import save_artifact, load_artifact

df, meta, status = load_artifact("nyc_raw")
if status != "LOADED":
    df = fetch_nyc_311()     # ← our ingest code
    save_artifact("nyc_raw", df, {"source": "socrata API"})
"""

from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

from src.runtime.config import get_config

CFG = get_config()
ARTIFACTS_DIR = Path(CFG["paths"]["ARTIFACTS_DIR"])

ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "nyc_raw": {
        "filename": "nyc_raw.parquet",
        "schema_min": [
            "unique_key", "created_date", "closed_date",
            "descriptor", "latitude", "longitude"
        ],
        "producer": "src.ingest.nyc",
        "desc": "NYC 311 snow/ice requests (filtered).",
        "stale_on": ["config_hash", "code_tag"],
    },
    "tor_filtered": {
        "filename": "tor_filtered.parquet",
        "schema_min": [
            "created_date", "snapshot_date", "status",
            "service_request_type", "robust_pseudo_id"
        ],
        "producer": "src.ingest.toronto",
        "desc": "Toronto 311 snow/ice requests (deduped across snapshots).",
        "stale_on": ["config_hash", "code_tag"],

    #downstream artifacts

    "nyc_spatial_clusters": {
        "filename": "nyc_spatial_clusters.parquet",
        "schema_min": ["unique_key", "spatial_cluster"],
        "producer": "src.nyc.spatial",
        "desc": "HDBSCAN cluster labels for NYC requests.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "wx_boro": {
        "filename": "wx_boro.parquet",
        "schema_min": ["created_hour_utc", "borough", "temp", "prcp_mm"],
        "producer": "src.features.weather",
        "desc": "Hourly Meteostat features aggregated to borough.",
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
}

def _lib_versions() -> Dict[str, str]:
    out = {}
    for mod in ("numpy", "pandas"):
        try:
            m = __import__(mod)
            out[mod] = m.__version__
        except Exception:
            pass
    return out

def _artifact_spec(name: str) -> Dict[str, Any]:
    if name not in ARTIFACTS:
        raise KeyError(f"Artifact '{name}' not registered.")
    return ARTIFACTS[name]

def artifact_path(name: str) -> Path:
    return ARTIFACTS_DIR / _artifact_spec(name)["filename"]

def meta_path(name: str) -> Path:
    return artifact_path(name).with_suffix(".parquet.meta.json")

# In src/runtime/artifacts.py

# REPLACE your old save_artifact function with this one:

def save_artifact(name: str,
                  df: pd.DataFrame,
                  extra_meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Atomically write <name>.parquet and side-car JSON meta.
    This version correctly handles nested lists and dicts in the metadata.
    """
    spec = _artifact_spec(name)
    path = artifact_path(name)
    tmp  = path.with_suffix(f"{path.suffix}.tmp") # Ensure temp name is unique

    # write parquet
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

    meta: Dict[str, Any] = {
        "artifact": name,
        "filename": spec["filename"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": spec.get("producer"),
        "config_hash": CFG["config_hash"],
        "code_tag": CFG["code_tag"],
        "lib_versions": _lib_versions(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "desc": spec.get("desc", ""),
    }
    
    if extra_meta:
        # This new recursive function can handle any nested structure
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(item) for item in obj]
            # Base cases for single values
            if isinstance(obj, (pd.Timestamp, np.datetime64)):
                ts = pd.to_datetime(obj)
                if ts.tzinfo is None:
                    return ts.tz_localize("UTC").isoformat()
                return ts.tz_convert("UTC").isoformat()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if pd.isna(obj):
                return None
            return obj

        # Sanitize the entire extra_meta dictionary recursively
        sanitized_extra_meta = _sanitize(extra_meta)
        meta.update(sanitized_extra_meta)

    meta_path(name).write_text(json.dumps(meta, indent=2))
    print(f"💾 Saved {name}: {len(df):,} rows → {path}")
    
def load_artifact(name: str,
                  expect_fresh: bool = True
                  ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
    """
    Returns (DataFrame | None, meta dict | {}, status str)
    status ∈ {'LOADED', 'MISSING', 'STALE (…)','SCHEMA_WARN'}
    """
    spec = _artifact_spec(name)
    p, mp = artifact_path(name), meta_path(name)
    if not p.exists() or not mp.exists():
        return None, {}, "MISSING"

    try:
        df = pd.read_parquet(p)
    except Exception:
        return None, {}, "MISSING"

    meta = json.loads(mp.read_text()) if mp.exists() else {}
    status = "LOADED"

    # staleness check
    if expect_fresh:
        for fld in spec.get("stale_on", []):
            if meta.get(fld) != CFG.get(fld):
                status = f"STALE ({fld})"
                break

    # schema check
    need = set(spec["schema_min"])
    if not need.issubset(df.columns):
        status = "SCHEMA_WARN" if status == "LOADED" else status

    return df, meta, status

def list_artifacts_status() -> None:
    for k in ARTIFACTS:
        df, _, st = load_artifact(k, expect_fresh=False)
        rows = len(df) if df is not None else "-"
        print(f"{k:<18} {st:<12} {rows:>8}")

if __name__ == "__main__":
    list_artifacts_status()