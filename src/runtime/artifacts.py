# src/runtime/artifacts.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from src.runtime.config import get_config

CFG = get_config()
ARTIFACTS_DIR = Path(CFG["paths"]["ARTIFACTS_DIR"])
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS: Dict[str, Dict[str, Any]] = {
    # Ingestion
    "nyc_raw": {
        "filename": "nyc_raw.parquet",
        "schema_min": ["unique_key", "created_date", "closed_date", "descriptor", "latitude", "longitude"],
        "producer": "src.ingest.nyc",
        "desc": "NYC 311 snow/ice 311 SRs filtered via Socrata.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "tor_filtered": {
        "filename": "tor_filtered.parquet",
        "schema_min": [
            "created_date", "snapshot_date", "status",
            "service_request_type", "robust_pseudo_id"
        ],
        "producer": "src.ingest.toronto",
        "desc": "Toronto 311 snapshots filtered to snow/ice with pseudo IDs.",
        "stale_on": ["config_hash", "code_tag"],
    },

    # Features (7.x)
    "wx_boro": {
        "filename": "wx_boro.parquet",
        "schema_min": ["created_hour_utc", "borough", "temp", "prcp_mm"],
        "producer": "src.features.weather",
        "desc": "Meteostat hourly features by borough (with rollups).",
        "stale_on": ["config_hash", "code_tag"],
    },
    "nyc_geo": {
        "filename": "nyc_geo.parquet",
        "schema_min": ["unique_key", "created_date", "created_hour_utc", "latitude", "longitude", "tract_geoid", "borough"],
        "producer": "src.features.spatialize",
        "desc": "NYC SRs spatial join to census tracts + borough.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "nyc_boro_wx": {
        "filename": "nyc_boro_wx.parquet",
        "schema_min": ["unique_key", "created_date", "created_hour_utc", "borough", "prcp_mm"],
        "producer": "src.features.join_weather",
        "desc": "NYC SRs joined to borough-hour weather.",
        "stale_on": ["config_hash", "code_tag"],
    },
    "nyc_geo_acs2": {
        "filename": "nyc_geo_acs2.parquet",
        "schema_min": ["unique_key", "tract_geoid", "median_hh_income"],
        "producer": "src.features.acs",
        "desc": "NYC SRs joined to tract-level ACS metrics.",
        "stale_on": ["config_hash", "code_tag"],
    },

    # NYC analytics (6.x)
    "nyc_spatial_clusters": {
        "filename": "nyc_spatial_clusters.parquet",
        "schema_min": ["unique_key", "spatial_cluster"],
        "producer": "src.nyc.spatial",
        "desc": "HDBSCAN cluster labels for NYC requests.",
        "stale_on": ["config_hash", "code_tag"],
    },

    # Modeling (8.x)
    "model_base": {
        "filename": "model_base.parquet",
        "schema_min": ["unique_key", "ttc_hours", "event"],
        "producer": "src.modeling.build_base",
        "desc": "Unified modelling dataset (SR + WX + ACS).",
        "stale_on": ["config_hash", "code_tag"],
    },
}


def _spec(name: str) -> Dict[str, Any]:
    try:
        return ARTIFACTS[name]
    except KeyError:
        raise KeyError(f"Artifact '{name}' not registered.")

def _path(name: str) -> Path:
    return ARTIFACTS_DIR / _spec(name)["filename"]

def _meta_path(name: str) -> Path:
    p = _path(name)
    return p.with_suffix(p.suffix + ".meta.json")

def _lib_versions() -> Dict[str, str]:
    out = {}
    try:
        import numpy as np, pandas as pd, sklearn as sk
        out = {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sk.__version__}
    except Exception:
        pass
    return out

def save_artifact(name: str, df: pd.DataFrame, extra_meta: Optional[Dict[str, Any]] = None) -> None:
    """Atomic parquet write + JSON meta sidecar."""
    spec = _spec(name)
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)

    meta = {
        "artifact": name,
        "filename": spec["filename"],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "producer": spec.get("producer", ""),
        "config_hash": CFG.get("config_hash"),
        "code_tag": CFG.get("code_tag"),
        "lib_versions": _lib_versions(),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "desc": spec.get("desc", ""),
    }
    if extra_meta:
        def _san(v):
            if isinstance(v, pd.Timestamp):
                return (v.tz_convert("UTC") if v.tz is not None else v.tz_localize("UTC")).isoformat()
            return v
        meta.update({k: _san(v) for k, v in extra_meta.items()})
    _meta_path(name).write_text(json.dumps(meta, indent=2))
    print(f"💾 Saved {name}: {len(df):,} rows → {p}")

def load_artifact(name: str, expect_fresh: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], str]:
    """Return (df, meta, status) where status ∈ {'✅ LOADED','❌ MISSING','🔄 STALE (...)','⚠️ SCHEMA_WARN'}."""
    spec = _spec(name)
    p, mp = _path(name), _meta_path(name)
    if not p.exists() or not mp.exists():
        return None, None, "❌ MISSING"

    try:
        df = pd.read_parquet(p)
    except Exception:
        return None, None, "❌ MISSING"

    try:
        meta = json.loads(mp.read_text())
    except Exception:
        meta = {}

    status = "✅ LOADED"
    if expect_fresh:
        for fld in spec.get("stale_on", []):
            if meta.get(fld) != CFG.get(fld):
                status = f"🔄 STALE ({fld})"
                break

    need = set(spec.get("schema_min", []))
    if need and not need.issubset(df.columns):
        status = "⚠️ SCHEMA_WARN" if status == "✅ LOADED" else status

    return df, meta, status

def list_artifacts_status() -> List[Dict[str, str]]:
    """Tabular status for CLI."""
    rows: List[Dict[str, str]] = []
    for name, spec in ARTIFACTS.items():
        p, mp = _path(name), _meta_path(name)
        if p.exists() and mp.exists():
            st = p.stat()
            mod = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = f"{st.st_size/1024:.1f} KB" if st.st_size < 1024*1024 else f"{st.st_size/1024/1024:.1f} MB"
            df, meta, status = load_artifact(name, expect_fresh=True)
            if df is not None and isinstance(df, pd.DataFrame):
                # crude date range if standard columns exist
                dr_cols = [c for c in ["created_date", "closed_date", "snapshot_date"] if c in df.columns]
                if dr_cols:
                    d = pd.concat([pd.to_datetime(df[c], errors="coerce") for c in dr_cols], axis=0).dropna()
                    dr = f"{d.min():%Y-%m-%d} → {d.max():%Y-%m-%d}" if not d.empty else "N/A"
                else:
                    dr = "N/A"
                nrows = f"{len(df):,}"
            else:
                dr = "N/A"; nrows = "N/A"
            rows.append({
                "artifact": name,
                "status": status,
                "rows": nrows,
                "date_range": dr,
                "modified": mod,
                "notes": size_kb,
            })
        else:
            rows.append({
                "artifact": name,
                "status": "❌ MISSING",
                "rows": "N/A",
                "date_range": "N/A",
                "modified": "N/A",
                "notes": "",
            })
    return rows
