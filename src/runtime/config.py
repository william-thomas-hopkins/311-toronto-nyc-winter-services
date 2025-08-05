#src/runtime/config.py

#project wide run-time configurations
#create a single source of truth config dict, persists it as artifacts/CONFIG.json, and exposes 'get_config()' for all other modules

from __future__ import annotations
import os, sys, json, hashlib, random
from pathlib import Path
from typing import Any, Dict, Optional

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG: Dict[str, Any] = {
    "version": 4,
    "code_tag": "311_snow_modular_v1.0",
    # In src/runtime/config.py

    # In src/runtime/config.py

    "paths": {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "ARTIFACTS_DIR": str(PROJECT_ROOT / "artifacts"),
        "DATA_DIR": str(PROJECT_ROOT / "data"),
        # This is the NEW, specific path that toronto.py needs:
        "RAW_TORONTO_DIR": str(PROJECT_ROOT / "data" / "raw" / "toronto"),
        "PROCESSED_DATA_DIR": str(PROJECT_ROOT / "data" / "processed"),
        "FIG_DIR": str(PROJECT_ROOT / "artifacts" / "figs"),
        "CACHE_DIR": str(PROJECT_ROOT / "artifacts" / "cache"),
    },
    "api": {
    "NYC_APP_TOKEN_ENV_VAR": "NYC_APP_TOKEN",
    "NYC_BASE_URL": "https://data.cityofnewyork.us",
    "NYC_DATASET_ID": "erm2-nwe9",
    },
    "libs": {
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "scikit-learn": "1.4.2",
        "scikit-survival": "0.23.0",
        "lifelines": "0.27.8",
        "geopandas": "0.14.4",
        "hdbscan": "0.8.33",
        "meteostat": "1.6.8",
        "sodapy": "2.2.0",
        "censusdata": "1.15",
        "plotly": "5.22.0",
    },
    "analysis": {
        "REWORK_DIST_M": 50,
        "REWORK_WINDOW_H": 48,
        "APPLY_WINTER_GUARD": True,
        "WINTER_MONTHS": [11, 12, 1, 2, 3, 4],
        "TOP_DESC_PLOT": 5,
    },
    "seeds": {"PYTHON_SEED": 42, "NUMPY_SEED": 42},
}

def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj

def _compute_config_hash(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(_make_serializable(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def _apply_seeds(cfg: Dict[str, Any]) -> None:
    random.seed(cfg["seeds"]["PYTHON_SEED"])
    os.environ["PYTHONHASHSEED"] = str(cfg["seeds"]["PYTHON_SEED"])
    try:
        import numpy as np
        np.random.seed(cfg["seeds"]["NUMPY_SEED"])
    except ImportError:
        pass

def get_nyc_app_token() -> Optional[str]:
    return os.getenv(CONFIG["api"]["NYC_APP_TOKEN_ENV_VAR"])

_CONFIG_INSTANCE: Optional[Dict[str, Any]] = None

def get_config() -> Dict[str, Any]:
    """
    Lazy-initialise CONFIG; returns the *same* dict on every call.
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is not None:
        return _CONFIG_INSTANCE

    # create dirs
    for p in CONFIG["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)

    _apply_seeds(CONFIG)
    CONFIG["config_hash"] = _compute_config_hash(CONFIG)

    cfg_path = Path(CONFIG["paths"]["ARTIFACTS_DIR"]) / "CONFIG.json"
    cfg_path.write_text(json.dumps(_make_serializable(CONFIG), indent=2))

    print(f"✅ CONFIG initialised → {cfg_path}  (hash={CONFIG['config_hash']})")
    _CONFIG_INSTANCE = CONFIG
    return CONFIG

if __name__ == "__main__":
    get_config()

   
# we can run as `python -m src.runtime.config` for a quick sanity check