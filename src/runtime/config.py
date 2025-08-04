#src/runtime/config.py

import os
import sys
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, Any

"""
I origionally worked through a lot of the code code in a colab notebook. Here I'm creating a single source of truth and combining the core logic of cells 0.1, 0.2, 0.3
"""

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG: Dict[str, Any] ={
    "version": 4,
    "code_tag": "311_snow_modular_v1.0",
    "paths": {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "ARTIFACTS_DIR": str(PROJECT_ROOT / 'artifacts'),
        "DATA_DIR": str(PROJECT_ROOT / "data"),
        "RAW_DATA_DIR": str(PROJECT_ROOT / "data" / "raw"),
        "PROCESSED_DATA_DIR": str(PROJECT_ROOT / "data" / "processed"),
        "FIG_DIR": str(PROJECT_ROOT / "artifacts" / "figs"),
        "CACHE_DIR": str(PROJECT_ROOT / "artifacts" / "cache"),
    },
    "api": {
        "NYC_APP_TOKEN_ENV_VAR": "JyXhUB9SyOtl0BQhv5p98l7nl",
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
        # Parameters from Cell 0.1
        "REWORK_DIST_M": 50,
        "REWORK_WINDOW_H": 48,
        "APPLY_WINTER_GUARD": True,
        "WINTER_MONTHS": [11, 12, 1, 2, 3, 4],
        "TOP_DESC_PLOT": 5,
    },
    "seeds": {"PYTHON_SEED": 42, "NUMPY_SEED": 42},
}

def _make_serializable(obj: Any) -> Any:
    """Recursively makes an object JSON serializable (e.g. converts Path objects to strings)"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj

def compute_config_hash(config: Dict[str, Any]) -> str:
    """Computes a unique and deterministic SHA256 hash of the configuration dictionary."""
    serializable_config = _make_serializable(config)
    config_string = json.dumps(serializable_config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_string.encode()).hexdigest()[:16]

def apply_seeds(config: Dict[str, Any]):
    """Applies random seeds for reproducibility across libraries."""
    py_seed = config["seeds"]["PYTHON_SEED"]
    random.seed(py_seed)
    os.environ["PYTHONHASHSEED"] = str(py_seed)
    try:
        import numpy as np
        np.random.seed(config["seeds"]["NUMPY_SEED"])
    except ImportError:
        print("Warning: numpy not found. Skipping numpy seed application.", file=sys.stderr)

def get_nyc_app_token() -> str | None:
    """Safely retrieves the NYC App Token from environment variables."""
    return os.getenv(CONFIG["api"]["NYC_APP_TOKEN_ENV_VAR"])

_CONFIG_INSTANCE = None

def get_config() -> Dict[str, Any]:
    """
    The main entry point for the rest of the application to get the configuration.
    
    Initializes the project configuration on first call: creates directories,
    applies seeds, computes the hash, and writes CONFIG.json. On subsequent
    calls, it returns the already loaded configuration.
    
    Returns:
        The fully hydrated CONFIG dictionary.
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is not None:
        return _CONFIG_INSTANCE

    print("🔧 Initializing project configuration...")

    # Create all necessary directories defined in config paths
    for path_str in CONFIG["paths"].values():
        Path(path_str).mkdir(parents=True, exist_ok=True)

    # Apply seeds for reproducibility
    apply_seeds(CONFIG)

    # Add the computed hash to the config itself for tracking
    CONFIG["config_hash"] = compute_config_hash(CONFIG)
    
    # Persist CONFIG.json as the single source of truth on disk for this run
    config_path = Path(CONFIG["paths"]["ARTIFACTS_DIR"]) / "CONFIG.json"
    try:
        config_path.write_text(json.dumps(_make_serializable(CONFIG), indent=2))
    except IOError as e:
        print(f"Error: Could not write CONFIG.json to {config_path}. Check permissions.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    print("✅ Configuration initialized and written to disk.")
    print(f"  - Project Root:     {CONFIG['paths']['PROJECT_ROOT']}")
    print(f"  - Artifacts Dir:    {CONFIG['paths']['ARTIFACTS_DIR']}")
    print(f"  - Config Hash:      {CONFIG['config_hash']}")
    
    _CONFIG_INSTANCE = CONFIG
    return _CONFIG_INSTANCE

if __name__ == "__main__":
    """This block allows you to run the script directly from the command line to perform the initial setup and verify it works. Usage: python -m src.runtime.config"""
    
    get_config()

