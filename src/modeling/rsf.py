# src/modeling/rsf.py

"""
Random Survival Forest with memory-safe defaults.

Two modes:
- 'lite' (default): weather + ACS + borough only (no high-card descriptor)
- 'full': also includes descriptor (top-K levels only) via one-hot (still safe)

Inputs
- artifacts/model_base.parquet

Outputs
- artifacts/rsf_results_summary.json
- artifacts/rsf_permutation_importance.csv
- artifacts/rsf_model.joblib
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Literal, Tuple, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Import the SimpleImputer 
# This is the key component needed to handle missing numerical data.
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])

WX_PREFIXES = ("prcp_", "snowpx_", "freeze_")
ACS_COLS = ["median_hh_income", "median_age", "median_gross_rent", "share_65_plus"]
TOP_N_DESCRIPTORS = 30


def _prep(mode: Literal["lite", "full"] = "lite") -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    df, _, st = load_artifact("model_base")
    if "LOADED" not in st:
        raise SystemExit("❌ Need model_base first.")

    df = df.copy()
    df = df.dropna(subset=["ttc_hours", "event", "borough"])
    df["event"] = df["event"].astype(bool)

    # Separate column types for granular imputation 
    # We need to know which columns are weather vs. ACS to apply different imputation rules.
    wx_cols = [c for c in df.columns if any(c.startswith(p) for p in WX_PREFIXES)]
    acs_cols = [c for c in ACS_COLS if c in df.columns]
    cat_cols = ["borough"]

    if mode == "full" and "descriptor" in df.columns:
        top_levels = df["descriptor"].value_counts().head(TOP_N_DESCRIPTORS).index
        df["descriptor_topk"] = np.where(df["descriptor"].isin(top_levels), df["descriptor"], "OTHER")
        cat_cols += ["descriptor_topk"]

    X = df[wx_cols + acs_cols + cat_cols].copy()
    y = Surv.from_arrays(event=df["event"].to_numpy(), time=df["ttc_hours"].to_numpy())
    return X, y, wx_cols, acs_cols, cat_cols


def run(mode: Literal["lite", "full"] = "lite") -> None:
    X, y, wx_cols, acs_cols, cat_cols = _prep(mode=mode)


    # Each one first imputes NaNs and then scales the data.

    # For weather data, missing values are best treated as 0 (e.g., no rain).
    numeric_weather_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    numeric_acs_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    pre = ColumnTransformer(
        transformers=[
            ('weather', numeric_weather_pipe, wx_cols),
            ('acs', numeric_acs_pipe, acs_cols),
            ('cat', OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ],
        remainder='drop', 
        n_jobs=1,
    )

    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=1, 
        oob_score=False,
        random_state=CFG["seeds"]["PYTHON_SEED"],
    )

    pipe = Pipeline(steps=[("pre", pre), ("rsf", rsf)], verbose=False)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=CFG["seeds"]["PYTHON_SEED"], stratify=y["event"]
    )

    print("Fitting Random Survival Forest model... (This may take several minutes)")
    
    pipe.fit(X_tr, y_tr)
    print("Model fitting complete.")

    risk = pipe.predict(X_te)
    c_test = float(concordance_index_censored(y_te["event"], y_te["time"], risk)[0])
    print(f"Test Set Concordance-Index: {c_test:.4f}")

    # Permutation importance (already memory-safe)
    n_perm = min(2000, X_te.shape[0])
    print(f"Calculating permutation importance on a subsample of {n_perm} rows...")
    Xp = X_te.iloc[:n_perm].copy()
    yp = y_te[:n_perm]
    base_score = float(concordance_index_censored(yp["event"], yp["time"], pipe.predict(Xp))[0])

    importances = []
    rng = np.random.default_rng(CFG["seeds"]["PYTHON_SEED"])
    # Get feature names after one-hot encoding for the report
    feature_names = pipe.named_steps['pre'].get_feature_names_out()
    
    # We must permute the original columns in Xp, not the transformed ones.
    for col in Xp.columns:
        Xperm = Xp.copy()
        Xperm[col] = rng.permutation(Xperm[col].to_numpy())
        perm_score = float(concordance_index_censored(yp["event"], yp["time"], pipe.predict(Xperm))[0])
        importances.append({"feature": col, "delta_cindex": base_score - perm_score})

    imp_df = pd.DataFrame(importances).sort_values("delta_cindex", ascending=False)
    print("Permutation importance calculation complete.")

    # Save artifacts
    p_imp = ART / f"rsf_permutation_importance.csv"
    imp_df.to_csv(p_imp, index=False)
    print(f"💾 Saved RSF permutation importances → {p_imp}")

    p_model = ART / f"rsf_model.joblib"
    dump(pipe, p_model)
    print(f"💾 Saved RSF model → {p_model}")

    p_meta = ART / f"rsf_results_summary.json"
    p_meta.write_text(json.dumps({"cindex_test": c_test, "mode": mode, "n_estimators": rsf.n_estimators}, indent=2))
    print(f"💾 Saved RSF summary → {p_meta}")
    print("✅ RSF modeling complete.")


if __name__ == "__main__":
    mode_env = os.getenv("RSF_MODE", "lite")
    if mode_env not in ["lite", "full"]:
        mode_env = "lite"
    run(mode=mode_env)