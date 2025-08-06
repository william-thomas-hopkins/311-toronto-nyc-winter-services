# src/modeling/cox.py

"""
Cox models A/B/C with k‐fold C‐index and PDP‐style curves (CSV+PNG).

Inputs
    artifacts/model_base.parquet

Outputs
    artifacts/cox_modelA_coefs.csv
    artifacts/cox_modelB_coefs.csv
    artifacts/cox_modelC_coefs.csv
    artifacts/cox_results_summary.csv
    artifacts/cox_kfold_scores.csv
    artifacts/cox_pdp_<var>.csv
    artifacts/cox_pdp_<var>.png
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact

CFG = get_config()
ART = Path(CFG["paths"]["ARTIFACTS_DIR"])

WX_PREFIXES = ("prcp_", "snowpx_", "freeze_")
ACS_COLS = ["median_hh_income", "median_age", "median_gross_rent", "share_65_plus"]
TOP_N_DESCRIPTORS = 30 # Limit for model stability, same as RSF

def _write_meta(path: Path, meta: Dict) -> None:
    side = path.with_suffix(path.suffix + ".meta.json")
    side.write_text(json.dumps(meta, indent=2, default=str))


def _zsafe(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce")
    mu, sd = arr.mean(), arr.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=arr.index)
    z = (arr - mu) / sd
    return z.fillna(0.0)


def _prep_base() -> Tuple[pd.DataFrame, List[str]]:
    df, _, st = load_artifact("model_base")
    if "LOADED" not in st:
        raise SystemExit("❌ Need model_base first.")
    df = df.copy()

    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
    df["ttc_hours"] = pd.to_numeric(df["ttc_hours"], errors="coerce")
    df = df[df["ttc_hours"] >= 0]

    wx = [c for c in df if c.startswith(WX_PREFIXES)]
    use = wx + [c for c in ACS_COLS if c in df] + ["borough", "descriptor", "event", "ttc_hours"]
    sub = df[use].copy()

    for c in wx + ACS_COLS:
        if c in sub.columns:
            if any(c.startswith(p) for p in WX_PREFIXES):
                sub[c] = sub[c].fillna(0)
            else: # ACS
                sub[c] = sub[c].fillna(sub[c].median())
            sub[c] = _zsafe(sub[c])

    top_levels = sub["descriptor"].value_counts().head(TOP_N_DESCRIPTORS).index
    sub["desc_group"] = np.where(sub["descriptor"].isin(top_levels), sub["descriptor"], "OTHER")
    sub["desc_group"] = sub["desc_group"].fillna("UNKNOWN").astype("category")

    sub = pd.get_dummies(sub, columns=["desc_group"], prefix="desc", drop_first=True)
    desc_one_hot_cols = [c for c in sub.columns if c.startswith("desc_")]

    sub = sub.dropna(subset=["borough", "ttc_hours", "event"])
    return sub.reset_index(drop=True), desc_one_hot_cols


def _clean_for_fit(df: pd.DataFrame, covs: List[str], strata: List[str]) -> pd.DataFrame:
    required_cols = covs + ["ttc_hours", "event"] + strata
    X = df[required_cols].copy()
    X = X.dropna().reset_index(drop=True)
    return X


def _fit_cox(
    df: pd.DataFrame,
    covs: List[str],
    strata: List[str],
    label: str
) -> CoxPHFitter:
    X = _clean_for_fit(df, covs, strata)
    if X.empty:
        raise ValueError(f"DataFrame for model {label} is empty after cleaning. Check input data.")
        
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        X,
        duration_col="ttc_hours",
        event_col="event",
        strata=strata or None,
        show_progress=False,
    )

    p = ART / f"cox_model{label}_coefs.csv"
    cph.summary.reset_index().rename(columns={"index":"variable"}).to_csv(p, index=False)
    _write_meta(p, {"n_obs": len(X)})
    return cph


def _kfold(df: pd.DataFrame, covs: List[str], strata: List[str], label: str):
    # The strata parameter is handled by the fitter, not the k-fold function itself.
    # So we combine all necessary columns into one DataFrame.
    X = _clean_for_fit(df, covs, strata)
    if len(X) < 100:
        print(f"⚠️ Skipping k-fold for model {label}, not enough data (n={len(X)})")
        return {"model": label, "cindex_mean": np.nan, "cindex_std": np.nan}

    k = 5 if len(X) > 1000 else 3
    
    # --- THIS IS THE FIX ---
    # The `strata` keyword argument has been removed from this function call.
    # The CoxPHFitter will find the 'borough' column in the dataframe `X` and use it correctly.
    scores = k_fold_cross_validation(
        CoxPHFitter(penalizer=0.1, strata=strata), # Define strata inside the fitter object
        X,
        duration_col="ttc_hours",
        event_col="event",
        k=k,
        scoring_method="concordance_index",
    )
    # --- END OF FIX ---

    out = {"model": label, "cindex_mean": round(np.mean(scores),3), "cindex_std": round(np.std(scores),3)}
    df_scores = pd.DataFrame([{"fold": i+1, "cindex": s} for i,s in enumerate(scores)])
    df_scores.to_csv(ART / f"cox_kfold_{label}_scores.csv", index=False)
    return out


def _pdp(df: pd.DataFrame, cph: CoxPHFitter, var: str, strata_ref: Dict[str,str], label: str):
    grid = np.linspace(df[var].quantile(0.05), df[var].quantile(0.95), 25)
    baseline_values = df[cph.params_.index].median()
    
    rows = []
    for v in grid:
        row_df = pd.DataFrame([baseline_values])
        row_df[var] = v
        
        for s, val in strata_ref.items():
            row_df[s] = val

        ph = float(cph.predict_partial_hazard(row_df).iloc[0])
        rows.append({"value": v, "partial_hazard": ph})

    df_pdp = pd.DataFrame(rows)
    p_csv = ART / f"cox_pdp_{label}_{var}.csv"
    df_pdp.to_csv(p_csv, index=False)
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_pdp["value"], df_pdp["partial_hazard"], marker="o")
    ax.set_xlabel(f"Z-scored: {var}")
    ax.set_ylabel("Partial Hazard (Risk)")
    ax.set_title(f"PDP: {label} ‒ {var}")
    fig.tight_layout()
    p_png = ART / f"cox_pdp_{label}_{var}.png"
    fig.savefig(p_png, dpi=150)
    plt.close(fig)
    return p_csv, p_png


def run() -> None:
    df, desc_cols = _prep_base()

    wx = [c for c in df if c.startswith(WX_PREFIXES) and c.endswith('_z')]
    acs = [c for c in ACS_COLS if c in df]
    
    covA = wx + acs
    covB = wx
    covC = wx + acs + desc_cols

    print("Fitting Model A (Weather + ACS)...")
    mA = _fit_cox(df, covA, ["borough"], "A")
    
    print("Fitting Model B (Weather only)...")
    mB = _fit_cox(df, covB, ["borough"], "B")
    
    print("Fitting Model C (Weather + ACS + Descriptor)...")
    mC = _fit_cox(df, covC, ["borough"], "C") 

    print("Running k-fold cross-validation...")
    outA = _kfold(df, covA, ["borough"], "A")
    outB = _kfold(df, covB, ["borough"], "B")
    outC = _kfold(df, covC, ["borough"], "C")
    pd.DataFrame([outA, outB, outC]).to_csv(ART/"cox_kfold_scores.csv", index=False)

    print("Generating Partial Dependence Plots...")
    strata_ref = {"borough": df["borough"].mode()[0]}
    for v in ["prcp_24h_z","snowpx_24h_z","freeze_24h_min_z"]:
        if v in df:
            _pdp(df, mA, v, strata_ref, "A")
            _pdp(df, mC, v, strata_ref, "C")

    print("✅ Cox modeling complete.")


if __name__ == "__main__":
    run()