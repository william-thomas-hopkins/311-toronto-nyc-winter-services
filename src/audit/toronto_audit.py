# src/audit/toronto_audit.py

import pandas as pd
from src.runtime.artifacts import load_artifact

def run_audit():
    """
    Loads the tor_filtered artifact and runs a series of sanity checks.
    """
    print("--- Running Toronto Data Audit ---")
    df, meta, status = load_artifact("tor_filtered", expect_fresh=False)

    if status != "LOADED":
        print("❌ Artifact 'tor_filtered' not found. Please run the ingestion first.")
        return

    # 1. Basic Stats
    print("\n[1] Basic Statistics")
    print(f"  - Total Rows: {len(df):,}")
    print(f"  - Date Range: {df['created_date'].min():%Y-%m-%d} to {df['created_date'].max():%Y-%m-%d}")
    print(f"  - Unique Pseudo IDs: {df['robust_pseudo_id'].nunique():,}")
    
    # 2. Content Spot-Check (Is it snow-related?)
    print("\n[2] Content Spot-Check (Top 10)")
    
    print("  --- Service Request Type ---")
    print(df['service_request_type'].value_counts(normalize=True).head(10).to_string())
    
    if 'division' in df.columns:
        print("\n  --- Division ---")
        print(df['division'].value_counts(normalize=True).head(5).to_string())

    if 'section' in df.columns:
        print("\n  --- Section ---")
        print(df['section'].value_counts(normalize=True).head(5).to_string())
        
    print("\n--- Audit Complete ---")

if __name__ == "__main__":
    run_audit()