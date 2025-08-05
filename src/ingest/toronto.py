"""
Load Toronto monthly/yearly CSV snapshots, filter to snow/ice, build
robust_pseudo_id, deduplicate across snapshots and save → tor_filtered.parquet
"""

from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.runtime.config import get_config
from src.runtime.artifacts import load_artifact, save_artifact

CFG = get_config()
RAW_DIR = Path(CFG["paths"]["RAW_TORONTO_DIR"])

MONTH_ABBR = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
              'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
MONTH_FULL = {'JANUARY':1, 'FEBRUARY':2, 'MARCH':3, 'APRIL':4, 'MAY':5, 'JUNE':6,
              'JULY':7, 'AUGUST':8, 'SEPTEMBER':9, 'OCTOBER':10, 'NOVEMBER':11, 'DECEMBER':12}

def _infer_snapshot(fname: str) -> pd.Timestamp:
    """Infers the snapshot date from the raw CSV filename."""
    # Clean up common filename variations like "SR2022 (1).csv" -> "SR2022"
    name = Path(fname).stem.upper().split(' ')[0]

    # Rule 1: Yearly files like SR2024 -> Jan 8 of the next year
    m = re.fullmatch(r"SR(\d{4})", name)
    if m:
        year = int(m.group(1))
        return pd.Timestamp(year + 1, 1, 8, tz="UTC")

    # Rule 2: Monthly files like SR2025FEB or SR2025APRIL
    m = re.fullmatch(r"SR(\d{4})([A-Z]+)", name)
    if m:
        year = int(m.group(1))
        month_str = m.group(2)
        # Check both full name and abbreviation dictionaries
        mon = MONTH_FULL.get(month_str) or MONTH_ABBR.get(month_str)

        if mon is None:
            raise ValueError(f"Un-recognised month '{month_str}' in filename: {fname}")

        if mon == 12:
            return pd.Timestamp(year + 1, 1, 8, tz="UTC")
        return pd.Timestamp(year, mon + 1, 8, tz="UTC")

    raise ValueError(f"Un-recognised filename format: {fname}")

RENAME = {
    'Creation Date':'created_date',
    'Service Request Type':'service_request_type',
    'First 3 Chars of Postal Code':'postal_fsa',
    'First 3 Characters of Postal Code':'postal_fsa',
    'Division':'division',
    'Section':'section',
    'Status':'status',
    'Ward':'ward',
    'Intersection Street 1':'int_st1',
    'Intersection Street 2':'int_st2',
}

# Use a non-capturing group (?:...) to prevent the UserWarning
SNOW_RE = re.compile(r'\b(?:snow|ice|icy|salt|sanding|plow|plowing|sleet|slush|windrow)\b', re.I)

def _winter_mask(df: pd.DataFrame) -> pd.Series:
    """Creates a boolean mask for snow-related service requests within winter months."""
    m = pd.Series(False, index=df.index)
    for c in ('service_request_type', 'division', 'section'):
        if c in df.columns:
            m |= df[c].astype(str).str.contains(SNOW_RE, na=False, regex=True)
    
    if CFG["analysis"]["APPLY_WINTER_GUARD"]:
        months = set(CFG["analysis"]["WINTER_MONTHS"])
        cd = pd.to_datetime(df['created_date'], errors='coerce', utc=True)
        m &= cd.dt.month.isin(months)
    return m

def _norm(s: pd.Series) -> pd.Series:
    """Normalizes a string series for ID generation."""
    return (s.fillna("").astype(str)
              .str.upper().str.replace(r'\s+', ' ', regex=True).str.strip())

def _build_pseudo(df: pd.DataFrame) -> pd.Series:
    """Builds the robust pseudo ID for deduplication."""
    cd  = pd.to_datetime(df['created_date'], utc=True, errors='coerce')
    cds = cd.dt.strftime('%Y-%m-%dT%H:%M:%S')
    parts = [
        cds,
        _norm(df.get('service_request_type', '')),
        _norm(df.get('division', '')),
        _norm(df.get('section', '')),
        _norm(df.get('ward', '')),
        _norm(df.get('postal_fsa', '')),
        _norm(df.get('int_st1', '')),
        _norm(df.get('int_st2', '')),
    ]
    combo = pd.concat(parts, axis=1).agg('|'.join, axis=1)
    return combo.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])

def fetch_toronto() -> pd.DataFrame:
    """Loads, processes, and combines all Toronto CSV snapshots."""
    csvs = sorted(RAW_DIR.glob("SR*.csv"))
    if not csvs:
        raise SystemExit(f"No SR*.csv files found in the directory: {RAW_DIR}")

    frames: List[pd.DataFrame] = []
    print(f"Found {len(csvs)} Toronto CSV files to process...")
    for fp in csvs:
        try:
            df = pd.read_csv(fp, low_memory=False).rename(columns=RENAME)
            df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
            df['created_date'] = pd.to_datetime(df['created_date'], utc=True, errors='coerce')
            
            df = df[_winter_mask(df)].copy()
            if df.empty:
                continue

            df['snapshot_date']   = _infer_snapshot(fp.name)
            df['robust_pseudo_id'] = _build_pseudo(df)
            df['source_file']      = fp.name
            df['uncertainty_window_days'] = (
                df['snapshot_date'] - df['created_date']
            ).dt.total_seconds() / 86_400.0
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Could not process file {fp.name}. Error: {e}")

    if not frames:
        raise SystemExit("After filtering, no rows remain from any of the source files.")

    combined = (pd.concat(frames, ignore_index=True)
                  .sort_values(['created_date', 'snapshot_date'])
                  .reset_index(drop=True))

    # deduplicate across ALL snapshots to get unique requests
    final_df = combined.loc[combined.groupby('robust_pseudo_id')['snapshot_date'].idxmax()]
    return final_df.reset_index(drop=True)

def build():
    """Main function to build and save the tor_filtered artifact."""
    df, _, st = load_artifact("tor_filtered")
    if st == "LOADED":
        print("✅ tor_filtered artifact is already fresh.")
        return

    df = fetch_toronto()
    meta = {
        "files_used": sorted(list(df['source_file'].unique())),
        "date_range": {
            "min": str(df['created_date'].min()),
            "max": str(df['created_date'].max())
        },
        "median_uncertainty_days": float(df['uncertainty_window_days'].median()),
    }
    save_artifact("tor_filtered", df, meta)

if __name__ == "__main__":
    build()