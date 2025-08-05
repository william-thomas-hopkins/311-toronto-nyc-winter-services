"""
Fetch snow/ice 311 tickets for NYC via the Socrata API and save → nyc_raw.parquet
Re-runs are incremental & idempotent.
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta
import hashlib, os, re, time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sodapy import Socrata

from src.runtime.config import get_config, get_nyc_app_token
from src.runtime.artifacts import load_artifact, save_artifact

CFG = get_config()
TOKEN = get_nyc_app_token() or ""

DOMAIN = CFG["api"]["NYC_BASE_URL"].replace("https://", "")
DATASET = CFG["api"]["NYC_DATASET_ID"]

def _server_predicate() -> str:
    inc = (
        "("
        "upper(descriptor) like '%SNOW%' OR upper(descriptor) like '%ICE%' OR "
        "upper(descriptor) like '%ICY%' OR upper(descriptor) like '%SLEET%' OR "
        "upper(descriptor) like '%SLUSH%' OR upper(descriptor) like '%PLOW%' OR "
        "upper(descriptor) like '%PLOWED%' OR upper(descriptor) like '%SALT%' OR "
        "upper(complaint_type) like '%SNOW%'"
        ")"
    )
    exc = (
        "("
        "upper(descriptor) not like '%POLICE%' AND "
        "upper(descriptor) not like '%ICE CREAM%' AND "
        "upper(descriptor) not like '%LICENSE%' AND "
        "upper(complaint_type) not like '%POLICE%' AND "
        "upper(complaint_type) not like '%LICENSE%'"
        ")"
    )
    return f"{inc} AND {exc}"

PREDICATE     = _server_predicate()
PRED_HASH     = hashlib.md5(PREDICATE.encode()).hexdigest()[:16]
MIN_DATE      = datetime(2022, 1, 1, tzinfo=timezone.utc)
PER_PAGE      = 50_000

def _month_windows(t0: datetime, t1: datetime):
    cur = datetime(t0.year, t0.month, 1, tzinfo=timezone.utc)
    while cur < t1:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(nxt, t1)
        cur = nxt

def _apply_text_guard(df: pd.DataFrame) -> pd.DataFrame:
    pat_inc = re.compile(r"\b(?:snow|ice|icy|slush|sleet|plow|salt)\b", re.I)
    pat_exc = re.compile(r"\b(?:police|ice\s*cream|license)\b", re.I)
    desc = df.get("descriptor", pd.Series("", index=df.index)).astype(str)
    ctyp = df.get("complaint_type", pd.Series("", index=df.index)).astype(str)
    keep = (desc.str.contains(pat_inc) | ctyp.str.contains(pat_inc)) & \
           ~(desc.str.contains(pat_exc) | ctyp.str.contains(pat_exc))
    return df[keep].copy()

def _fetch_window(cli: Socrata, t0: datetime, t1: datetime) -> List[dict]:
    rows, offset = [], 0
    where = (
        f"{PREDICATE} AND created_date >= '{t0:%Y-%m-%dT%H:%M:%S}' "
        f"AND created_date < '{t1:%Y-%m-%dT%H:%M:%S}'"
    )
    while True:
        batch = cli.get(DATASET, where=where, order="created_date",
                        limit=PER_PAGE, offset=offset)
        if not batch:
            break
        rows.extend(batch); offset += len(batch)
        if len(batch) < PER_PAGE:
            break
    return rows

def fetch_nyc() -> pd.DataFrame:
    cli = Socrata(DOMAIN, app_token=TOKEN or None, timeout=60)
    # incremental logic
    existing, meta, st = load_artifact("nyc_raw", expect_fresh=False)
    if "LOADED" in st and meta.get("predicate_hash") == PRED_HASH:
        start = pd.to_datetime(meta.get("max_created_date")).tz_convert("UTC") - timedelta(hours=1)
        existing_ok = True
    else:
        start, existing_ok, existing = MIN_DATE, False, None

    end = datetime.now(timezone.utc)
    all_rows: list[dict] = []
    for w0, w1 in _month_windows(start, end):
        all_rows.extend(_fetch_window(cli, w0, w1))

    df_new = pd.DataFrame.from_records(all_rows)
    for dt in ("created_date", "closed_date", "resolution_action_updated_date"):
        if dt in df_new.columns:
            df_new[dt] = pd.to_datetime(df_new[dt], utc=True, errors="coerce")
    for num in ("latitude", "longitude"):
        if num in df_new.columns:
            df_new[num] = pd.to_numeric(df_new[num], errors="coerce")

    df_new = _apply_text_guard(df_new)
    if existing_ok and existing is not None:
        df_all = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    if "unique_key" in df_all.columns:
        df_all = df_all.drop_duplicates("unique_key", keep="last")
    df_all = df_all.sort_values("created_date").reset_index(drop=True)

    return df_all


def build():
    df, _, st = load_artifact("nyc_raw")
    if st == "LOADED":
        print("nyc_raw already fresh ✅")
        return
    df = fetch_nyc()
    meta = {
        "predicate_hash": PRED_HASH,
        "max_created_date": df["created_date"].max(),
        "pct_closed": float(df["closed_date"].notna().mean() * 100),
    }
    save_artifact("nyc_raw", df, meta)

if __name__ == "__main__":
    build()