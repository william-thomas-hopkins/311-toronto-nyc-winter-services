# cli/run.py

"""
Simple CLI dispatcher.

Usage:
  python -m cli.run <command>

Common:
  status

Build:
  build-toronto
  build-nyc
  build-wx-boro
  build-nyc-geo
  build-nyc-boro-wx
  build-nyc-geo-acs
  build-model-base

Audit:
  audit-toronto

Analyze:
  analyze-nyc-survival
  analyze-nyc-temporal
  analyze-nyc-spatial
  analyze-nyc-rework

Model:
  model-cox
  model-rsf           (env RSF_MODE=lite|full, default lite)
"""

from __future__ import annotations
import os
import sys

from src.runtime.config import get_config

def _config_banner():
    # This function is called to initialize the global config and print a confirmation.
    cfg = get_config()
    print(f"✅ CONFIG initialised → {cfg['paths']['ARTIFACTS_DIR']}/CONFIG.json  (hash={cfg['config_hash']})\n")

# status 
def cmd_status():
    from src.runtime.artifacts import list_artifacts_status
    print("\n🔍 Checking artifact status...")
    # This uses pandas to create a nicely formatted table of artifacts
    from pandas import DataFrame
    print(DataFrame(list_artifacts_status()).to_string())

#  build 
def cmd_build_toronto():
    from src.ingest.toronto import build as build_toronto
    build_toronto()

def cmd_build_nyc():
    from src.ingest.nyc import build as build_nyc
    build_nyc()

def cmd_build_wx_boro():
    from src.features.weather import build as build_wx
    build_wx()

def cmd_build_nyc_geo():
    from src.features.spatial import build as build_geo
    build_geo()

def cmd_build_nyc_boro_wx():
    from src.features.join_weather import build as build_join
    build_join()

def cmd_build_nyc_geo_acs():
    from src.features.acs import build as build_acs
    build_acs()

def cmd_build_model_base():
    from src.modeling.build_base import build as build_base
    build_base()

# audit 
def cmd_audit_toronto():
    from src.audit.toronto_audit import run_audit as audit_toronto
    audit_toronto()

# analyze 
def cmd_analyze_nyc_survival():
    from src.nyc.survival import analyze
    analyze()

def cmd_analyze_nyc_temporal():
    from src.nyc.temporal import analyze
    analyze()

def cmd_analyze_nyc_spatial():
    from src.nyc.spatial import analyze
    analyze()

def cmd_analyze_nyc_rework():
    from src.nyc.rework import analyze
    analyze()

#  modeling 
def cmd_model_cox():
    from src.modeling.cox import run as cox_run
    cox_run()

def cmd_model_rsf():
    mode = os.getenv("RSF_MODE", "lite")
    from src.modeling.rsf import run as rsf_run
    rsf_run(mode=mode)

# This dictionary is the core of the dispatcher. It maps command strings to functions.
CMDS = {
    "status": cmd_status,

    "build-toronto": cmd_build_toronto,
    "build-nyc": cmd_build_nyc,
    "build-wx-boro": cmd_build_wx_boro,
    "build-nyc-geo": cmd_build_nyc_geo,
    "build-nyc-boro-wx": cmd_build_nyc_boro_wx,
    "build-nyc-geo-acs": cmd_build_nyc_geo_acs,
    "build-model-base": cmd_build_model_base,

    "audit-toronto": cmd_audit_toronto,

    "analyze-nyc-survival": cmd_analyze_nyc_survival,
    "analyze-nyc-temporal": cmd_analyze_nyc_temporal,
    "analyze-nyc-spatial": cmd_analyze_nyc_spatial,
    "analyze-nyc-rework": cmd_analyze_nyc_rework,

    "model-cox": cmd_model_cox,
    "model-rsf": cmd_model_rsf,
}

def main():
    _config_banner()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print("Usage: python -m cli.run <command>\n")
        print("Available Commands:")
        for k in sorted(CMDS):
            print(f"  {k}")
        sys.exit(0)

    cmd = sys.argv[1]
    fn = CMDS.get(cmd)
    if not fn:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    
   
    fn()

if __name__ == "__main__":
    main()