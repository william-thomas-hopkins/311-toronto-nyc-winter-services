# cli/run.py
from __future__ import annotations
import sys
from textwrap import shorten

from src.runtime.config import get_config
from src.runtime.artifacts import (
    list_artifacts_status,
    load_artifact,
)
# Stage 2/3 builders
from src.ingest.nyc import build as build_nyc_raw   
from src.ingest.toronto import build as build_toronto_filtered
from src.audit.toronto_audit import run_audit as audit_toronto_report  

def _print_status_table(rows: list[dict]) -> None:
    if not rows:
        print(" (no artifacts registered)")
        return
    # widths
    name_w = max(10, max(len(r["artifact"]) for r in rows))
    stat_w = 14
    rows_w = 10
    dr_w   = 28
    mod_w  = 17

    header = f"{'Artifact':<{name_w}} {'Status':<{stat_w}} {'Rows':>{rows_w}}  {'Date Range':<{dr_w}} {'Modified':<{mod_w}}  Notes"
    print(header)
    print("-" * len(header))
    for r in rows:
        notes = r.get("notes","")
        # keep notes short so table stays readable
        notes = shorten(str(notes), width=40, placeholder="…")
        print(f"{r['artifact']:<{name_w}} {r['status']:<{stat_w}} {r['rows']:>{rows_w}}  {r['date_range']:<{dr_w}} {r['modified']:<{mod_w}}  {notes}")

def cmd_status():
    get_config()  # ensure CONFIG.json exists + paths are ready
    print("\n🔍 Checking artifact status...")
    rows = list_artifacts_status()
    _print_status_table(rows)

def cmd_build_toronto():
    get_config()
    print("\n🍁 Building Toronto filtered artifact...")
    build_toronto_filtered()
    print("✅ Toronto build complete.")

def cmd_build_nyc():
    get_config()
    print("\n🏙️  Building NYC raw artifact...")
    build_nyc_raw()
    print("✅ NYC build complete.")

def cmd_audit_toronto():
    get_config()
    print("\n🔍 Auditing Toronto filtered artifact...")
    audit_toronto_report()
    print("--- Audit Complete ---")

def main():
    get_config()  # prints config line once per process
    cmds = {
        "status": cmd_status,
        "build-toronto": cmd_build_toronto,
        "build-nyc": cmd_build_nyc,
        "audit-toronto": cmd_audit_toronto,
    }

    if len(sys.argv) == 1:
        print("""
Usage: python -m cli.run <command>

Available Commands:
    status              - Check the status of all artifacts.
    
    build-toronto       - Build the 'tor_filtered' artifact from raw CSVs.
    build-nyc           - Build the 'nyc_raw' artifact from the Socrata API.
    
    audit-toronto       - Run a series of checks on the Toronto artifact.
    
(More commands will be enabled as we build the project)
""".strip())
        return

    cmd = sys.argv[1]
    fn = cmds.get(cmd)
    if not fn:
        print(f"Unknown command: {cmd}")
        print("Try: python -m cli.run")
        sys.exit(2)
    fn()

if __name__ == "__main__":
    main()
