# cli/run.py

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file at the project root
load_dotenv()

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import the config initializer at the top level
from src.runtime.config import get_config

def print_usage():
    """Prints the available commands."""
    print("\nUsage: python -m cli.run <command>")
    print("""
Available Commands:
    status              - Check the status of all artifacts.
    
    build-toronto       - Build the 'tor_filtered' artifact from raw CSVs.
    build-nyc           - Build the 'nyc_raw' artifact from the Socrata API.
    
    audit-toronto       - Run a series of checks on the Toronto artifact.
    
    (More commands will be enabled as we build the project)
    """)

def main():
    """The main entry point for the command-line interface."""
    get_config()

    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    if command == "status":
        from src.runtime.artifacts import list_artifacts_status
        print("\n🔍 Checking artifact status...")
        list_artifacts_status()

    elif command == "build-toronto":
        print("\n🍁 Building Toronto filtered artifact...")
        from src.ingest.toronto import build
        build()
        print("✅ Toronto build complete.")

    elif command == "build-nyc":
        print("\n🏙️  Building NYC raw artifact...")
        from src.ingest.nyc import build
        build()
        print("✅ NYC build complete.")

    elif command == "audit-toronto":
        print("\n🔍 Auditing Toronto filtered artifact...")
        from src.audit.toronto_audit import run_audit
        run_audit()
        
    else:
        print(f"\n❌ Unknown command: '{command}'")
        print_usage()

if __name__ == "__main__":
    main()