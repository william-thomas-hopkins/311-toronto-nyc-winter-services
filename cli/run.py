# cli/run.py

import sys
import os
from pathlib import Path

# This ensures that we can import from the 'src' directory
# no matter where we run the script from.
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def main():
    """Simple CLI to run project tasks."""
    if len(sys.argv) < 2:
        print("Usage: python -m cli.run <command>")
        print("Available commands: status")
        return

    command = sys.argv[1]

    if command == "status":
        # We need to initialize the config first for paths to be set up
        from src.runtime.config import get_config
        get_config() 
        
        # Now we can safely import and use the artifacts module
        from src.runtime.artifacts import list_artifacts_status
        print("🔍 Checking artifact status...")
        list_artifacts_status()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()