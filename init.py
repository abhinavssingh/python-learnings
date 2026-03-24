# bootstrap.py
import json
import sys
from pathlib import Path


def register_paths():
    # Root is folder containing this file
    project_root = Path(__file__).resolve().parent

    # Add root
    sys.path.insert(0, str(project_root))

    # Load settings.json
    cfg = project_root / "settings.json"
    with open(cfg, "r") as f:
        paths = json.load(f).get("paths", [])

    # Register paths
    for rel in paths:
        full = (project_root / rel).resolve()
        sys.path.insert(0, str(full))

    return project_root


# autorun
register_paths()
