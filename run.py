#!/usr/bin/env python3
"""
Main runner to execute selected Python modules in order.

Supports nested packages like:
  Module-2/NumPy/*.py
  Module-2/Pandas/*.py

Usage:
  python run.py --config runlist.json
  python run.py --only "Module-2.NumPy.numpy_*"
  python run.py --only "NumPy.*" --only "Module-2.Pandas.pandas_*"
  python run.py --list
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from init import register_paths

register_paths()


from lib.utility.logger import Logger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent


# Any folder you consider a "root" for code discovery.
# The runner will recurse and treat any directory with __init__.py as a package.
DISCOVERY_ROOTS = [
    REPO_ROOT,                 # discover top-level packages (if any)
    REPO_ROOT / "Module-2",    # discover nested packages under Module-2
    # add more roots if needed, e.g., REPO_ROOT / "lib", REPO_ROOT / "Module-1"
]


def is_package_dir(path: Path) -> bool:
    return path.is_dir() and (path / "__init__.py").exists()


def discover_packages() -> List[Path]:
    """Recursively find package directories (those that contain __init__.py)."""
    packages: List[Path] = []
    for root in DISCOVERY_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            p = Path(dirpath)
            if is_package_dir(p):
                packages.append(p)
    # Deduplicate
    unique: List[Path] = []
    seen = set()
    for p in packages:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def module_name_from_path(py_file: Path) -> Optional[str]:
    """
    Convert a file path to an importable dotted module name.
    Example: /repo/Module-2/NumPy/numpy_basics_report.py -> Module-2.NumPy.numpy_basics_report
    Only valid if every directory up to a discovery root boundary has __init__.py (i.e., is a package).
    """
    # Find the nearest discovery root that is a parent of the file
    candidates = [root for root in DISCOVERY_ROOTS if root in py_file.parents]
    if not candidates:
        return None
    # Choose the deepest root so that the module path is correct
    base = max(candidates, key=lambda r: len(str(r)))
    rel = py_file.relative_to(base)
    parts = list(rel.parts)
    if not parts or not parts[-1].endswith(".py"):
        return None
    parts[-1] = parts[-1][:-3]  # strip .py
    # Verify every folder from base to the file (excluding the .py) is a package
    cur = base
    for part in parts[:-1]:
        cur = cur / part
        if not is_package_dir(cur):
            return None
    # Build dotted name relative to the chosen base root
    dotted = ".".join(([base.name] if base != REPO_ROOT else []) + parts)
    return dotted


def discover_modules():
    modules = []

    # Load folders from settings.json (used also by bootstrap)
    from init import register_paths
    project_root = register_paths()

    # Load config directly
    import json
    config = json.loads((project_root / "settings.json").read_text())

    search_paths = config.get("paths", [])

    for rel in search_paths:
        folder = (project_root / rel).resolve()

        if not folder.exists():
            continue

        # Recursively scan all *.py files
        for py in folder.rglob("*.py"):

            # Skip non-script files
            if py.name.startswith("_"):
                continue

            # Convert to dotted module path
            try:
                rel_path = py.relative_to(project_root)
            except ValueError:
                continue

            dotted = str(rel_path.with_suffix("")).replace(
                "\\", ".").replace("/", ".")

            modules.append(dotted)

    # Return unique, sorted list
    return sorted(set(modules))


def import_and_call(module_path: str, func_name: str = "main") -> int:
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        Logger.error(f"Import failed for {module_path}: {e}")
        return 1

    func = getattr(mod, func_name, None)
    if callable(func):
        Logger.info(f"Calling {module_path}.{func_name}()")
        try:
            rv = func()
            return 0 if (rv is None or rv == 0) else int(rv)
        except SystemExit as se:
            return int(se.code) if isinstance(se.code, int) else 1
        except Exception as e:
            Logger.error(
                f"Error while running {module_path}.{func_name}(): {e}")
            return 1
    else:
        Logger.debug(
            f"No callable `{func_name}` in {module_path}, falling back to `python -m`.")
        return 1


def run_as_module(module_path: str) -> int:
    cmd = [sys.executable, "-m", module_path]
    Logger.info(f"Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def load_runlist(path: Path) -> List[Tuple[str, Optional[str]]]:
    """
    JSON format:
    { "run": [ {"module": "Module-2.NumPy.numpy_basics_report", "function": "main"} ] }
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    run_items = data.get("run", [])
    if not isinstance(run_items, list):
        raise ValueError("`run` must be a list.")
    result: List[Tuple[str, Optional[str]]] = []
    for item in run_items:
        mod = item.get("module")
        func = item.get("function")
        if not mod:
            raise ValueError("Each item in `run` requires a 'module' key.")
        result.append((mod, func))
    return result


def filter_by_patterns(modules: List[str], patterns: List[str]) -> List[str]:
    selected = []
    for pat in patterns:
        # Normalize pattern to dotted imports (users may type slashes)
        pat = pat.replace("/", ".")
        selected.extend([m for m in modules if fnmatch.fnmatch(m, pat)])
    # dedupe in discovery order
    seen = set()
    ordered = []
    for m in modules:
        if m in selected and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run selected project scripts.")
    parser.add_argument("-c", "--config", type=str,
                        help="Path to runlist.json at project root")
    parser.add_argument("-o", "--only", action="append",
                        help="Glob(s), e.g. 'Module-2.NumPy.numpy_*'")
    parser.add_argument("--function", default="main",
                        help="Default function to call if present")
    parser.add_argument("--list", action="store_true",
                        help="List discovered modules and exit")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop at first failing script")
    args = parser.parse_args()

    # Ensure imports from repo root work
    sys.path.insert(0, str(REPO_ROOT))

    discovered = discover_modules()

    if args.list:
        print("Discovered modules:")
        for m in discovered:
            print(f"  - {m}")
        return 0

    to_run: List[Tuple[str, Optional[str]]] = []

    if args.config:
        runlist_path = Path(args.config)
        if not runlist_path.is_absolute():
            runlist_path = REPO_ROOT / runlist_path
        to_run = load_runlist(runlist_path)
        # Fill missing function with default
        to_run = [(m, f if f is not None else args.function)
                  for (m, f) in to_run]
    elif args.only:
        filtered = filter_by_patterns(discovered, args.only)
        if not filtered:
            print("No modules matched the given patterns.")
            return 1
        to_run = [(m, args.function) for m in filtered]
    else:
        # Default: run everything discovered
        to_run = [(m, args.function) for m in discovered]

    overall_rc = 0
    for module_path, func_name in to_run:
        print("\n" + "=" * 80)
        print(f"▶ Running: {module_path}  (function: {func_name})")
        print("=" * 80)
        rc = import_and_call(module_path, func_name=func_name)
        if rc != 0:
            rc = run_as_module(module_path)

        if rc != 0:
            print(f"✖ {module_path} failed with exit code {rc}")
            overall_rc = rc
            if args.stop_on_error:
                break
        else:
            print(f"✓ {module_path} completed successfully")

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
