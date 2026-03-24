#!/usr/bin/env python3
"""
Master Test Script
Runs all 3 execution modes to verify:
✔ init.py path registration
✔ direct script execution
✔ module execution (python -m)
✔ runner execution (python run.py)
✔ discovery and pattern matching
✔ runlist.json functionality
"""

import os
import subprocess
from pathlib import Path
import sys
import textwrap
from lib.logger import log_error

ROOT = Path(__file__).resolve().parent

# Scripts to test directly
DIRECT_SCRIPTS = [
    ROOT / "Module-2" / "NumPy" / "numpy_basics_report.py",
    ROOT / "Module-2" / "NumPy" / "numpy_transpose_report.py",
]

# Modules to test via `python -m`
MODULE_COMMANDS = [
    "Module-2.NumPy.numpy_basics_report",
    "Module-2.NumPy.numpy_transpose_report",
]


def banner(title):
    print("\n" + "=" * 80)
    print(f"▶ {title}")
    print("=" * 80)


def run_cmd(cmd):
    """Run a shell command and show output live."""
    print(f"\n$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    print("\n" + ("✓ SUCCESS" if result.returncode == 0 else "✖ FAILED"))
    return result.returncode


def main():
    print("\n\n==============================")
    print(" PYTHON PROJECT TEST SUITE ")
    print("==============================\n")

    # 1. Test init import
    banner("1. Testing init import")
    try:
        from init import register_paths
        register_paths()
        print("✓ init.py imported successfully")
    except Exception as e:
        print(f"✖ init import failed: {e}")
        log_error(f"init import failed: {e}")
        sys.exit(1)

    # 2. Test direct script execution
    banner("2. Testing direct script execution")
    for script in DIRECT_SCRIPTS:
        run_cmd(f'python "{script}"')

    # 3. Test python -m execution
    banner("3. Testing module execution: python -m")
    for mod in MODULE_COMMANDS:
        run_cmd(f'python -m "{mod}"')

    # 4. Test runner listing
    banner("4. Testing run.py --list")
    run_cmd("python run.py --list")

    # 5. Test runner pattern match
    banner('5. Testing run.py --only "Module-2.NumPy.*"')
    run_cmd('python run.py --only "Module-2.NumPy.*"')

    # 6. Test runner using runlist.json
    if (ROOT / "runlist.json").exists():
        banner("6. Testing run.py --config runlist.json")
        run_cmd("python run.py --config runlist.json")
    else:
        print("Skipping runlist.json test (file not found).")

    print("\n\n==============================")
    print(" TEST SUITE COMPLETE ")
    print("==============================\n")


if __name__ == "__main__":
    main()
