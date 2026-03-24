# tools/build_tailwind.py

from __future__ import annotations


import sys
from pathlib import Path
from lib.logger import log_error

try:
    # pip install pytailwindcss
    import pytailwindcss as tw
except Exception as e:
    log_error(f"Failed to import pytailwindcss: {e}")
    print("pytailwindcss is not installed. Run: pip install pytailwindcss")
    raise


def main() -> int:
    # Resolve repo root as the directory two levels up from this file, adjust if needed
    root = Path(__file__).resolve().parent  # C:\IHFC\python-learnings
    assets = root / "assets"
    dist = root / "lib" / "html"

    input_css = assets / "input.css"   # e.g., contains: @import "tailwindcss";
    output_css = dist / "theme.css"

    print("Building Tailwind CSS...")
    print(f"ROOT    : {root}")
    print(f"INPUT   : {input_css}")
    print(f"OUTPUT  : {output_css}")

    # Sanity checks
    if not input_css.exists():
        print(f"ERROR: Input CSS not found at: {input_css}")
        print("Create it with something like:\n  @import \"tailwindcss\";")
        log_error("Input CSS file not found.")
        return 2

    dist.mkdir(parents=True, exist_ok=True)

    # Build argument list (DO NOT pass a single string)
    args = [
        "-i", str(input_css),
        "-o", str(output_css),
        "--minify",
    ]

    # Ensure the CLI binary exists or is downloaded
    # tw.run will auto-download if auto_install=True
    try:
        # Use cwd=root so relative @imports (if any) resolve from project root
        result = tw.run(args, auto_install=True, cwd=str(root))
    except Exception as e:
        log_error(f"Tailwind CLI invocation failed: {e}")
        print("Tailwind CLI invocation failed.")
        raise

    # pytailwindcss returns CompletedProcess; check code if available
    # (Some versions return stdout; if so, just ensure file exists)
    if isinstance(result, int) and result != 0:
        print(f"Tailwind exited with code {result}")
        return result

    if not output_css.exists():
        print("ERROR: Tailwind reported success but output file was not created.")
        return 3

    # Optional: also emit a min file name you prefer
    min_path = output_css.with_suffix(".min.css")
    try:
        # If you want a short name (tw.min.css), otherwise skip this block
        min_path.write_text(output_css.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    size_kb = output_css.stat().st_size / 1024
    print(f"Built {output_css} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
