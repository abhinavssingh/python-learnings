from __future__ import annotations

from pathlib import Path

# Load Tailwind theme.css from same folder
TAILWIND_CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")


TAILWIND_CSS = (Path(__file__).parent / "theme.css").read_text()


def build_html_page(title: str, body_html: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
{TAILWIND_CSS}

/* --- FORCE LIGHT THEME (NO DARK MODE) --- */
:root {{
  color-scheme: light;
}}
</style>
</head>

<body class="bg-gray-50 text-slate-900 p-6">
  <div class="max-w-screen-2xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{title}</h1>
    {body_html}
  </div>
</body>
</html>
"""
