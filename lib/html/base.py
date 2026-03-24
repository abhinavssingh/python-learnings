from pathlib import Path

TAILWIND_CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")


def build_html_page(title: str, body_html: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{title}</title>

<style>
{TAILWIND_CSS}

/* Default theme (light) */
:root {{
  color-scheme: light;
}}

html.light {{
  --bg-color: #f8fafc;
  --fg-color: #0f172a;
}}

html.dark {{
  --bg-color: #0f172a;
  --fg-color: #e2e8f0;
}}

body {{
  background-color: var(--bg-color);
  color: var(--fg-color);
}}
</style>

<script>
function toggleTheme() {{
    const html = document.documentElement;
    const current = html.classList.contains("dark") ? "dark" : "light";
    const next = current === "light" ? "dark" : "light";
    html.classList.remove(current);
    html.classList.add(next);
    localStorage.setItem("report-theme", next);
}}

window.onload = () => {{
    const saved = localStorage.getItem("report-theme") || "light";
    document.documentElement.classList.add(saved);
}};
</script>

</head>

<body class="p-6">
  <div class="max-w-screen-2xl mx-auto">

    <div class="flex justify-between items-center mb-6">
      <h1 class="text-2xl font-bold">{title}</h1>

      <button onclick="toggleTheme()"
        class="px-4 py-2 rounded-lg bg-slate-700 text-white hover:bg-slate-800 transition">
        Toggle Theme
      </button>
    </div>

    {body_html}
  </div>
</body>
</html>
"""
