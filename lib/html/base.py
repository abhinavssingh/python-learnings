"""
HTML Page Builder

Provides page builder class for creating complete HTML pages with Tailwind CSS.
"""

from pathlib import Path

TAILWIND_CSS = (Path(__file__).parent /
                "theme.css").read_text(encoding="utf-8")


class PageBuilder:
    """Builder class for creating complete HTML pages."""

    def __init__(self):
        """Initialize PageBuilder with Tailwind CSS."""
        self.tailwind_css = TAILWIND_CSS

    def build_page(self, title: str, body_html: str) -> str:
        """
        Build a complete HTML page with Tailwind CSS styling and theme toggle.

        Args:
            title: Page title (shown in browser tab and header)
            body_html: Body content HTML

        Returns:
            Complete HTML page as string
        """
        return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{title}</title>

<script src="https://cdn.plot.ly/plotly-3.4.0.min.js" charset="utf-8"></script>


<script>
  window.MathJax = {{
            tex: {{
                inlineMath: [['\\(', '\\)']],
      displayMath: [['\\[', '\\]']]
    }},
    svg: {{fontCache: 'global' }}
  }};
</script>

<script async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>


<style>
{self.tailwind_css}

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


/* ---------- Enforce readable text in dark cards ---------- */
.dark .card,
.dark table,
.dark td,
.dark th,
.dark pre {{
        color: #e5e7eb; /* slate-200 */
}}

/* ---------------- Lazy Loading and Animation ---------------*/
.chart-card.hidden {{
    display: none;
}}

.chart-card {{
    animation: fadeIn 300ms ease-in;
}}

@keyframes fadeIn {{
    from {{opacity: 0; }}
        to  {{opacity: 1; }}
    }}


/* Math container fixes */
.math-container {{
            overflow - x: auto;
  word-break: normal;
}}

/* Ensure math is visible in light theme */
.math-container mjx-container {{
            color: inherit !important;
}}

/* Prevent MathJax from shrinking matrices too hard */
mjx-container[jax="CHTML"] {{
            max - width: 100%;
  overflow-x: auto;
}}

/* Optional: tighten vertical spacing */
.math-container .my-4 {{
            margin - top: 1rem;
  margin-bottom: 1rem;
}}

</style>

<script>
/* -------- Theme Toggle -------- */
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

/* -------- Row Toggle (DataFrame Collapsible) -------- */
function toggleRows(uid, showAll) {{
    var rows = document.querySelectorAll("tr[data-row='" + uid + "']");
    var visible = 0;

    rows.forEach(function(row) {{
        var initiallyHidden = row.getAttribute("data-hidden") === "1";

        if (!showAll && initiallyHidden) {{
            row.style.display = "none";
        }} else {{
            row.style.display = "table-row";
            visible++;
        }}
    }});

    var info = document.getElementById(uid + "-info");
    if (info) {{
        info.textContent = showAll
            ? "Showing all " + rows.length + " rows"
            : "Showing " + visible + " of " + rows.length + " rows";
    }}

    var showLessBtn = document.getElementById(uid + "-showless");
    if (showLessBtn) {{
        showLessBtn.style.display = showAll ? "inline-flex" : "none";
    }}

    if (!showAll && info) {{
        info.scrollIntoView({{ behavior: "smooth", block: "start" }});
    }}
}}


/* -------- Optimized Filter Rows (Collapsible-safe & Debounced) -------- */
let filterTimeout = null;

function filterRows(uid, query) {{
    if (filterTimeout) {{
        clearTimeout(filterTimeout);
    }}

    filterTimeout = setTimeout(function () {{
        const q = query.trim().toLowerCase();
        const rows = document.querySelectorAll("tr[data-row='" + uid + "']");
        let visible = 0;

        rows.forEach(function(row) {{
            const initiallyHidden = row.getAttribute("data-hidden") === "1";
            const text = row.textContent.toLowerCase();

            if (q) {{
                // Filtering active → ignore collapsible state
                if (text.includes(q)) {{
                    row.style.display = "table-row";
                    visible++;
                }} else {{
                    row.style.display = "none";
                }}
            }} else {{
                // Filter cleared → restore collapsible default
                if (initiallyHidden) {{
                    row.style.display = "none";
                }} else {{
                    row.style.display = "table-row";
                    visible++;
                }}
            }}
        }});

        const info = document.getElementById(uid + "-info");
        if (info) {{
            info.textContent = q
                ? "Filtered: " + visible + " of " + rows.length + " rows"
                : "Showing first " + visible + " of " + rows.length + " rows";
        }}

        // Reset Show Less button when filter cleared
        const showLessBtn = document.getElementById(uid + "-showless");
        if (showLessBtn) {{
            showLessBtn.style.display = q ? "inline-flex" : "none";
        }}

    }}, 150); // debounce
}}

/* -------- Modal Logic (Layout Stable) -------- */
function openModalFromTemplate(templateId) {{
    var tpl = document.getElementById(templateId);
    if (!tpl) return;

    document.getElementById("modal-content").innerHTML = tpl.innerHTML;
    document.getElementById("global-modal").style.display = "flex";
}}

function closeModal() {{
    document.getElementById("global-modal").style.display = "none";
}}

document.addEventListener("keydown", function(e) {{
    if (e.key === "Escape") {{
        closeModal();
    }}
}});

/* -------- Plotly Chart Modal (multi-chart safe) -------- */

function openChartModal(templateId, plotlyVar) {{
    const tpl = document.getElementById(templateId);
    if (!tpl) return;

    document.getElementById("modal-content").innerHTML = tpl.innerHTML;
    document.getElementById("global-modal").style.display = "flex";

    const container = document.getElementById(templateId + "_modal_chart");
    if (!container) return;

    const store  = window.__PLOT_STORE__[plotlyVar];
    if (!store ) {{
    console.error("Plotly json not found:", plotlyVar);
    return ;
    }}

    Plotly.newPlot(
        container,
        store .fig.data,
        store .fig.layout,
        store .fig.config || {{}}
    );
}}



/* -------- Chart Batch Loader -------- */


let CHART_BATCH_SIZE = 2;
let currentChartIndex = 0;

function loadMoreCharts() {{
    const cards = Array.from(document.querySelectorAll(".chart-card"))
        .sort((a, b) => Number(a.dataset.chartIndex) - Number(b.dataset.chartIndex));

    let shownNow = 0;

    for (let i = currentChartIndex; i < cards.length && shownNow < CHART_BATCH_SIZE; i++) {{
        const card = cards[i];
        card.classList.remove("hidden");

        // ✅ Show parent wrapper div
        const parentWrapper = card.parentElement;
        if (parentWrapper) {{
            parentWrapper.style.display = "block";
        }}

        // ✅ Render Plotly chart if not already rendered
        const plotDiv = card.querySelector("[data-plotly-var]");
        if (plotDiv && !plotDiv.dataset.rendered) {{
            const plotId = plotDiv.id;
            const store = window.__PLOT_STORE__[plotId];

            if (store) {{
                Plotly.newPlot(plotDiv, store.fig.data, store.fig.layout, store.fig.config || {{}});
                plotDiv.dataset.rendered = "1";
            }}
        }}

        shownNow++;
    }}

    currentChartIndex += shownNow;

    // ✅ Update counter
    const status = document.getElementById("chart-status");
    if (status) {{
        status.textContent = "Showing " + currentChartIndex + " of " + cards.length + " charts";
    }}

    // ✅ Hide button if all charts loaded
    if (currentChartIndex >= cards.length) {{
        const btn = document.getElementById("load-more-btn");
        if (btn) btn.style.display = "none";
    }}
}}


document.addEventListener("DOMContentLoaded", () => {{
            loadMoreCharts();
        }});

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

<!-- -------- Global Modal (does NOT affect layout) -------- -->
<div id="global-modal"
    style="display:none"
    class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">

    <div class="bg-white dark:bg-slate-800
                max-w-6xl w-full mx-6
                rounded-xl shadow-lg
                max-h-[85vh] overflow-y-auto">

    <div class="flex justify-between items-center
                border-b border-slate-300 dark:border-slate-700
                px-6 py-4">

        <h3 class="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Details
        </h3>

        <button onclick="closeModal()"
                class="text-slate-500 hover:text-slate-900
                    dark:text-slate-400 dark:hover:text-white text-2xl">
        &times;
        </button>
    </div>

    <div id="modal-content" class="p-6"></div>
    </div>
</div>

</body>
</html>
"""
