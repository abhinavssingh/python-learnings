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
            function openChartModal(templateId, plotlyVarName) {{
                        const tpl = document.getElementById(templateId);
                        if (!tpl) return;

                        document.getElementById("modal-content").innerHTML = tpl.innerHTML;
                        document.getElementById("global-modal").style.display = "flex";

                        const container = document.getElementById(templateId + "_modal_chart");
                        if (!container) return;

                        const fig = window[plotlyVarName];
                        if (!fig) {{
                        console.error("Plotly figure not found:", plotlyVarName);
                        return ;
                    }}

                        Plotly.newPlot(
                            container,
                            fig.data,
                            fig.layout,
                            fig.config || {{}}
                        );
                    }}
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
