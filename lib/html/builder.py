"""
HtmlBuilder: A unified class-based interface for building HTML reports.

This class provides a single import point that internally delegates to the
modular builder classes (PageBuilder, ComponentsBuilder, RenderersBuilder),
keeping the codebase clean and fully object-oriented.

Usage:
    from lib.html import HtmlBuilder
    builder = HtmlBuilder()
    html = builder.build_page(
        "Report Title",cl
        builder.grid([
            builder.card("Card 1", builder.render_array(np_array)),
            builder.card("Card 2", builder.render_dict({"key": "value"})),
        ])
    )
"""

# Import builder classes
from .base import PageBuilder
from .components import ComponentsBuilder
from .renderers import RenderersBuilder


class HtmlBuilder:
    """
    Unified interface for HTML report generation.

    Internally delegates to PageBuilder, ComponentsBuilder, and RenderersBuilder
    for clean code organization. All HTML functionality accessible through a
    single class instance.
    """

    def __init__(self):
        """Initialize HtmlBuilder with all builder classes."""
        self.pages = PageBuilder()
        self.components = ComponentsBuilder()
        self.renderers = RenderersBuilder()

    # ============================================================================
    # PAGE BUILDER (delegate to PageBuilder)
    # ============================================================================

    def build_page(self, title: str, body_html: str) -> str:
        """
        Build a complete HTML page with Tailwind CSS styling and theme toggle.

        Args:
            title: Page title (shown in browser tab and header)
            body_html: Body content HTML

        Returns:
            Complete HTML page as string
        """
        return self.pages.build_page(title, body_html)

    # ============================================================================
    # COMPONENTS (delegate to ComponentsBuilder)
    # ============================================================================

    def card(self, title: str, content: str) -> str:
        """Create a standard card component."""
        return self.components.card(title, content)

    def grid(self, cards: list[str], columns: int = 3) -> str:
        """Create a responsive grid layout for cards."""
        return self.components.grid(cards, columns)

    def full_width_card(self, title: str, content: str) -> str:
        """Create a full-width card component."""
        return self.components.full_width_card(title, content)

    def chart_card(self, title: str, content: str, plotly_var: str) -> str:
        """Create a card optimized for charts."""
        return self.components.chart_card(title, content, plotly_var)

    def chart_grid(self, cards: list[str]) -> str:
        """Create a responsive 2x2 layout for charts."""
        return self.components.chart_grid(cards)

    def chart_full_width(self, title, content: str) -> str:
        """Wrap chart content with a fixed height container."""
        return self.components.chart_full_width(title, content)

    # ============================================================================
    # RENDERERS (delegate to RenderersBuilder)
    # ============================================================================

    def render_array(self, arr) -> str:
        """Render a NumPy array with shape and dtype information."""
        return self.renderers.render_array(arr)

    def render_series(self, s, max_visible_rows: int = 5) -> str:
        """Render a Pandas Series or list."""
        return self.renderers.render_series(s, max_visible_rows)

    def render_dataframe(self, df, max_visible_rows: int = 5) -> str:
        """Render a Pandas DataFrame with scrolling and modal view."""
        return self.renderers.render_dataframe(df, max_visible_rows)

    def render_dict(self, d: dict, title: str | None = None, max_visible_rows: int = 1) -> str:
        """Render a Python dictionary as a styled table."""
        return self.renderers.render_dict(d, title, max_visible_rows)

    def render_kv(self, rows, max_visible_rows: int = 5) -> str:
        """Render key-value pairs in a table."""
        return self.renderers.render_kv(rows, max_visible_rows)

    def render_pre(self, text: str, max_visible_lines: int = 15) -> str:
        """Render preformatted text (code snippets, error messages)."""
        return self.renderers.render_pre(text, max_visible_lines)

    def render_dataframe_collapsible(self, df, initial_rows: int = 15) -> str:
        """Render a DataFrame with "Show more" / "Show less" controls."""
        return self.renderers.render_dataframe_collapsible(df, initial_rows)
