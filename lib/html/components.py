

"""
HTML Components Builder

Provides component classes for building HTML cards, grids, and layouts.
"""


class ComponentsBuilder:
    """Builder class for HTML components (cards, grids, layouts)."""

    def card(self, title: str, content: str) -> str:
        """
        Create a standard card component.

        Args:
            title: Card title
            content: Card content HTML

        Returns:
            Card HTML
        """
        return f"""
<div class="rounded-xl border border-slate-300 dark:border-slate-700
            p-6 bg-white dark:bg-slate-800 shadow-md">

    <h2 class="text-lg font-semibold mb-4 text-slate-800 dark:text-slate-100">
        {title}
    </h2>

    <div class="space-y-3">
        {content}
    </div>
</div>
"""

    def grid(self, cards: list[str], columns: int = 3) -> str:
        """
        Create a responsive grid layout for cards.

        Args:
            cards: List of card HTML strings
            columns: Number of columns on desktop (default 3)

        Returns:
            Grid HTML
        """
        return f"""
<div class="grid gap-6 md:grid-cols-2 xl:grid-cols-{columns}">
    {''.join(cards)}
</div>
"""

    def full_width_card(self, title: str, content: str) -> str:
        """
        Create a full-width card component.

        Args:
            title: Card title
            content: Card content HTML

        Returns:
            Full-width card HTML
        """
        return f"""
<section class="w-full rounded-xl border border-slate-300 dark:border-slate-700
               bg-white dark:bg-slate-800 shadow-md p-6 mb-6">

    <h2 class="text-lg font-semibold mb-4 text-slate-800 dark:text-slate-100">
        {title}
    </h2>

    <div class="w-full overflow-x-auto">
        {content}
    </div>
</section>
"""

    def chart_card(self, title: str, content: str) -> str:
        """
        Create a card optimized for charts (less padding, better proportions).

        Args:
            title: Card title
            content: Card content HTML

        Returns:
            Chart card HTML
        """
        return f"""
<div class="rounded-xl border border-slate-300 dark:border-slate-700
            bg-white dark:bg-slate-800 shadow-md p-4">

    <h3 class="text-base font-semibold mb-3
               text-slate-800 dark:text-slate-100">
        {title}
    </h3>

    <div class="w-full h-full">
        {content}
    </div>
</div>
"""

    def chart_grid_2x2(self, cards: list[str]) -> str:
        """
        Create a responsive 2x2 layout for charts.

        Args:
            cards: List of chart card HTML strings

        Returns:
            Grid HTML
        """
        return f"""
<div class="grid gap-6 grid-cols-1 md:grid-cols-2">
    {''.join(cards)}
</div>
"""

    def chart_container(self, content: str, height: int = 300) -> str:
        """
        Wrap chart content with a fixed height container.

        Args:
            content: Chart content HTML
            height: Container height in pixels (default 300)

        Returns:
            Container HTML
        """
        return f"""
<div class="w-full" style="height: {height}px;">
    {content}
</div>
"""
