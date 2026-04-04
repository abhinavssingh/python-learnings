"""
HTML Components Builder

Provides component classes for building HTML cards, grids, and layouts.
"""

import uuid


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
<!-- ======================= Card: {title} ======================= -->
<div class="rounded-xl border border-slate-300 dark:border-slate-700
            p-6 bg-white dark:bg-slate-800 shadow-md">

    <!-- Card header -->
    <h2 class="text-lg font-semibold mb-4 text-slate-800 dark:text-slate-100">
        {title}
    </h2>

    <!-- Card body -->
    <div class="space-y-3">
        {content}
    </div>

</div>
<!-- ===================== End Card: {title} ===================== -->
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
<!-- ======================= Card Grid ======================= -->
<div class="grid gap-6 md:grid-cols-2 xl:grid-cols-{columns}">
    {''.join(cards)}
</div>
<!-- ===================== End Card Grid ===================== -->
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
<!-- ================ Full Width Card: {title} ================ -->
<section class="w-full rounded-xl border border-slate-300 dark:border-slate-700
            bg-white dark:bg-slate-800 shadow-md p-6 mb-6">

    <!-- Full-width card header -->
    <h2 class="text-lg font-semibold mb-4 text-slate-800 dark:text-slate-100">
        {title}
    </h2>

    <!-- Full-width card content -->
    <div class="w-full overflow-x-auto">
        {content}
    </div>

</section>
<!-- ============== End Full Width Card: {title} ============== -->
"""

    def chart_card(
        self,
        title: str,
        content: str,
        plotly_var: str,
        index: int,
        height: int = 500,
        modal_height: int = 500
    ) -> str:
        """
        Create a chart card with modal expansion support.

        Args:
            title: Chart title
            content: Inline chart HTML
            plotly_var: Plotly JS variable name (window-scoped)
            modal_height: Height of chart when shown in modal
        """
        uid = f"chart_{uuid.uuid4().hex[:8]}"

        return f"""
<!-- ======================= Chart Card: {title} ======================= -->

<div class="chart-card hidden rounded-xl border border-slate-300 dark:border-slate-700
            bg-white dark:bg-slate-800 shadow-md p-4" data-chart-index="{index}">

    <h3 class="text-base font-semibold mb-3
            text-slate-800 dark:text-slate-100">
        {title}
    </h3>

    <div class="w-full" style="height: {height}px;">
        {content}
    </div>

    <button onclick="openChartModal('{uid}', '{plotly_var}')"
            class="mt-3 text-sm px-3 py-1.5 rounded
                bg-slate-600 text-white hover:bg-slate-700">
        View details
    </button>

    <template id="{uid}">
        <div id="{uid}_modal_chart"
            class="w-full"
            style="height: {modal_height}px;">
        </div>
    </template>
</div>

<!-- ===================== End Chart Card: {title} ===================== -->
"""

    def chart_grid(self, cards: list[str], initial_visible: int = 4) -> str:
        """
        Render charts in a flexible grid with progressive loading.

        - Mobile: 1 column
        - Tablet/Desktop: 2 columns
        - Shows `initial_visible` charts first
        """
        wrapped_cards = "".join(
            f"""
<div data-chart-index="{i}"
    style="display:{'block' if i < initial_visible else 'none'}">
    {card}
</div>
"""
            for i, card in enumerate(cards)
        )

        return f"""
<!-- ======================= Chart Grid ======================= -->

<div class="text-sm text-slate-500 dark:text-slate-400 mb-3 text-center"
    id="chart-status">
    Showing 0 of {len(cards)} charts
</div>

<div class="grid gap-6 grid-cols-1 md:grid-cols-2 pt-6" id="chart-grid">
    {wrapped_cards}
</div>

<div class="flex justify-center mt-6 pb-6">
    <button id="load-more-btn"
            onclick="loadMoreCharts()"
            class="px-5 py-2 rounded-md bg-slate-600 text-white hover:bg-slate-700">
        Load more charts
    </button>
</div>
<!-- ===================== End Chart Grid ===================== -->
"""

    def chart_full_width(self, title: str, content: str, height: int = 320) -> str:
        """
        Render a full-width chart container with title and fixed height.
        """
        return f"""
<!-- =================== Full Width Chart: {title} =================== -->
<section class="w-full rounded-xl border border-slate-300 dark:border-slate-700
            bg-white dark:bg-slate-800 shadow-md p-6 mb-6">

    <!-- Full width chart title -->
    <h2 class="text-lg font-semibold mb-4
            text-slate-800 dark:text-slate-100">
        {title}
    </h2>

    <!-- Fixed height chart container -->
    <div class="w-full" style="height: {height}px;">
        {content}
    </div>

</section>
<!-- ================= End Full Width Chart: {title} ================= -->
"""
