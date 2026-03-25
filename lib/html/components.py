

def card(title: str, content: str) -> str:
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


def grid(cards: list[str], columns=3) -> str:
    return f"""
<div class="grid gap-6 md:grid-cols-2 xl:grid-cols-{columns}">
    {''.join(cards)}
</div>
"""


def full_width_card(title: str, content: str) -> str:
    return f"""
<section class="w-full rounded-xl border border-slate-300 dark:border-slate-700
               bg-white dark:bg-slate-800 shadow-md p-6">

    <h2 class="text-lg font-semibold mb-4 text-slate-800 dark:text-slate-100">
        {title}
    </h2>

    <div class="w-full overflow-x-auto">
        {content}
    </div>
</section>
"""
