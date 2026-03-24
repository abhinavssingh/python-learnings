def card(title: str, content: str) -> str:
    return f"""
<div class="rounded-xl bg-white shadow-md border border-gray-200 p-4">
    <h2 class="text-lg font-semibold mb-3">{title}</h2>
    {content}
</div>
"""


def grid(cards: list[str], columns: int = 3) -> str:
    inner = "\n".join(cards)
    return f"""
<div class="grid gap-6 grid-cols-1 md:grid-cols-2 xl:grid-cols-{columns}">
  {inner}
</div>
"""
