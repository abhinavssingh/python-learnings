from dataclasses import dataclass
from typing import Optional


@dataclass
class Formula:
    key: str
    title: str
    latex: str
    category: str
    subcategory: Optional[str] = None

    def render(self, builder, display: bool = True):
        return builder.math_card(
            self.title,
            builder.render_latex_formula(self.latex, display=display)
        )
