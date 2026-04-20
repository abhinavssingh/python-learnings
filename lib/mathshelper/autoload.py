from .formula import Formula
from .metadata import FORMULA_METADATA


def auto_register_formulas(registry, constants_modules):
    """
    Register Formula objects automatically from LaTeX constants
    across multiple modules.
    """
    for module in constants_modules:
        for const_name, meta in FORMULA_METADATA.items():
            latex_value = getattr(module, const_name, None)
            if latex_value is None:
                continue

            registry.add(
                Formula(
                    key=const_name.replace("_ltx", ""),
                    title=meta["title"],
                    latex=latex_value,
                    category=meta["category"],
                    subcategory=meta.get("subcategory"),
                )
            )
