# lib/mathshelper/__init__.py

from . import probability_formulas, statistics_formulas
from .autoload import auto_register_formulas
from .formularegistry import FormulaRegistry

# Global, package-level registry
FORMULA_REGISTRY = FormulaRegistry()

# Auto-register ALL formulas from constants modules
auto_register_formulas(
    registry=FORMULA_REGISTRY,
    constants_modules=[
        probability_formulas,
        statistics_formulas,
    ]
)

__all__ = ["FORMULA_REGISTRY"]
