# lib/mathshelper/__init__.py

from .autoload import auto_register_formulas
from .formularegistry import FormulaRegistry
from .formulas import constants_modules

# Global, package-level registry
FORMULA_REGISTRY = FormulaRegistry()

# Auto-register ALL formulas from constants modules
auto_register_formulas(
    registry=FORMULA_REGISTRY,
    constants_modules=constants_modules
)

__all__ = ["FORMULA_REGISTRY"]
