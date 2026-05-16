import importlib
import pkgutil

# Stores all imported constant modules
constants_modules = []

# Auto-discover and import modules inside this package
for _, module_name, _ in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    constants_modules.append(module)

# Optional: define what gets exported on "import *"
__all__ = ["constants_modules"]
