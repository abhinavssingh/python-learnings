import importlib
import pkgutil

# Final aggregated metadata
FORMULA_METADATA = FORMULA_METADATA = {}
for _, module_name, _ in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")

    # Scan module attributes
    for attr_name in dir(module):
        # ✅ Only pick metadata dicts
        if attr_name.endswith("_METADATA"):
            metadata_dict = getattr(module, attr_name)

            if isinstance(metadata_dict, dict):
                FORMULA_METADATA.update(metadata_dict)

# Export only the final registry
__all__ = ["FORMULA_METADATA"]
