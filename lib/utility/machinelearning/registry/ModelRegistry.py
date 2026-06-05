import importlib
import inspect
import pkgutil
from typing import Dict

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper


class ModelRegistry:
    """
    Auto-discovery registry for all model wrappers.
    Dynamically loads all wrappers from models folder.
    """

    def __init__(self):
        self._registry: Dict[str, BaseModelWrapper] = {}
        self._load_models()

    # ---------------------------------------------------
    # ✅ AUTO LOAD ALL MODELS
    # ---------------------------------------------------

    def _load_models(self):

        base_package = "lib.utility.machinelearning.models"

        package = importlib.import_module(base_package)

        for _, name, _ in pkgutil.walk_packages(
                package.__path__,
                prefix=base_package + "."
        ):
            try:
                module = importlib.import_module(name)
            except Exception as e:
                print(f"Skipping module {name}: {e}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):

                # ✅ ensure valid wrapper and skip base classes
                if (
                    issubclass(obj, BaseModelWrapper)
                    and obj is not BaseModelWrapper
                    and not obj.__name__.endswith("ModelWrapper")   # ✅ THIS LINE
                ):
                    try:
                        instance = obj()

                        model_name = obj.__name__.replace("Wrapper", "")

                        self._registry[model_name] = instance

                        print(f"✅ Registered: {model_name}")  # debug

                    except Exception as e:
                        print(f"❌ Skipping {obj.__name__}: {e}")
                        continue

    # ---------------------------------------------------
    # GET MODEL
    # ---------------------------------------------------
    def get_model(self, model_name: str) -> BaseModelWrapper:
        if model_name not in self._registry:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        return self._registry[model_name]

    # ---------------------------------------------------
    # GET ALL MODELS
    # ---------------------------------------------------
    def get_all_models(self) -> Dict[str, BaseModelWrapper]:
        return self._registry

    # ---------------------------------------------------
    # MANUAL REGISTRATION (OPTIONAL)
    # ---------------------------------------------------
    def register_model(self, model_name: str, wrapper: BaseModelWrapper):
        self._registry[model_name] = wrapper

    def get_models_by_task(self, task: str):
        return {
            name: model
            for name, model in self._registry.items()
            if getattr(model, "task", None) == task
        }
