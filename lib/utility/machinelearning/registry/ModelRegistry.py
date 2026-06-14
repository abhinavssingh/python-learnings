import copy
import importlib
import inspect
import pkgutil
from typing import Dict

from lib.utility.logger import Logger
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
            except Exception:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):

                if (
                    issubclass(obj, BaseModelWrapper)
                    and obj is not BaseModelWrapper
                    and not inspect.isabstract(obj)
                ):
                    try:
                        instance = obj()

                        model_name = obj.__name__.replace("Wrapper", "").strip()

                        self._registry[model_name] = instance
                        Logger.info(f"Registered model wrapper: {model_name} ({obj.__module__})")

                    except Exception:
                        continue

    # ---------------------------------------------------
    # GET MODEL
    # ---------------------------------------------------

    def get_model(self, model_name: str) -> BaseModelWrapper:
        if model_name not in self._registry:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        return copy.deepcopy(self._registry[model_name])

    # ---------------------------------------------------
    # GET ALL MODELS
    # ---------------------------------------------------

    def get_all_models(self) -> Dict[str, BaseModelWrapper]:
        return copy.deepcopy(self._registry)

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

    def get_models_by_family(self, family: str):
        return {
            name: model
            for name, model in self._registry.items()
            if getattr(model, "family", None) == family
        }
