from __future__ import annotations

from functools import wraps

from lib.utility.logger import Logger


class ExceptionLoggingMixin:
    """Mixin that logs uncaught public method exceptions and re-raises them."""

    _SKIP_WRAP_NAMES = {
        "__class__",
        "__dict__",
        "__repr__",
        "__str__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__init_subclass__",
    }

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name in ExceptionLoggingMixin._SKIP_WRAP_NAMES or name.startswith("_"):
            return attr

        if not callable(attr) or getattr(attr, "_exception_logged_wrapper", False):
            return attr

        @wraps(attr)
        def wrapped(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except Exception as exc:
                class_name = type(self).__name__
                Logger.error(f"{class_name}.{name} failed: {exc}")
                raise

        setattr(wrapped, "_exception_logged_wrapper", True)
        return wrapped
