from typing import Dict, List, Optional, Union

from .formula import Formula


class FormulaRegistry:

    def __init__(self):
        self._store: Dict[str, Formula] = {}

    def add(self, formula: Formula):
        """Add a formula to the registry"""
        if formula is None:
            raise ValueError("Formula cannot be None")

        self._store[formula.key] = formula

    def get(self, key: str) -> Formula:
        return self._store[key]

    # -----------------------
    # INTERNAL HELPER
    # -----------------------
    def _to_list(self, value):
        if isinstance(value, str):
            return [value]
        return value

    # -----------------------
    # CATEGORY FILTER
    # -----------------------
    def by_category(self, category: Union[str, List[str]]) -> List[Formula]:
        category = self._to_list(category)

        result: List[Formula] = []
        for f in self._store.values():
            f_categories = (
                self._to_list(f.category)
                if f.category is not None
                else []
            )

            if any(cat in f_categories for cat in category):
                result.append(f)

        return result

    # -----------------------
    # SUBCATEGORY FILTER
    # -----------------------
    def by_subcategory(
        self, subcategory: Union[str, List[str]]
    ) -> List[Formula]:

        subcategory = self._to_list(subcategory)

        result: List[Formula] = []
        for f in self._store.values():
            if f.subcategory is None:
                continue

            f_subcategories = self._to_list(f.subcategory)

            if any(sub in f_subcategories for sub in subcategory):
                result.append(f)

        return result

    # -----------------------
    # COMBINED FILTER
    # -----------------------
    def filter(
        self,
        category: Optional[Union[str, List[str]]] = None,
        subcategory: Optional[Union[str, List[str]]] = None,
    ) -> List[Formula]:

        results: List[Formula] = list(self._store.values())

        if category:
            category = self._to_list(category)
            results = [
                f for f in results
                if any(
                    cat in self._to_list(f.category)
                    for cat in category
                )
            ]

        if subcategory:
            subcategory = self._to_list(subcategory)
            results = [
                f for f in results
                if f.subcategory and any(
                    sub in self._to_list(f.subcategory)
                    for sub in subcategory
                )
            ]

        return results

    # -----------------------
    # GET ALL
    # -----------------------
    def all(self) -> List[Formula]:
        return list(self._store.values())
