from .formula import Formula


class FormulaRegistry:
    def __init__(self):
        self._store = {}

    def add(self, formula: Formula):
        self._store[formula.key] = formula

    def get(self, key: str) -> Formula:
        return self._store[key]

    def by_category(self, category: str):
        return [f for f in self._store.values() if f.category == category]

    def all(self):
        return list(self._store.values())
