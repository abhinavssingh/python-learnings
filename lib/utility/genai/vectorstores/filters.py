from typing import Any, Callable

Filter = dict[str, Any] | Callable[[dict[str, Any]], bool] | None


def matches_filter(metadata: dict[str, Any], filter: Filter) -> bool:
    """
    Mongo-style metadata filtering (the same idea Chroma / Qdrant use).

        {"page": 5}                     equality
        {"page": {"$in": [4, 5]}}       membership
        {"page": {"$gte": 4, "$lt": 7}} ranges
        {"source": {"$ne": "x.pdf"}}    not equal
        lambda m: m["page"] > 3         any callable
    """
    if filter is None:
        return True
    if callable(filter):
        return bool(filter(metadata))
    for key, condition in filter.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                if op == "$eq" and value != target:
                    return False
                if op == "$ne" and value == target:
                    return False
                if op == "$in" and value not in target:
                    return False
                if op == "$nin" and value in target:
                    return False
                if value is None and op in {"$gt", "$gte", "$lt", "$lte"}:
                    return False
                if op == "$gt" and not value > target:
                    return False
                if op == "$gte" and not value >= target:
                    return False
                if op == "$lt" and not value < target:
                    return False
                if op == "$lte" and not value <= target:
                    return False
                if op == "$contains" and target not in str(value):
                    return False
        elif value != condition:
            return False
    return True
