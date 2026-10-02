from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver


def get_checkpointer(kind: str = "memory", path: str | Path | None = None):
    """
    memory -> InMemorySaver (lost when the process exits)
    sqlite -> SqliteSaver (conversation threads survive restarts)
    """
    if kind == "memory":
        return InMemorySaver()
    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        if path is None:
            raise ValueError("path is required for the sqlite checkpointer")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    raise ValueError(f"Unknown checkpointer '{kind}'")
