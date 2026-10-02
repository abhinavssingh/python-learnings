from pathlib import Path

import pandas as pd

from ..abstractions.document import Document


class TabularLoader:
    """
    Turn each row of a CSV / Excel file into a Document.

    content_columns  -> joined as "column: value" lines (the text that gets embedded)
    metadata_columns -> stored in metadata (useful for filtering)
    """

    def __init__(self, path: str | Path, content_columns: list[str] | None = None,
                 metadata_columns: list[str] | None = None, max_rows: int | None = None):
        self.path = Path(path)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.max_rows = max_rows

    def read_frame(self) -> pd.DataFrame:
        if self.path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(self.path)
        else:
            df = pd.read_csv(self.path)
        return df.head(self.max_rows) if self.max_rows else df

    def load(self) -> list[Document]:
        df = self.read_frame()
        columns = self.content_columns or [c for c in df.columns if c not in self.metadata_columns]
        documents = []
        for row_number, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in columns if pd.notna(row[col]))
            metadata = {"source": self.path.name, "row": int(row_number), "type": "table"}
            for col in self.metadata_columns:
                value = row[col]
                metadata[col] = value.item() if hasattr(value, "item") else value
            documents.append(Document(content, metadata))
        return documents
