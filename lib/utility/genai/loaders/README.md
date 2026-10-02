# Loaders

Read source data into the framework's `Document` representation. The package
includes loaders for PDF, plain text, CSV/Excel tabular data, and directories,
plus `clean_pdf_text` for PDF text cleanup.

```python
from lib.utility.genai.loaders import PDFLoader
```

Use a loader before chunking and indexing when creating a retrieval pipeline.
The HR policy RAG examples show this flow through the shared factory; the
dataset helpers in `datasets.py` provide a ready-made corpus for those lessons.
