# Reports

`GenAIReport` builds and saves the HTML reports used by the
GenAI learning scripts. It wraps the repository's shared HTML builder and
provides helpers for text, tables, key/value cards, grids, plots, and raw HTML.

```python
from lib.utility.genai.reports import GenAIReport

report = GenAIReport("My experiment")
report.grid([report.text("Result", "Add a short result summary.")])
```

Call `report.save(script_file, filename)` from a script to save the report in
that script's `reports/` folder. Set `GENAI_OPEN_REPORTS=0` to prevent reports
from opening automatically.
