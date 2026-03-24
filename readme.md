# Python Learnings

This repository contains a structured collection of Python learning exercises, utilities, report generators, and project experiments.  
It is organized into multiple modules to keep NumPy, Pandas, fundamental concepts, and utilities clearly separated.

The project includes a dynamic execution system (`run.py`) and a configuration-driven path management system (`init.py` + `settings.json`) to support flexible script execution without requiring Python packages or `__init__.py` files.
---
## Folder Structure
```
│   .editorconfig       # Defines consistent coding style across the project for all editors.   
│   .gitignore          # Specifies which files/folders Git should not track.
│   settings.json
├───.vscode
│       tasks.json      # Defines tasks you can run through Terminal → Run Task
├───assets
│       input.css       # Input Tailwind CSS file used by build_tailwind.py to generate final CSS.
├───lib
│   │   logger.py       # Lightweight logging utility for consistent console output (info/debug/error).
│   │   report_utils.py # Common report-building utilities for HTML or text generation.
│   ├───html            # HTML rendering engine which have many files.
│   init.py             # Used for dynamic path registration.
│   run.py              # Runner to execute through through Terminal → Run Task
│   runlist.json        # List of modules to run in sequence via run.py.
├───logs
├───Module-1            # Contains Python fundamentals
└───Module-2
    ├───NumPy           # NumPy Array Samples 
    └───Pandas          # Pandas Series and Dataframe Samples

```
---
## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/abhinavssingh/python-learnings.git
    ```
2. Navigate to the repo:
    ```bash
    cd python-learnings
    ```
3. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4. (Optional) Exclude Jupyter notebook output from version control by adding the following to your `.gitignore` file:
        ```
        # Ignore Jupyter notebook checkpoints and output
        **/.ipynb_checkpoints/
        *.ipynb
        ```
       Or, use the [`nbstripout`](https://github.com/kynan/nbstripout) tool to automatically strip output cells before committing:
        ```bash
        pip install nbstripout
        nbstripout --install
        ```
---
## Script Execution

All scripts support **three execution modes**, enabled through `init.py`.

### 1. Direct Script Execution
```
python Module-2/NumPy/numpy_basics_report.py
```

This mode runs the script using its file location as the working directory.  
`init.py` ensures imports such as:

```python
from lib.logger import log_info
from lib.html.base import build_html_page
```
work correctly.

---

### 2. Module Execution (`python -m package.module`)
```
python -m Module-2.NumPy.numpy_basics_report
```
This executes the script as a Python module.

---

### 3. Runner-Based Execution (`run.py`)
#### List all discovered modules:
```
python run.py --list
```

#### Run selected modules (pattern matching):
```
python run.py --only "Module-2.NumPy.*"
```

#### Run scripts defined in `runlist.json`:
```
python run.py --config runlist.json
```
---
## How Dynamic Path Resolution Works

### `init.py`
- Locates the project root by finding `settings.json`.
- Adds all configured folders to `sys.path`.
- Allows scripts to run without Python packages or `__init__.py` files.
- Ensures compatibility with all execution modes.

### `settings.json`
Defines importable directories:
```
{
    "paths": [
        "lib",
        "lib/tools",
        "Module-1",
        "Module-2"
    ]
}
```

---

## Tests
```
python test.py
```
This confirms:
- direct execution
- module execution
- runner discovery
- pattern matching
- runlist execution

---
## Tech Stack Used

- **Python 3.x** — Core programming language
- **Jupyter Notebook** — Interactive coding and documentation
- **Git** — Version control
- **VS Code** — Recommended code editor
- **pip** — Package management
- **tailwindcss** -- For report generation

---
## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.
---
## License

This project is licensed under the Apache License.