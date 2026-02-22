# Python Learnings

Welcome to the **python-learnings** repository! This repo contains code samples, notes, and resources for learning and experimenting with Python.

## Contents

- `basics/` — Python fundamentals and syntax
- `projects/` — Mini-projects and exercises
- `notebooks/` — Jupyter notebooks for interactive learning
- `resources.md` — Useful links and references

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
## Tech Stack Used

- **Python 3.x** — Core programming language
- **Jupyter Notebook** — Interactive coding and documentation
- **Git** — Version control
- **VS Code** — Recommended code editor
- **pip** — Package management

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.