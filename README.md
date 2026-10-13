# Gauss Solver

A desktop app for solving systems of linear algebraic equations (SLAE) with the
Gaussian elimination method, featuring a step-by-step mathematical report.

## Features

- Gaussian elimination with partial pivoting (NumPy)
- Detects unique solution, no solution, and infinite solutions
- Step-by-step report: row operations, forward/back substitution
- PyQt5 GUI with editable matrix input
- Export results to Excel (openpyxl)
- PyInstaller build spec included (`gauss-solver.spec`)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Run tests:

```bash
python -m unittest test_gauss_solver
```

## Project layout

| File | Purpose |
|------|---------|
| `gauss_solver.py` | Core solver (no GUI dependencies) |
| `main_window.py` | PyQt5 main window |
| `main.py` | Application entry point |
| `test_gauss_solver.py` | Unit tests |
| `example_*.txt` | Sample systems of equations |

## License

MIT — see [LICENSE](LICENSE).
