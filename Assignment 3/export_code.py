"""Copies the code cells of 'Assignment 3 Homework.ipynb' into code/*.py, one file per problem,
so the report's "Full code" sections (\\lstinputlisting) always match the notebook.

Run from this folder after changing the notebook, then recompile the report:
    python export_code.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
NOTEBOOK = HERE / "Assignment 3 Homework.ipynb"
OUT = HERE / "code"

# report listing -> (notebook cell id, label printed above the cell), in notebook order
SECTIONS = {
    "setup_and_error_tools.py": [("a3-02", "Setup: imports, seeds, shared error-analysis tools"),
                                 ("a3-v0", "Validation of the error-analysis tools")],
    "problem_1A.py": [("a3-06", "Problem 1 functions: Ising energy and Metropolis sampler"),
                      ("a3-v1", "Validation of the Ising sampler"),
                      ("a3-07", "Problem 1A: runs at T = 2 from both starts, burn-in, trace plot"),
                      ("a3-08", "Problem 1A: autocorrelation, tau_int, blocking error")],
    "problem_1B.py": [("a3-12", "Problem 1B: pilot and production chains at every temperature"),
                      ("a3-13", "Problem 1B: plot of <|m|> and tau_int against T")],
    "problem_2A.py": [("a3-18", "Problem 2 functions: action, sampler, exact results"),
                      ("a3-v2", "Validation of the path-integral sampler"),
                      ("a3-19", "Problem 2A(i): scan of the hit size h"),
                      ("a3-20", "Problem 2A(i): production run and error analysis"),
                      ("a3-20b", "Problem 2A(i): plots")],
    "results_json.py": [("a3-30", "Validation summary and results.json")],
}


def main():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    index = {c["id"]: k for k, c in enumerate(cells)}
    OUT.mkdir(exist_ok=True)
    for filename, ids in SECTIONS.items():
        parts = []
        for cid, label in ids:
            k = index[cid]
            parts.append(f"# ---- notebook cell {k + 1}: {label} ----\n" + "".join(cells[k]["source"]).rstrip() + "\n")
        (OUT / filename).write_text("\n\n".join(parts), encoding="utf-8")
        print(f"wrote code/{filename}")


if __name__ == "__main__":
    main()
