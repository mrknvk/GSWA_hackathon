"""Check or clear notebook execution outputs before public redistribution."""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def clear_execution_outputs(notebook: dict) -> bool:
    """Remove execution outputs and widget state in place; report any change."""
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue

        has_outputs = bool(cell.get("outputs"))
        has_execution_count = cell.get("execution_count") is not None
        if has_outputs or has_execution_count:
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True

    metadata = notebook.get("metadata", {})
    if "widgets" in metadata:
        del metadata["widgets"]
        changed = True

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true", help="Clear outputs in place")
    args = parser.parse_args()
    notebooks_with_outputs = []

    for path in sorted((PROJECT_ROOT / "ml_pipeline").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if not clear_execution_outputs(notebook):
            continue

        notebooks_with_outputs.append(path.name)
        if args.clear:
            path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

    if notebooks_with_outputs and not args.clear:
        names = ", ".join(notebooks_with_outputs)
        parser.exit(1, f"Stored execution data in {names}; rerun with --clear\n")

    print("Notebook output check passed; code and source data are unchanged.")


if __name__ == "__main__":
    main()
