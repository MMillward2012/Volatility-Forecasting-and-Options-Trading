"""Remove saved quote-bearing notebook outputs; keep sources and research results."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUOTE_NOTEBOOKS = {
    "forward_inference_diagnostics.ipynb", "iv_surface_diagnostics.ipynb",
    "raw_svi_diagnostics.ipynb", "ssvi_surface_diagnostics.ipynb",
    "single_expiry_SSVI_slice.ipynb",
}


def prepare_public_notebooks(check=False):
    affected = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        changed = False
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            private_example = path.name == "trading_feasibility.ipynb" and "example_columns" in source
            if cell["cell_type"] == "code" and (path.name in QUOTE_NOTEBOOKS or private_example):
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    changed = True
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    cell.get("metadata", {}).pop("execution", None)
        if notebook.get("metadata", {}).pop("widgets", None) is not None:
            changed = True
        if changed:
            affected.append(path.name)
            if not check:
                path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    if check and affected:
        raise SystemExit("Quote-bearing outputs need clearing: " + ", ".join(affected))
    print("Public-output check passed." if check else "Cleared outputs: " + ", ".join(affected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    prepare_public_notebooks(parser.parse_args().check)
