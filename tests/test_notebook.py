from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "vllm_client.ipynb"


def test_notebook_is_valid_and_has_no_outputs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [], "clear the notebook outputs before committing"
            assert cell.execution_count is None
