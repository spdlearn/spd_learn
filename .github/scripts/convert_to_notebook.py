"""
Convert Python example scripts to Jupyter notebooks for Colab integration.

This script converts Sphinx-Gallery style Python examples to Jupyter notebooks,
prepending installation cells for running in Google Colab.
"""

from __future__ import annotations

import argparse
import copy

from pathlib import Path

import nbformat

from sphinx_gallery import gen_gallery
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery.py_source_parser import split_code_and_text_blocks


# Installation cell content for SPD Learn notebooks
INSTALL_CELL_CONTENT = """%pip install -q spd_learn moabb braindecode scikit-learn matplotlib

# For GPU support (recommended for faster training)
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
"""


def convert_script_to_notebook(
    src_file: Path, output_file: Path, gallery_conf: dict
) -> None:
    """Convert a single Python script to a Jupyter notebook.

    Parameters
    ----------
    src_file : Path
        Path to the Python script.
    output_file : Path
        Path to the output notebook.
    gallery_conf : dict
        Sphinx-Gallery configuration dictionary.
    """
    # Parse the Python file
    _file_conf, blocks = split_code_and_text_blocks(str(src_file))

    # Convert to notebook (returns a dict, not a notebook object)
    example_nb_dict = jupyter_notebook(blocks, gallery_conf, str(src_file.parent))

    # Convert dict to nbformat notebook object
    example_nb = nbformat.from_dict(example_nb_dict)

    # Check if installation is already present
    first_source = ""
    if getattr(example_nb, "cells", None):
        try:
            first_source = example_nb.cells[0].source
        except (IndexError, AttributeError):
            first_source = ""

    # Prepend installation cell if not already present
    if "pip install" not in first_source or "spd_learn" not in first_source:
        install_cell = nbformat.v4.new_code_cell(source=INSTALL_CELL_CONTENT)
        install_cell.metadata["language"] = "python"
        example_nb.cells.insert(0, install_cell)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_notebook(example_nb, output_file)


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert a Python example script to a Jupyter notebook."
    )
    parser.add_argument("--input", required=True, help="Path to the Python script.")
    parser.add_argument("--output", required=True, help="Path to the output notebook.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    gallery_conf = copy.deepcopy(gen_gallery.DEFAULT_GALLERY_CONF)

    convert_script_to_notebook(input_path, output_path, gallery_conf)
    print(f"Notebook saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
