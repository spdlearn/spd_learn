"""Pytest configuration for testing documentation code blocks.

Uses sybil to extract and test Python code blocks from RST files.
"""

from doctest import ELLIPSIS

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser


# Configure sybil to parse Python code blocks in RST files
pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS),
        PythonCodeBlockParser(),
    ],
    patterns=["source/**/*.rst"],
).pytest()
