"""Shared local paths for the SWellEx-96 examples.

Edit the two paths below before running the examples. The input and output roots
may be the same directory.
"""

from pathlib import Path


# Root containing the downloaded Arrays, EventS5, and/or EventS59 folders.
SWELLEX96_DATA_ROOT = Path("/path/to/downloaded/swellex96")

# Root for converted recordings and derived WaveKernelRT products. Set this to
# the same path as SWELLEX96_DATA_ROOT if inputs and outputs should be together.
SWELLEX96_OUTPUT_ROOT = Path("/path/to/downloaded/swellex96")