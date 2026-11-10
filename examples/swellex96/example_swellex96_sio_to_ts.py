"""Convert a Swellex96 recording from its native formats to WaveKernelRT's formats
including exporting its array geometry."""

# ------------------------------------------------------------------
# Assumed directory layout
# ------------------------------------------------------------------
# Place swellex96_utilities.py beside this script and install WaveKernelRT
# in the Python environment used to run it.
#
# data_root/
#   Arrays/
#     positions_hlanorth_raw.txt
#     positions_hlasouth_raw.txt
#   EventS5/
#     J1312340.hla.north.sio
#     J1312340.hla.south.sio
#   EventS59/
#     J1341145.hla.north.sio
#     J1341145.hla.south.sio
#
# Only the geometry file and recording for the selected array/event are
# required. Folder and file names must match the capitalization shown above.
#
# Outputs are written under output_root using the same event folders:
#   Arrays/hla_<north-or-south>_geometry.txt
#   EventS5/J1312340.hla.<north-or-south>.ts
#   EventS59/J1341145.hla.<north-or-south>.ts
#
# Each run writes one geometry file and one .ts file for the selections below.
# Output directories are created when needed. Existing output files are
# overwritten. data_root and output_root may point to the same directory.

# General imports
import numpy as np
from datetime import datetime, timezone

from example_paths import SWELLEX96_DATA_ROOT, SWELLEX96_OUTPUT_ROOT
from swellex96_utilities import sio_open, sio_read, convert_geometry
from wavekernelrt.core.ts_file_utils import ts_write_header, ts_write_sequence


# ------------------------------------------------------------------
# User settings — edit paths in example_paths.py
# ------------------------------------------------------------------
data_root = SWELLEX96_DATA_ROOT
output_root = SWELLEX96_OUTPUT_ROOT

event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"


# ------------------------------------------------------------------
# Resolve inputs and outputs from the selections
# ------------------------------------------------------------------
# Select the recording, start time, and channel ordering.

# Read and create the geometry file the way WaveKernelRT expects it
file_in = data_root / "Arrays" / f"positions_hla{array.lower()}_raw.txt"
file_out = output_root / "Arrays" / f"hla_{array.lower()}_geometry.txt"

file_out.parent.mkdir(parents=True, exist_ok=True)

convert_geometry(file_in,file_out)

# set the channels per each array
match array.lower():
    case "north":
       # north channels
        channels = np.array([1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,19,21,22,23,24,26,27,
                             28,29,30,32])
    case "south":
        # south channels
        channels = np.array([1,2,3,4,5,6,7,8,10,11,12,13,15,17,18,19,20,21,22,23,24,25,26,
                             27,28,29,30,32])

match event.lower():
    case "s5":
        filename_sio = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.sio"
        filename_ts = output_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
        start_time = datetime(1996, 5, 10, 23, 40, 0, tzinfo=timezone.utc)
    case "s59":
        filename_sio = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.sio"
        filename_ts = output_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"
        start_time = datetime(1996, 5, 13, 11, 45, 0, tzinfo=timezone.utc)

filename_ts.parent.mkdir(parents=True, exist_ok=True)

start_unix_nanoseconds = int(round(start_time.timestamp() * 1e9))

# ------------------------------------------------------------------
# Convert the recording
# ------------------------------------------------------------------
sample_rate_hz = 3276.8

sio_header = sio_open(filename_sio,sample_rate_hz)

ts_header = ts_write_header(filename_ts,sio_header["num_seqs"],sample_rate_hz,start_unix_nanoseconds,
                            sio_header["pts_per_seq"],channels)

for idx in range(sio_header["num_seqs"]):
    data_NT = sio_read(sio_header,idx)
    ts_write_sequence(ts_header,data_NT)

sio_header["fid"].close()
ts_header["fid"].close()
