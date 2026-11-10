"""Compare original and reconstructed Swellex96 channel timeseries.

Run the TS-to-FFT and FFT-to-TS examples without frequency/channel cropping.
The original is a line; reconstructed samples are x markers.
The paired examples use a rectangular window and matching record lengths.
Small floating-point differences and discarded trailing data are expected.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt
from example_paths import SWELLEX96_OUTPUT_ROOT
from wavekernelrt.core.ts_file_utils import ts_read_header, ts_read_sequence


# ------------------------------------------------------------------
# User settings — edit these selections
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
reconstructed_root = SWELLEX96_OUTPUT_ROOT
event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

channel_index = 0  # Zero-based column position, not the channel ID.
sequence_index = 0
number_of_samples = 1000

match event.lower():
    case "s5":
        original_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
    case "s59":
        original_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"

output_base = reconstructed_root / original_file.relative_to(data_root)
reconstructed_file = output_base.with_name(f"{output_base.stem}_reconstructed.ts")


# ------------------------------------------------------------------
# Read one matching record from each file
# ------------------------------------------------------------------
original_header = ts_read_header(original_file)
try:
    original = ts_read_sequence(original_header, sequence_index)
finally:
    original_header["fid"].close()

reconstructed_header = ts_read_header(reconstructed_file)
try:
    reconstructed = ts_read_sequence(reconstructed_header, sequence_index)
finally:
    reconstructed_header["fid"].close()


# ------------------------------------------------------------------
# Compare one channel: original line and reconstructed x markers
# ------------------------------------------------------------------
samples = np.arange(number_of_samples)
channel_id = original_header["channels"][channel_index]
fig, ax = plt.subplots()
ax.plot(samples, original[channel_index, :number_of_samples], "-", label="Original")
ax.plot(samples, reconstructed[channel_index, :number_of_samples], "x", markersize=4, label="Reconstructed")
ax.set_xlabel("Sample index within record")
ax.set_ylabel("Amplitude [input sample units]")
ax.set_title(f"Channel {channel_id:g}: original versus reconstructed")
ax.legend()
ax.grid(True)

plt.show()
