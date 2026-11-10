"""Inspect channel FFT amplitude and phase.

The input root comes from example_paths.py.
Selections below are zero-based array positions, not channel IDs or bearings.
Only the requested records are loaded. Figures are displayed, not saved.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt
from example_paths import SWELLEX96_OUTPUT_ROOT
from wavekernelrt.core import analysis_images
from wavekernelrt.core.fft_file_utils import fft_read_header, fft_read_multiple_sequence


# ------------------------------------------------------------------
# User settings — edit these selections
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

first_sequence = 0
number_of_sequences = 120
channel_index = 0
frequency_limits_hz = (0.0, 1000.0)  # Display limits only; data are not cropped.

match event.lower():
    case "s5":
        input_base = data_root / "EventS5" / f"J1312340.hla.{array.lower()}"
    case "s59":
        input_base = data_root / "EventS59" / f"J1341145.hla.{array.lower()}"

input_file = input_base.with_name(f"{input_base.name}.fft")


# ------------------------------------------------------------------
# Read the header and selected records
# ------------------------------------------------------------------
header = fft_read_header(input_file)
try:
    stop_sequence = min(first_sequence + number_of_sequences, header["Nseq"])
    sequences = np.arange(first_sequence, stop_sequence)
    if sequences.size == 0:
        raise ValueError("No records selected; check first_sequence and the file's Nseq")
    data_TQN = fft_read_multiple_sequence(header, sequences)
finally:
    header["fid"].close()

print(f"File: {input_file}")
print(f"Records: {header['Nseq']}; loaded {sequences[0]} through {sequences[-1]}")
print(f"Channel IDs: {header['channels']}")
print(f"Frequency bins: {header['freq'][0]:g} to {header['freq'][-1]:g} Hz")

# Times are relative to the first spectrum timestamp in the file.
times_secs = sequences * header["secs_per_seq"]
channel_id = header["channels"][channel_index]


# ------------------------------------------------------------------
# Frequency versus time for one channel
# ------------------------------------------------------------------
# Inputs remain complex. The library displays amplitude in dB and phase in radians.
fig, axes, meshes = analysis_images.fft_spectrum_image_TQ(data_TQN,plot_type="complex",freq=header["freq"],
    times=times_secs,idxN=channel_index,amp_disp="dB",phs_disp="unwrapped",
    time_label="Seconds from first spectrum",title=f"Channel {channel_id:g}")

# ------------------------------------------------------------------
# Frequency versus channel, averaged over loaded records
# ------------------------------------------------------------------
# This averages linear amplitudes, not complex spectra or PSD power.
fig, ax, mesh = analysis_images.fft_spectrum_image_QN(data_TQN,plot_type="amplitude",freq=header["freq"],
    channels=header["channels"],collapse_func="mean",amp_disp="dB",title="Mean FFT amplitude")

# Display all figures. Use fig.savefig(...) to save a particular figure.
plt.show()
