"""Inspect achieved DMR white-noise gain and its effective constraint.

Run example_swellex96_pipeline_ts_to_bf.py with DMR WNG output enabled first.
The three figures show one absolute frequency-by-bearing WNG snapshot, a
constraint-relative snapshot clipped to +/-10 dB, and the WNG time history for
one selected frequency and beam. Selections use stored WNG record indices; a
WNG sequence stride means these need not equal source FFT indices.
"""

# General imports
import matplotlib.pyplot as plt
import numpy as np

from example_paths import SWELLEX96_OUTPUT_ROOT
from wavekernelrt.core import analysis_images
from wavekernelrt.core.wng_file_utils import (wng_constraint_linear_J,wng_read_header,
                                              wng_read_multiple_sequence)


# ------------------------------------------------------------------
# User settings — edit these selections
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

snapshot_sequence = 100  # Zero-based stored WNG record.
first_sequence = 0
number_of_sequences = None  # None loads through the final stored record.

look_bearing_deg = 250.0
frequency_hz = 112.0

# Display limits are in dB because wng_disp="dB" below. Leave as None for
# automatic limits shared by the ordinary and constraint-highlighted images.
wng_limits_dB = (None, None)

match event.lower():
    case "s5":
        input_base = data_root / "EventS5" / f"J1312340.hla.{array.lower()}"
    case "s59":
        input_base = data_root / "EventS59" / f"J1341145.hla.{array.lower()}"
    case _:
        raise ValueError("event must be 'S5' or 'S59'")

input_file = input_base.with_name(f"{input_base.name}_dmr.wng")


# ------------------------------------------------------------------
# Read the header and requested WNG records
# ------------------------------------------------------------------
header = wng_read_header(input_file)
try:
    if number_of_sequences is None:
        stop_sequence = header["Nseq"]
    else:
        stop_sequence = min(first_sequence + number_of_sequences,header["Nseq"])
    sequences = np.arange(first_sequence, stop_sequence)
    if sequences.size == 0:
        raise ValueError("No records selected; check first_sequence and the file's Nseq")
    if snapshot_sequence not in sequences:
        raise ValueError("snapshot_sequence must be among the records selected for loading")

    wng_QJT, headings_T = wng_read_multiple_sequence(header, sequences)
    constraint_wng_J = wng_constraint_linear_J(header)
finally:
    header["fid"].close()

snapshot_index = int(np.flatnonzero(sequences == snapshot_sequence)[0])
times_seconds_T = sequences * float(header["secs_per_seq"])
beam_bearings_deg_J = np.rad2deg(header["steer_rbe_rad_J3"][:, 1])

# Select the nearest stored frequency and bearing. Bearing distance wraps at
# north, so 359 degrees is close to 1 degree.
frequency_index = int(np.argmin(np.abs(header["freq"] - frequency_hz)))
bearing_error_deg_J = np.abs((beam_bearings_deg_J - look_bearing_deg + 180.0) % 360.0 - 180.0)
beam_index = int(np.argmin(bearing_error_deg_J))
selected_frequency_hz = float(header["freq"][frequency_index])
selected_bearing_deg = float(beam_bearings_deg_J[beam_index])

effective_constraint_dB_J = 10.0 * np.log10(constraint_wng_J)

print(f"File: {input_file}")
print(f"Stored records: {header['Nseq']}; loaded {sequences[0]} through {sequences[-1]}")
print(f"WNG request: {header['wngc_val_dB']:g} dB relative to unshaded CBF; "
      f"tolerance: {header['wngc_tol_dB']:g} dB")
print(f"Snapshot record: {snapshot_sequence}; array heading: {headings_T[snapshot_index]:g} degrees true")
print(f"Frequency: requested {frequency_hz:g} Hz; selected {selected_frequency_hz:g} Hz "
      f"(index {frequency_index})")
print(f"Bearing: requested {look_bearing_deg:g} deg; selected {selected_bearing_deg:g} deg "
      f"(beam index {beam_index})")
print(f"Effective absolute constraint for selected beam: {effective_constraint_dB_J[beam_index]:.2f} dB")


# ------------------------------------------------------------------
# Ordinary WNG snapshot: frequency vertically, relative bearing horizontally
# ------------------------------------------------------------------
vmin, vmax = wng_limits_dB
snapshot_title = (f"DMR WNG record {snapshot_sequence}, "
                  f"{times_seconds_T[snapshot_index]:g} seconds from file start")

fig1, ax, mesh = analysis_images.wng_image_QJ(wng_QJT,freq=header["freq"],beams=beam_bearings_deg_J,
    idxT=snapshot_index,title=snapshot_title,wng_disp="dB",vmin=vmin,vmax=vmax)


# ------------------------------------------------------------------
# Constraint-relative snapshot
# Negative values pass and run from black (-10 dB) to white (0 dB). Positive
# values fail and use the color scale from 0 to +10 dB. Values are clipped at
# both endpoints.
# ------------------------------------------------------------------
fig2, ax, mesh = analysis_images.wng_image_QJ(wng_QJT,freq=header["freq"],beams=beam_bearings_deg_J,
    idxT=snapshot_index,constraint_wng=constraint_wng_J,title=f"{snapshot_title}\nConstraint diagnostic",
    wng_disp="dB")


# ------------------------------------------------------------------
# WNG history for one frequency and beam, with constraint reference
# ------------------------------------------------------------------
fig3, ax, line = analysis_images.wng_line_T(wng_QJT,idxQ=frequency_index,idxJ=beam_index,
    times=times_seconds_T,freq=header["freq"],beams=beam_bearings_deg_J,constraint_wng=constraint_wng_J,
    title="DMR WNG time history",wng_disp="dB",time_label="Seconds from first WNG record")


# Display all figures. Use fig.savefig(...) to save a particular figure.
plt.show()
