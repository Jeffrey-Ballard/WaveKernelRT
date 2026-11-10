"""Inspect band-energy bearing-time records from an Energy Detect pipeline.

The input root comes from example_paths.py.
Selections below are zero-based array positions, not channel IDs or bearings.
Only the requested records are loaded. The figure is saved and displayed.
See energy_detect_visualization.md for display sizing and export behavior.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from example_paths import SWELLEX96_OUTPUT_ROOT
from wavekernelrt.core import analysis_images
from wavekernelrt.core.btr_file_utils import btr_read_header, btr_read_sequence


# ------------------------------------------------------------------
# User settings — edit these selections
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

beamformer = "dmr"  # Match the pipeline: "cbf" or "dmr".
background_est = True
rank_order = True

first_sequence = 0
num_seqs = -1  # -1 reads all remaining records from first_sequence.
band_index = 0
elevation_index = 0
bearing_display = "geographic"  # "relative" or "geographic"

# Color limits refer to the stored values: ranks for rank_uint8 output and dB
# for float32 energy-detect output. None autoscales.

pixels_per_sample = 1.0  # Logical pixels: 2.0 enlarges; 0.5 shrinks.

# Bearing ticks adapt to the rendered data-area width. The viewer chooses the
# finest interval below that keeps adjacent ticks at least this far apart.
bearing_tick_intervals_deg = (15, 30, 45, 90)
minimum_bearing_tick_spacing_px = 45

match event.lower():
    case "s5":
        input_base = data_root / "EventS5" / f"J1312340.hla.{array.lower()}"
    case "s59":
        input_base = data_root / "EventS59" / f"J1341145.hla.{array.lower()}"

cmap = "viridis"  # For example: "gray", "gray_r", "jet", or "viridis".
vmin = 0
vmax = None
colorbar = False

if background_est:
    bkgd_txt = "bkgd"
    bkgd_title = "Bkgd Normalized"
else:
    bkgd_txt = "no_bkgd"
    bkgd_title = "No Bkgd Norm"

if rank_order:
    rnk_txt = "rank_uint8"
    rnk_title = "Rank Ordered"
else:
    rnk_txt = "float32"
    rnk_title = "Band Statistic [dB]"
    cmap = "jet"  # For example: "gray", "gray_r", "jet", or "viridis".
    if background_est:
        vmin = 15
        vmax = 35
    else:
        vmin = 10
        vmax = 35

    colorbar = True

input_file = input_base.with_name(f"{input_base.name}_{beamformer.lower()}_energy_detect_{bkgd_txt}_{rnk_txt}.btr")


# ------------------------------------------------------------------
# Read the header and selected records
# ------------------------------------------------------------------
header = btr_read_header(input_file)
try:
    if num_seqs == -1:
        stop_sequence = header["Nseq"]
    elif num_seqs > 0:
        stop_sequence = min(first_sequence + num_seqs, header["Nseq"])
    else:
        raise ValueError("num_seqs must be -1 or a positive record count")
    sequences = np.arange(first_sequence, stop_sequence)
    if sequences.size == 0:
        raise ValueError("No records selected; check first_sequence and the file's Nseq")
    data_EBAT, array_headings_deg_T = btr_read_sequence(header,sequences,idx_elv=elevation_index)
finally:
    header["fid"].close()

print(f"File: {input_file}")
print(f"Records: {header['Nseq']}; loaded {sequences[0]} through {sequences[-1]}")
print(f"Frequency bands [Hz]: {header['bands_B2']}")
print(f"Elevations [degrees]: {header['elev_deg_E']}")
print(f"Stored bearing frame: {header['bearing_frame']}")
print(f"Array headings [degrees true]: {array_headings_deg_T[0]:g} through {array_headings_deg_T[-1]:g}")

# E = selected elevation, B = band, A = bearing, T = record.
# E index 0 refers to the one elevation requested above.
data_TA = data_EBAT[0, band_index, :, :].T
times_secs = sequences * header["secs_per_seq"]
band = header["bands_B2"][band_index]
elevation_deg = header["elev_deg_E"][elevation_index]


# ------------------------------------------------------------------
# Select the display bearing frame
# ------------------------------------------------------------------
relative_bearings_deg_A = np.asarray(header["azim_deg_A"],dtype=np.float64)

match bearing_display.lower():
    case "relative":
        display_data_TA = data_TA
        display_bearings_deg_A = relative_bearings_deg_A
        bearing_label = "Relative bearing [degrees]"
        bearing_tick_formatter = None

    case "geographic" | "geo":
        # A near-full-circle product is remapped onto one common geographic
        # grid. Nearest-neighbor display remapping preserves stored rank and
        # dB values rather than interpolating them.
        if np.ptp(relative_bearings_deg_A) >= 300.0:
            display_bearings_deg_A = np.linspace(0.0,360.0,relative_bearings_deg_A.size,endpoint=False)
            display_data_TA = np.empty_like(data_TA)
            for index_t, heading_deg in enumerate(array_headings_deg_T):
                geographic_deg_A = (relative_bearings_deg_A + heading_deg) % 360.0
                circular_error_GA = np.abs(
                    (display_bearings_deg_A[:,None] - geographic_deg_A[None,:] + 180.0) % 360.0 - 180.0)
                nearest_A = np.argmin(circular_error_GA,axis=1)
                display_data_TA[index_t] = data_TA[index_t, nearest_A]
        else:
            # Preserve a partial sector's adjacency when it or the changing
            # array heading crosses true north. Coordinates remain monotonic;
            # tick labels wrap modulo 360 below.
            relative_unwrapped_deg_A = np.rad2deg(np.unwrap(np.deg2rad(relative_bearings_deg_A)))
            heading_unwrapped_deg_T = np.rad2deg(np.unwrap(np.deg2rad(array_headings_deg_T)))
            geographic_unwrapped_TA = relative_unwrapped_deg_A[None,:] + heading_unwrapped_deg_T[:,None]
            bearing_spacing_deg = float(np.median(np.diff(relative_unwrapped_deg_A)))
            display_bearings_deg_A = np.arange(np.min(geographic_unwrapped_TA),
                np.max(geographic_unwrapped_TA) + bearing_spacing_deg / 2,bearing_spacing_deg)
            display_data_TA = np.full((data_TA.shape[0],display_bearings_deg_A.size),np.nan,dtype=np.float64)
            
            for index_t, geographic_deg_A in enumerate(geographic_unwrapped_TA):
                in_sector_G = ((display_bearings_deg_A >= geographic_deg_A[0] - bearing_spacing_deg / 2)
                    & (display_bearings_deg_A <= geographic_deg_A[-1] + bearing_spacing_deg / 2))
                nearest_GA = np.abs(display_bearings_deg_A[in_sector_G,None] - geographic_deg_A[None,:])
                nearest_A = np.argmin(nearest_GA,axis=1)
                display_data_TA[index_t,in_sector_G] = data_TA[index_t,nearest_A]

        bearing_label = "Geographic bearing [degrees true]"
        bearing_tick_formatter = FuncFormatter(lambda value,position: f"{value % 360:g}")

    case _:
        raise ValueError("bearing_display must be 'relative' or 'geographic'")


# ------------------------------------------------------------------
# Time versus bearing for one frequency band
# ------------------------------------------------------------------
# Display the stored rank or dB values directly.
# Explicit time edges preserve the record duration even for a single record.
time_edges = np.append(times_secs, times_secs[-1] + header["secs_per_seq"])
time_edges -= header["secs_per_seq"] / 2

# btr_image sizes the data area in logical pixels; a colorbar adds a margin.
# Avoid resizing or tight_layout if you want to preserve that pixel mapping.
fig, ax, mesh = analysis_images.btr_image(display_data_TA,row_data=time_edges,col_data=display_bearings_deg_A,
    row_label="Seconds from first record",col_label=bearing_label,cmap=cmap,vmin=vmin,vmax=vmax,
    pixels_per_sample=pixels_per_sample,colorbar=colorbar,bearing_tick_intervals_deg=bearing_tick_intervals_deg,
    minimum_bearing_tick_spacing_px=minimum_bearing_tick_spacing_px)

if colorbar:
    if background_est:
        mesh.colorbar.set_label("Normalized band statistic [dB re 1 Hz]")
    else:
        mesh.colorbar.set_label("Band power [dB re 1 power unit]")

ax.set_title(
    f"Energy Detect, {beamformer.upper()} {band[0]:g}–{band[1]:g} Hz\n"
    f"Elevation {elevation_deg:g}°\n"
    f"{bkgd_title}, {rnk_title}\n"
    f"Most Recent Array HDG {array_headings_deg_T[-1]:.1f}°",
    fontweight="bold")

if bearing_tick_formatter is not None:
    ax.xaxis.set_major_formatter(bearing_tick_formatter)

fig.canvas.draw()

print("Data shape:", display_data_TA.shape)
print("Figure DPI:", fig.dpi)
print("Display pixel ratio:", fig.canvas.device_pixel_ratio)
print("Canvas logical pixels:", fig.canvas.get_width_height())
print("Canvas physical pixels:", fig.canvas.get_width_height(physical=True))

bounds = ax.get_window_extent()
print("Data area rendering pixels:", bounds.width, bounds.height)

# fig.savefig(input_file.with_suffix(".jpeg"))

# Display all figures. Use fig.savefig(...) to save a particular figure.
plt.show()
