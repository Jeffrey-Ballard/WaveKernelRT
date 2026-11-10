"""Compare one stored DMR beampattern with a Taylor-shaded CBF.

The DMR multiplication weights are held fixed while a simulated source is
swept through relative bearing.  This fixed-weight look sweep is the natural
beampattern for adaptive weights.  The requested frequency and steering
direction select the nearest values stored in the BFW file.
"""

# General imports
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from default_configs_swellex96 import array_type, soundspeed_ftpers
from example_paths import SWELLEX96_OUTPUT_ROOT
from wavekernelrt.core.beampattern import BeamPattern
from wavekernelrt.core.bf_weight_file_utils import bf_weight_read_header,bf_weight_read_single_sequence
from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.general_utils import old_man_eyes
from wavekernelrt.core.shading import Shading
from wavekernelrt.core.steeringvectors import SteeringVectors
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3


# ------------------------------------------------------------------
# User settings — edit these selections
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
dmr_weight_file = data_root / "EventS5" / "J1312340.hla.south_dmr.bfw"

sequence = 100  # Zero-based stored BFW record (not the original DMR sequence).
look_bearing_deg = 250.0
look_elevation_deg = 0.0
frequency_hz = 112.0

# Optional known source bearing for illustrating adaptive null placement.
# Set to a relative bearing in degrees to display it, or None to hide it.
source_bearing_deg = 228

# Display settings
sweep_bearing_deg_A = np.linspace(0.0, 360.0, 721)
minimum_response_dB = -40.0


# ------------------------------------------------------------------
# Resolve the SWellEx-96 geometry from the weight-file name
# ------------------------------------------------------------------
filename_lower = dmr_weight_file.name.lower()
matching_arrays = [name for name in ("north","south") if f".{name}" in filename_lower]
if len(matching_arrays) != 1:
    raise ValueError(
        "The DMR weight filename must contain exactly one of '.north' or "
        "'.south' so its SWellEx-96 geometry can be selected")

array = matching_arrays[0]
geometry_file = data_root / "Arrays" / f"hla_{array}_geometry.txt"
geom = Geometry(geometry_file,array_type=array_type,soundspeed_ftpers=soundspeed_ftpers)


# ------------------------------------------------------------------
# Read the header and one stored weight record
# ------------------------------------------------------------------
header = bf_weight_read_header(dmr_weight_file)
try:
    weights_QJN, array_heading_deg = bf_weight_read_single_sequence(header,sequence)
finally:
    header["fid"].close()

if geom.xyz_m_N3.shape[0] != header["Nchan"]:
    raise ValueError(
        f"Geometry has {geom.xyz_m_N3.shape[0]} elements, but the BFW file "
        f"contains {header['Nchan']} weights per beam")

# Select the nearest stored frequency.
frequency_index = int(np.argmin(np.abs(header["freq"] - frequency_hz)))
selected_frequency_hz = float(header["freq"][frequency_index])

# Select the nearest stored steering direction. Bearing differences wrap at
# north so, for example, 359 degrees is close to 1 degree.
steer_J3 = header["steer_rbe_rad_J3"]
steer_bearing_deg_J = np.rad2deg(steer_J3[:, 1])
steer_elevation_deg_J = np.rad2deg(steer_J3[:, 2])
bearing_error_deg_J = np.abs((steer_bearing_deg_J - look_bearing_deg + 180.0) % 360.0 - 180.0)
elevation_error_deg_J = np.abs(steer_elevation_deg_J - look_elevation_deg)
beam_index = int(np.argmin(bearing_error_deg_J**2 + elevation_error_deg_J**2))
selected_steer_3 = steer_J3[beam_index]
selected_bearing_deg = float(steer_bearing_deg_J[beam_index])
selected_elevation_deg = float(steer_elevation_deg_J[beam_index])
selected_range_ft = float(selected_steer_3[0] * 3.2808)

relative_time_seconds = sequence * float(header["secs_per_seq"])
timestamp_seconds = float(header["start_unix_seconds"]) + relative_time_seconds
timestamp_utc = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)

print(f"File: {dmr_weight_file}")
print(f"Stored records: {header['Nseq']}; selected record: {sequence}")
print(f"Stored bearing frame: {header['bearing_frame']}")
print(f"Array heading: {array_heading_deg:g} degrees true")
print(f"Record time: {relative_time_seconds:g} seconds from file start ({timestamp_utc.isoformat()})")
print(f"Frequency: requested {frequency_hz:g} Hz; selected {selected_frequency_hz:g} Hz "
      f"(index {frequency_index})")
print(
    "Steering direction: requested "
    f"({look_bearing_deg:g}, {look_elevation_deg:g}) deg; selected "
    f"({selected_bearing_deg:g}, {selected_elevation_deg:g}) deg (beam index {beam_index})")


# ------------------------------------------------------------------
# Form the stored DMR and default Taylor-shaded CBF weights
# ------------------------------------------------------------------
# BFW records contain already-conjugated multiplication coefficients, which
# is exactly the convention expected by BeamPattern.
dmr_weight_1N = weights_QJN[frequency_index,beam_index,:][None,:]

selected_replica_1N = SteeringVectors(geom,selected_steer_3[None,:],[selected_frequency_hz],
                                      precision="float32",flag_relative=True).complex_phasors_QNJ[0,:,0][None,:]

# Shading defaults are Taylor 30 dB, nbar=4. Use the per-beam method so this
# example also remains valid if it is adapted to a curved-array geometry.
taylor_shading_1N = Shading(geom).taylor_shading_for_beams(selected_steer_3[None,:],flag_relative=True)
cbf_weight_1N = taylor_shading_1N * np.conj(selected_replica_1N)
cbf_weight_1N /= np.sum(cbf_weight_1N * selected_replica_1N,axis=1,keepdims=True)


# ------------------------------------------------------------------
# Hold each weight vector fixed and sweep the source bearing
# ------------------------------------------------------------------
arrival_rbe_rad_A3 = construct_rbe_X3(selected_range_ft,sweep_bearing_deg_A,selected_elevation_deg)
beampattern = BeamPattern(geom=geom,freq=[selected_frequency_hz],Nbearings=1,Nelevations=1,
                          focus_range_ft=selected_range_ft,flag_relative=True)

dmr_response_A = beampattern.compute_beampattern_look_sweep_JQ(dmr_weight_1N,arrival_rbe_rad_A3)[:,0]
cbf_response_A = beampattern.compute_beampattern_look_sweep_JQ(cbf_weight_1N,arrival_rbe_rad_A3)[:,0]

dmr_response_dB_A = 20.0 * np.log10(np.maximum(dmr_response_A, 1e-12))
cbf_response_dB_A = 20.0 * np.log10(np.maximum(cbf_response_A, 1e-12))


# ------------------------------------------------------------------
# Plot both fixed-weight look sweeps on the same axes
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
ax.plot(sweep_bearing_deg_A,dmr_response_dB_A,label="DMR (stored adaptive weights)",linewidth=1.5)
ax.plot(sweep_bearing_deg_A,cbf_response_dB_A,label="CBF (Taylor 30 dB, nbar=4)",linewidth=1.5)
ax.axvline(selected_bearing_deg,color="black",linestyle="-.",linewidth=2.0,zorder=5,
           label="DMR look direction")

if source_bearing_deg is not None:
    ax.axvline(source_bearing_deg,color="tab:red",linestyle="--",linewidth=1.5,label="Source direction")

ax.set_xlim(float(sweep_bearing_deg_A[0]), float(sweep_bearing_deg_A[-1]))
ax.set_ylim(minimum_response_dB, 3.0)
ax.set_xticks(np.arange(0.0, 361.0, 30.0))
ax.set_xlabel("Relative bearing (deg)")
ax.set_ylabel("Response (dB re unity)")
ax.set_title(
    f"DMR and CBF fixed-weight look sweeps at {selected_frequency_hz:g} Hz\n"
    f"BFW record {sequence}, {relative_time_seconds:g} seconds from file start, "
    f"steering bearing {selected_bearing_deg:.1f} deg")
ax.grid(True)
old_man_eyes(ax)
ax.legend()

# fig.savefig(dmr_weight_file.with_suffix(".jpeg"))

# Display the figure. Use fig.savefig(...) to save it.
plt.show()
