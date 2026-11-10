"""Convert Swellex96 channel timeseries to beamformed timeseries.

First run example_swellex96_sio_to_ts.py to create the input .ts file.
Paths come from example_paths.py. Processing defaults come from
default_configs_swellex96.py. Existing output files are overwritten.
Select steering bearings below; optional channel cropping selects a subarray.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import DEFAULT_TS_INPUT, DEFAULT_TStoFFT, DEFAULT_FFTtoTS
from default_configs_swellex96 import default_fft_to_cbf,default_fft_to_dmr,array_type,soundspeed_ftpers, focus_range_ft

from wavekernelrt.pipeline.configs import OutputConfig, PipelineConfig, ChannelCropConfig

from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3
from wavekernelrt.core.geometry import Geometry
from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.run_pipeline import run_pipeline


# ------------------------------------------------------------------
# User settings — edit paths in example_paths.py
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
output_root = SWELLEX96_OUTPUT_ROOT

event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

beamformer = "cbf"  # "cbf" or "dmr"
bearings_deg = [90.0]  # Ascending bearings; 0° north, 90° east.
elevation_deg = 0.0

# None uses all channels. To use a subarray, supply ascending, zero-based
# input-column indices, e.g. list(range(14)) for the first 14 channels.
# These are positions in the input data, not hydrophone channel labels.
channel_indices = None

match event.lower():
    case "s5":
        input_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
    case "s59":
        input_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"

output_base = output_root / input_file.relative_to(data_root)
output_file = output_base.with_name(f"{output_base.stem}_{beamformer.lower()}_beams.bts")


# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = replace(DEFAULT_TS_INPUT, file=input_file)
# Supply the full input geometry. The builder selects the cropped geometry
# and recomputes its phase center and shading when channel cropping is used.
geometry_file = data_root / "Arrays" / f"hla_{array.lower()}_geometry.txt"
geom = Geometry(geometry_file, array_type=array_type, soundspeed_ftpers=soundspeed_ftpers)

match beamformer.lower():
    case "cbf":
        beamforming_config = default_fft_to_cbf(geom)
    case "dmr":
        beamforming_config = default_fft_to_dmr(geom)
    case _:
        raise ValueError("beamformer must be 'cbf' or 'dmr'")

beamforming_config = replace(beamforming_config,steer_rbe_rad_J3=construct_rbe_X3(focus_range_ft,bearings_deg,elevation_deg))

output_config = OutputConfig(file=output_file, type="bts")


# ------------------------------------------------------------------
# Assemble the pipeline: TS → optional channel crop → FFT → beams → BTS
# ------------------------------------------------------------------
stages = []
if channel_indices is not None:
    stages.append(ChannelCropConfig(idx_channels_to_keep=channel_indices))

stages.extend((DEFAULT_TStoFFT, beamforming_config, DEFAULT_FFTtoTS))

run_config = PipelineConfig(input=input_config,stages=tuple(stages),output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"BTS file written to: {output_file}")
