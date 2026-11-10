"""Convert a Swellex96 beam spectrum to a band-energy bearing-time record.

First create a .bfft file, for example by running the TS-to-Energy-Detect
example with save_intermediate_beams enabled. Paths come from example_paths.py.
Processing defaults come from default_configs_swellex96.py. Existing output
files are overwritten.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import (DEFAULT_BGKD_EST,DEFAULT_ENERGY_DETECT,DEFAULT_FFTtoPSD,
                                       DEFAULT_PSDAvg5secs)

from wavekernelrt.pipeline.configs import InputConfig, OutputConfig, PipelineConfig
from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.run_pipeline import run_pipeline


# ------------------------------------------------------------------
# User settings — edit paths in example_paths.py
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
output_root = SWELLEX96_OUTPUT_ROOT

event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

beamformer = "dmr"  # "cbf" or "dmr"
background_est = False
rank_order = False

match event.lower():
    case "s5":
        input_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}_{beamformer.lower()}.bfft"
    case "s59":
        input_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}_{beamformer.lower()}.bfft"

output_base = output_root / input_file.relative_to(data_root)

background = DEFAULT_BGKD_EST if background_est else None
output_type = "rank_uint8" if rank_order else "float32"
energy_detect_config = replace(DEFAULT_ENERGY_DETECT,background=background,output=output_type)
bkgd_txt = "bkgd" if background_est else "no_bkgd"
output_file = output_base.with_name(f"{output_base.stem}_energy_detect_{bkgd_txt}_{output_type}.btr")


# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = InputConfig(file=input_file, type="bfft")
output_config = OutputConfig(file=output_file, type="btr")

print(input_file)

# ------------------------------------------------------------------
# Assemble the pipeline: beam FFT → PSD → time average → band energy in dB
# ------------------------------------------------------------------
stages = [DEFAULT_FFTtoPSD, DEFAULT_PSDAvg5secs, energy_detect_config]

run_config = PipelineConfig(input=input_config,stages=tuple(stages),output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"Energy Detect BTR file written to: {output_file}")
