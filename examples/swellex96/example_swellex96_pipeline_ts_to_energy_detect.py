"""Convert Swellex96 channel timeseries to a band-energy bearing-time record.

First run example_swellex96_sio_to_ts.py to create the input .ts file.
Paths come from example_paths.py. Processing defaults come from
default_configs_swellex96.py. Existing output files are overwritten.
Optional tees save intermediate spectra while the pipeline continues.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import (DEFAULT_TS_INPUT,DEFAULT_TStoFFT,DEFAULT_FFTtoPSD,DEFAULT_PSDAvg5secs,
                                       DEFAULT_ENERGY_DETECT,DEFAULT_BGKD_EST,default_fft_to_cbf,
                                       default_fft_to_dmr,array_type,soundspeed_ftpers)
from wavekernelrt.pipeline.configs import OutputConfig, PipelineConfig, TeeConfig
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
save_intermediate_fft = False
save_intermediate_beams = True
background_est = False
rank_order = False

match event.lower():
    case "s5":
        input_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
    case "s59":
        input_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"

output_base = output_root / input_file.relative_to(data_root)

background = DEFAULT_BGKD_EST if background_est else None
output_type = "rank_uint8" if rank_order else "float32"
energy_detect_config = replace(DEFAULT_ENERGY_DETECT,background=background,output=output_type)
bkgd_txt = "bkgd" if background_est else "no_bkgd"
output_file = output_base.with_name(f"{output_base.stem}_{beamformer.lower()}_energy_detect_"
                                    f"{bkgd_txt}_{output_type}.btr")


# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = replace(DEFAULT_TS_INPUT, file=input_file)
geometry_file = data_root / "Arrays" / f"hla_{array.lower()}_geometry.txt"
geom = Geometry(geometry_file, array_type=array_type, soundspeed_ftpers=soundspeed_ftpers)

match beamformer.lower():
    case "cbf":
        beamforming_config = default_fft_to_cbf(geom)
    case "dmr":
        beamforming_config = default_fft_to_dmr(geom)
    case _:
        raise ValueError("beamformer must be 'cbf' or 'dmr'")
output_config = OutputConfig(file=output_file, type="btr")


# ------------------------------------------------------------------
# Assemble the pipeline: TS → FFT → beams → PSD → averaging → band energy
# ------------------------------------------------------------------
stages = [DEFAULT_TStoFFT]

if save_intermediate_fft:
    stages.append(TeeConfig(output=OutputConfig(file=output_base.with_suffix(".fft"), type="fft")))

stages.append(beamforming_config)

if save_intermediate_beams:
    stages.append(TeeConfig(output=OutputConfig(
        file=output_base.with_name(f"{output_base.stem}_{beamformer.lower()}.bfft"),type="bfft")))

stages.extend((DEFAULT_FFTtoPSD, DEFAULT_PSDAvg5secs, energy_detect_config))

run_config = PipelineConfig(input=input_config,stages=tuple(stages),output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"Energy Detect BTR file written to: {output_file}")
