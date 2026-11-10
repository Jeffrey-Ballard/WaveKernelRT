"""Convert Swellex96 channel timeseries to FFT spectra for round-trip recovery.

First run example_swellex96_sio_to_ts.py to create the input .ts file.
Paths come from example_paths.py. Processing defaults come from
default_configs_swellex96.py.
This example uses a rectangular window to preserve even the first sample.
Unlike a Tukey taper, it does not suppress spectral leakage at frame edges.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import DEFAULT_TS_INPUT, DEFAULT_TStoFFT

from wavekernelrt.pipeline.configs import OutputConfig, PipelineConfig
from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.run_pipeline import run_pipeline


# ------------------------------------------------------------------
# User settings — edit paths in example_paths.py
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
output_root = SWELLEX96_OUTPUT_ROOT

event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

match event.lower():
    case "s5":
        filename_ts = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
    case "s59":
        filename_ts = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"

input_file = filename_ts
output_file = (output_root / input_file.relative_to(data_root)).with_suffix(".fft")


# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = replace(DEFAULT_TS_INPUT, file=input_file)
# No taper: avoid losing the first sample to a zero-valued window endpoint.
ts_to_fft_config = replace(DEFAULT_TStoFFT, window_name="boxcar")

output_config = OutputConfig(file=output_file,type="fft")


# ------------------------------------------------------------------
# Assemble the pipeline: TS → FFT
# ------------------------------------------------------------------
run_config = PipelineConfig(input=input_config,stages=(ts_to_fft_config,),output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"FFT file written to: {output_file}")
