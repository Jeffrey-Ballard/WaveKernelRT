"""Recover channel timeseries from saved Swellex96 FFT spectra.

First run example_swellex96_pipeline_ts_to_fft.py to create the input .fft file.
Paths come from example_paths.py. The original converted .ts file is used to
match the reconstructed output record length.
Recovery is within floating-point precision; incomplete trailing data is dropped.
Processing defaults come from default_configs_swellex96.py.
Existing output files are overwritten.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import DEFAULT_FFTtoTS
from wavekernelrt.pipeline.configs import InputConfig, OutputConfig, PipelineConfig
from wavekernelrt.core.ts_file_utils import ts_read_header

from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.run_pipeline import run_pipeline


# ------------------------------------------------------------------
# User settings — edit paths in example_paths.py
# ------------------------------------------------------------------
data_root = SWELLEX96_OUTPUT_ROOT
output_root = SWELLEX96_OUTPUT_ROOT
original_root = SWELLEX96_OUTPUT_ROOT

event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"

match event.lower():
    case "s5":
        input_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.fft"
    case "s59":
        input_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.fft"

output_base = output_root / input_file.relative_to(data_root)
# Keep the original .ts file intact, even when both roots are the same.
output_file = output_base.with_name(f"{output_base.stem}_reconstructed.ts")


# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = InputConfig(file=input_file, type="fft")
# Inherit FFT settings, but match the original TS record length for comparison.
original_file = (original_root / input_file.relative_to(data_root)).with_suffix(".ts")
original_header = ts_read_header(original_file)
try:
    fft_to_ts_config = replace(DEFAULT_FFTtoTS, Npt_per_seq=original_header["Npt_per_seq"])
finally:
    original_header["fid"].close()
output_config = OutputConfig(file=output_file, type="ts")


# ------------------------------------------------------------------
# Assemble the pipeline: FFT → TS
# ------------------------------------------------------------------
run_config = PipelineConfig(input=input_config,stages=(fft_to_ts_config,),output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"Reconstructed TS file written to: {output_file}")
