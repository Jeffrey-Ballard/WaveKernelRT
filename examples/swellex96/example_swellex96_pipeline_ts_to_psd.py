"""Convert Swellex96 channel timeseries to averaged PSD spectra.

First run example_swellex96_sio_to_ts.py to create the input .ts file.
Paths come from example_paths.py. Processing defaults come from
default_configs_swellex96.py. Existing output files are overwritten.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import DEFAULT_TS_INPUT,DEFAULT_TStoFFT,DEFAULT_FFTtoPSD,DEFAULT_PSDAvg5secs

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
        input_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
    case "s59":
        input_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"

output_base = output_root / input_file.relative_to(data_root)
output_file = output_base.with_suffix(".psd")


# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = replace(DEFAULT_TS_INPUT, file=input_file)
ts_to_fft_config = DEFAULT_TStoFFT
fft_to_psd_config = DEFAULT_FFTtoPSD
psd_avg_config = DEFAULT_PSDAvg5secs
output_config = OutputConfig(file=output_file, type="psd")


# ------------------------------------------------------------------
# Assemble the pipeline: TS → FFT → PSD → averaging
# ------------------------------------------------------------------
run_config = PipelineConfig(input=input_config,stages=(ts_to_fft_config,fft_to_psd_config,psd_avg_config),
                            output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"PSD file written to: {output_file}")
