"""Convert SWellEx-96 channel time series to beam-frequency records.

First run example_swellex96_sio_to_ts.py to create the input .ts file.
Paths come from example_paths.py. Processing defaults come from
default_configs_swellex96.py. Existing output files are overwritten.
An optional tee saves channel spectra while the pipeline continues. DMR runs
can independently save adaptive weights and achieved white-noise gain.
"""

# General imports
from dataclasses import replace

from example_paths import SWELLEX96_OUTPUT_ROOT
from default_configs_swellex96 import DEFAULT_TS_INPUT,DEFAULT_TStoFFT,default_fft_to_cbf,default_fft_to_dmr, array_type, soundspeed_ftpers
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

beamformer = "dmr"  # "cbf" or "dmr"
save_intermediate_fft = False

# DMR diagnostics are independent. Set each write flag to True or False. A
# band is an inclusive (low_hz, high_hz) tuple; None saves every frequency bin.
# A sequence stride of 10 saves source sequences 0, 10, 20, and so on.
match beamformer:
    case "dmr":
        write_dmr_weights = True
        dmr_weight_band_Hz = (90,115)
        dmr_weight_seq_stride = 10  # Writes sequences 0, 10, 20, ...
        write_dmr_wng = True
        dmr_wng_band_Hz = dmr_weight_band_Hz
        dmr_wng_seq_stride = dmr_weight_seq_stride  # Writes sequences 0, 10, 20, ...
    case _:
        write_dmr_weights = False
        write_dmr_wng = False

match event.lower():
    case "s5":
        input_file = data_root / "EventS5" / f"J1312340.hla.{array.lower()}.ts"
    case "s59":
        input_file = data_root / "EventS59" / f"J1341145.hla.{array.lower()}.ts"

output_base = output_root / input_file.relative_to(data_root)

output_file = output_base.with_name(f"{output_base.stem}_{beamformer.lower()}.bfft")

# ------------------------------------------------------------------
# Apply the defaults to this recording
# ------------------------------------------------------------------
input_config = replace(DEFAULT_TS_INPUT, file=input_file)
geometry_file = data_root / "Arrays" / f"hla_{array.lower()}_geometry.txt"
geom = Geometry(geometry_file, array_type=array_type, soundspeed_ftpers=soundspeed_ftpers)

match beamformer.lower():
    case "cbf":
        if write_dmr_weights or write_dmr_wng:
            raise ValueError("DMR weight/WNG outputs require beamformer='dmr'")
        
        beamforming_config = default_fft_to_cbf(geom)
    case "dmr":
        beamforming_config = replace(default_fft_to_dmr(geom),write_weights=write_dmr_weights,
                                     weight_band_Hz=dmr_weight_band_Hz,
                                     weight_seq_stride=dmr_weight_seq_stride,write_wng=write_dmr_wng,
                                     wng_band_Hz=dmr_wng_band_Hz,wng_seq_stride=dmr_wng_seq_stride)
    case _:
        raise ValueError("beamformer must be 'cbf' or 'dmr'")
    
output_config = OutputConfig(file=output_file, type="bfft")


# ------------------------------------------------------------------
# Assemble the pipeline: TS → FFT → beams
# ------------------------------------------------------------------
stages = [DEFAULT_TStoFFT]

if save_intermediate_fft:
    stages.append(TeeConfig(output=OutputConfig(file=output_base.with_suffix(".fft"),type="fft")))

stages.append(beamforming_config)

run_config = PipelineConfig(input=input_config,stages=tuple(stages),output=output_config)


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------
pipeline = build_pipeline(run_config)

run_pipeline(source=pipeline.source,stages=pipeline.stages,sink=pipeline.sink)

print(f"Beam FFT file written to: {output_file}")
if write_dmr_weights:
    weight_file = output_file.with_suffix(".bfw")
    print(f"DMR weight file written to: {weight_file}")
if write_dmr_wng:
    print(f"DMR WNG file written to: {output_file.with_suffix('.wng')}")
