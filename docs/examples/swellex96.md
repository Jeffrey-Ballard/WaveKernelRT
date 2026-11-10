# SWellEx-96 workflow

These examples apply WaveKernelRT to recordings from the public SWellEx-96
shallow-water acoustics experiment. The dataset is not distributed with this
repository.

## Download the data

Visit the [SWellEx-96 website](https://swellex96.ucsd.edu/index.htm):

1. Under **Arrays**, download the horizontal-line-array geometry files for the
   north and/or south array.
2. Under **Events**, download the horizontal-line-array `.sio` recording for
   Event S5 and/or Event S59.

Only the geometry and recording for the array/event combination you intend to
run are required. Preserve the following names and capitalization because the
conversion example resolves them directly:

```text
swellex96/
|-- Arrays/
|   |-- positions_hlanorth_raw.txt
|   `-- positions_hlasouth_raw.txt
|-- EventS5/
|   |-- J1312340.hla.north.sio
|   `-- J1312340.hla.south.sio
`-- EventS59/
    |-- J1341145.hla.north.sio
    `-- J1341145.hla.south.sio
```

## Configure local paths

Open `examples/swellex96/example_paths.py` and set the two paths:

```python
from pathlib import Path

SWELLEX96_DATA_ROOT = Path("/path/to/downloaded/swellex96")
SWELLEX96_OUTPUT_ROOT = Path("/path/to/processed/swellex96")
```

`SWELLEX96_DATA_ROOT` contains the downloaded `Arrays`, `EventS5`, and/or
`EventS59` folders. `SWELLEX96_OUTPUT_ROOT` receives the converted geometry,
converted recordings, and derived products. The two values may be the same if
you prefer to keep input and output files together.

The conversion script creates the required output directories. All later
pipeline and visualization scripts read from the output root, so these paths
only need to be configured once.

## Select an event and array

Each script has a short user-settings section containing:

```python
event = "S5"       # "S5" or "S59"
array = "south"    # "north" or "south"
```

Use the same selections throughout one workflow. Beamforming examples also let
you select `cbf` or `dmr`. Processing parameters shared by the examples are in
`examples/swellex96/default_configs_swellex96.py`.

## Step 1: convert the downloaded files

```bash
python examples/swellex96/example_swellex96_sio_to_ts.py
```

This converts the downloaded position file from north/east/depth meters into
WaveKernelRT's east/north/up geometry convention in inches. It also converts one
native `.sio` recording into a WaveKernelRT `.ts` channel-timeseries file.

Outputs for the default S5/south selection are:

```text
Arrays/hla_south_geometry.txt
EventS5/J1312340.hla.south.ts
```

Existing files with those names are overwritten.

## Processing workflows

The remaining pipelines all start from the converted `.ts` file. Full-recording
runtime and output size depend on the selected event, array, computer, and
processing settings. DMR beamforming and the full TS-to-Energy-Detect chain are the most
computationally demanding examples and may take substantially longer than the
basic-array examples.

### FFT round trip

Create saved channel spectra:

```bash
python examples/swellex96/example_swellex96_pipeline_ts_to_fft.py
```

Output: a matching `.fft` file. Inspect it with:

```bash
python examples/swellex96/example_swellex96_visualization_fft.py
```

Reconstruct channel timeseries from the spectra:

```bash
python examples/swellex96/example_swellex96_pipeline_fft_to_ts.py
```

Output: `*_reconstructed.ts`. Compare it with the original converted file:

```bash
python examples/swellex96/example_swellex96_visualization_ts.py
```

The round trip uses a rectangular FFT window and matching output records.
Floating-point differences and discarded incomplete trailing samples are
expected.

### Averaged channel PSD

```bash
python examples/swellex96/example_swellex96_pipeline_ts_to_psd.py
python examples/swellex96/example_swellex96_visualization_psd.py
```

The pipeline performs TS -> FFT -> PSD -> five-second temporal averaging and
writes a `.psd` file. The visualization shows frequency versus time for one
channel and an average across the selected records.

### Beamformed timeseries

```bash
python examples/swellex96/example_swellex96_pipeline_ts_to_bts.py
python examples/swellex96/example_swellex96_visualization_bts.py
```

The pipeline performs TS -> FFT -> CBF or DMR -> inverse FFT and writes
`*_cbf_beams.bts` or `*_dmr_beams.bts`. Configure relative steering bearings
and optional channel-subarray selection in the pipeline script. The
visualization extracts one beam, plots a waveform, and writes a resampled
`.wav` listening copy beside the `.bts` file.

### Energy Detect bearing-time records

```bash
python examples/swellex96/example_swellex96_pipeline_ts_to_energy_detect.py
python examples/swellex96/example_swellex96_visualization_energy_detect.py
```

The pipeline performs TS -> FFT -> CBF or DMR -> PSD -> temporal averaging ->
band-energy detection and writes an `*_energy_detect_*.btr` file. Optional tee
stages can also save intermediate `.fft` and `.bfft` products. A second example,
`example_swellex96_pipeline_bf_to_energy_detect.py`, starts from a saved `.bfft`
product. Match the viewer's beamformer, background, and output-representation
settings to the pipeline. Float32 values are stored in decibels; `rank_uint8`
values are relative azimuth ranks. The
[band-energy detection concept](../concepts/band-energy-detection.md) gives the
normalization equations, Abraham (2019) citations, and statistical limits. See
[Energy Detect visualization details](https://github.com/Jeffrey-Ballard/WaveKernelRT/blob/main/examples/swellex96/energy_detect_visualization.md)
for display sizing and export behavior. The
[BTR comparison gallery](../concepts/band-energy-detection.md#swellex-96-comparison-gallery)
shows CBF and DMR results for all four background and output-representation
combinations.

## Script catalog

| Script | Requires | Main result |
| --- | --- | --- |
| `example_swellex96_sio_to_ts.py` | Downloaded array and `.sio` files | Converted geometry and `.ts` |
| `example_swellex96_pipeline_ts_to_fft.py` | Converted `.ts` | `.fft` spectra |
| `example_swellex96_pipeline_fft_to_ts.py` | `.fft` and original converted `.ts` | Reconstructed `.ts` |
| `example_swellex96_pipeline_ts_to_psd.py` | Converted `.ts` | Averaged `.psd` |
| `example_swellex96_pipeline_ts_to_bf.py` | Converted `.ts` and geometry | Beam-frequency `.bfft`, optionally DMR `.bfw`/`.wng` diagnostics |
| `example_swellex96_pipeline_ts_to_bts.py` | Converted `.ts` and geometry | Beamformed `.bts` |
| `example_swellex96_pipeline_ts_to_energy_detect.py` | Converted `.ts` and geometry | Energy Detect `.btr`, optionally `.fft`/`.bfft` |
| `example_swellex96_pipeline_bf_to_energy_detect.py` | Beam spectrum `.bfft` | Energy Detect `.btr` |
| `example_swellex96_visualization_ts.py` | Original and reconstructed `.ts` | Comparison plot |
| `example_swellex96_visualization_fft.py` | `.fft` | Amplitude and phase plots |
| `example_swellex96_visualization_psd.py` | `.psd` | PSD plots |
| `example_swellex96_visualization_bts.py` | `.bts` | Waveform plot and `.wav` |
| `example_swellex96_visualization_dmr_weights.py` | DMR `.bfw` and geometry | Stored DMR and Taylor-CBF beampattern comparison |
| `example_visualization_dmr_wng.py` | DMR `.wng` | Absolute, constraint-relative, and time-history WNG diagnostics |
| `example_swellex96_visualization_energy_detect.py` | `.btr` | Bearing-time energy or rank image |

## Important behavior

- Pipeline outputs with an existing name are overwritten.
- Indices in user settings are zero-based positions, not hydrophone IDs or
  bearing values, unless the setting explicitly says otherwise.
- Absolute acoustic levels require calibrated input samples.
- Do not commit downloaded recordings or generated products to this repository.
- The DMR algorithm reference is listed in [Algorithm references](../references.md).
