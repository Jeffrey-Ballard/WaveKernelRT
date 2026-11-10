# Public API

WaveKernelRT provides a curated package-level interface for application code.
These names are aliases of their implementations; the underlying source files
have not moved and no wrapper objects are created.

Exports are loaded only when first accessed. A plain `import wavekernelrt`
therefore does not initialize plotting, scientific, or pipeline modules that
the application never uses.

```python
from wavekernelrt import Geometry, PipelineConfig, build_pipeline, run_pipeline
```

Existing module-path imports remain valid. New application code should prefer
the public imports on this page so internal modules can be reorganized without
requiring corresponding changes in user code.

WaveKernelRT is currently experimental and versioned below 1.0. The project
will document incompatible public-API changes, but the interface is not yet
covered by a long-term compatibility guarantee.

## Array analysis

| Public name | Purpose |
| --- | --- |
| `Geometry` | Load and inspect an array geometry |
| `Shading` | Construct spatial aperture shading |
| `SteeringVectors` | Calculate delays and complex steering replicas |
| `BeamPattern` | Construct weights and evaluate beampatterns |
| `construct_rbe_X3` | Build a range-bearing-elevation steering grid from user-facing units |

For example:

```python
from wavekernelrt import BeamPattern, Geometry, Shading

geom = Geometry(
    "array_geometry.txt",
    array_type="line",
    soundspeed_ftpers=5000.0,
)

shade = Shading(geom)
pattern = BeamPattern(geom)
pattern.compute_cbf_weight_QNJ(shade.hamming_shading_N)
```

The scientific conventions for these objects are described in
[Geometry and steering](../concepts/geometry-and-steering.md) and
[Shading and beampatterns](../concepts/shading-and-beampatterns.md).

## Plotting and visualization

Plotting functions use a grouped public namespace rather than adding every
axis orientation to the package root:

```python
from wavekernelrt.core import analysis_images

fig, ax, mesh = analysis_images.psd_image_TQ(
    psd_TQN,
    freq=freq_Hz_Q,
    times=times_s_T,
    idxN=0,
)
```

The named wrappers cover channel FFT and PSD, beam FFT and PSD, WNG, and BTR
products. The lower-level `image_data_RC`, `create_spectrum_image`,
`create_psd_image`, and `set_beam_ticks` functions are also supported public
interfaces for custom plotting code. See [Plotting and
visualization](../guides/plotting.md) for a function chooser and the [Plotting
API](api/plotting.md) for complete signatures.

## Pipeline construction and execution

The primary pipeline entry points are:

| Public name | Purpose |
| --- | --- |
| `PipelineConfig` | Describe the input, ordered stages, and output |
| `BuiltPipeline` | Runtime source, stage, and sink collection returned by the builder |
| `build_pipeline` | Validate a configuration and construct runtime objects |
| `run_pipeline` | Process a built source-stage-sink pipeline |

All user-facing configuration dataclasses are also exported at package level:

- `InputConfig` and `OutputConfig`
- `TStoFFTConfig` and `FFTtoTSConfig`
- `FreqCropConfig` and `ChannelCropConfig`
- `FFTtoPSDConfig` and `PSDAvgConfig`
- `ShadingConfig`, `FFTtoCBFConfig`, and `DMRElementConfig`
- `BackgroundEstimateConfig` and `BandEnergyConfig`
- `TeeConfig`

A complete pipeline can therefore use one import location:

```python
from pathlib import Path

from wavekernelrt import (
    FFTtoPSDConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    TStoFFTConfig,
    build_pipeline,
    run_pipeline,
)

config = PipelineConfig(
    input=InputConfig(file=Path("input.ts"), type="ts"),
    stages=(
        TStoFFTConfig(nfft=4096, advance=2048, window_name="hann"),
        FFTtoPSDConfig(),
    ),
    output=OutputConfig(file=Path("output.psd"), type="psd"),
)

pipeline = build_pipeline(config)
run_pipeline(pipeline.source, pipeline.stages, pipeline.sink)
```

The same pipeline API is available as a grouped namespace:

```python
from wavekernelrt.pipeline import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    build_pipeline,
    run_pipeline,
)
```

See the [Configuration reference](configuration.md) and
[Pipeline architecture](../concepts/pipeline-architecture.md) for field-level
and runtime details.

The generated references provide signatures and docstrings for the complete
curated interface:

- [Array-analysis API](api/array-analysis.md)
- [Plotting API](api/plotting.md)
- [Pipeline API](api/pipeline.md)

## Specialized file access

Binary header and record readers and writers remain in their format-specific
modules, such as:

```python
from wavekernelrt.core.ts_file_utils import ts_read_header, ts_read_sequence
from wavekernelrt.core.fft_file_utils import fft_read_header
```

These functions are documented and supported for direct file access, but are
not re-exported into the top-level namespace. A future dedicated file-I/O
namespace can collect them without crowding the primary analysis and pipeline
API. See [Binary file formats](file-formats.md) for the supported operations.

## Internal interfaces

Runtime stage classes under `wavekernelrt.processing`, block classes under
`wavekernelrt.pipeline.blocks`, source and sink implementations, validation
helpers, and algorithm-specific helper functions are implementation details.
Application pipelines should normally construct them through configuration
objects and `build_pipeline()`.

An internal object may remain importable because Python modules are directly
accessible, but importability alone does not make it part of the curated public
API.
