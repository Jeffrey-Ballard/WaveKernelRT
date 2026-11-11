# WaveKernelRT

[![CI](https://github.com/Jeffrey-Ballard/WaveKernelRT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Jeffrey-Ballard/WaveKernelRT/actions/workflows/ci.yml)

WaveKernelRT is an experimental Python framework for acoustic-array analysis and
streaming signal-processing pipelines. It brings array geometry, steering,
shading, beamforming, spectral processing, custom binary file I/O, and
visualization into one inspectable research codebase.

The project currently focuses on file-backed processing and algorithm
development. Its block-oriented pipeline is intended to support future live
audio, optimized C++, and GPU backends, which are currently under development.

## Why WaveKernelRT?

WaveKernelRT is being developed to make acoustic-array workflows easier to
compose, inspect, and test. Current capabilities include:

- Line, panel, cylindrical, and spherical array geometries
- Near- and far-field steering-vector calculations
- Uniform, Taylor, and Hamming spatial shading
- Conventional beamforming (CBF) and element-space DMR processing
- Time-series, FFT, inverse FFT, PSD, averaging, cropping, and band-energy stages
- Configurable source-stage-sink processing pipelines
- Readers and writers for WaveKernelRT time-series and spectral data products
- Geometry, beampattern, spectrum, time-series, and bearing-time visualization
  tools
- End-to-end examples using the publicly available SWellEx-96 experiment data

## Project status

WaveKernelRT is under active development and should currently be treated as
experimental research software. APIs and on-disk formats may change before a
stable release.

The pipeline operates on existing files; real-time device input/output is a
future goal. Unit-test coverage and public API documentation are also still in
progress.

## Requirements

- Python 3.12 or newer
- NumPy 1.26 or newer
- SciPy 1.12 or newer
- Matplotlib 3.8 or newer

WaveKernelRT itself is pure Python and is expected to work on any platform
supported by its dependencies. Development and testing have primarily taken
place on macOS, so feedback from other platforms is welcome.

## Installation

Clone the repository, create a virtual environment, and install the package in
editable mode:

```bash
git clone https://github.com/Jeffrey-Ballard/WaveKernelRT.git
cd WaveKernelRT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Verify the installation from the repository root:

```bash
python -c "from wavekernelrt.core.geometry import Geometry; print('WaveKernelRT import succeeded')"
```

For systems without Internet access, see the
[offline installation guide](docs/getting-started/offline-installation.md). The
repository can prepare a platform-specific transfer bundle containing
WaveKernelRT and all required dependency wheels.

## Docker

Docker can package WaveKernelRT, Python, and its runtime dependencies into an
isolated Linux image. Docker Desktop or another compatible container runtime is
required; the host Python installation is not used.

From the repository root, build the image:

```bash
docker build --progress=plain --tag wavekernelrt:local .
```

Create a host output directory and run the default line-array example:

```bash
mkdir -p docker-results
docker run --rm \
  --mount type=bind,source="$PWD/docker-results",target=/results \
  wavekernelrt:local
```

On Windows PowerShell, ensure Docker Desktop is using Linux containers and use
an absolute host path for the bind mount:

```powershell
New-Item -ItemType Directory -Force docker-results | Out-Null
$resultsPath = (Resolve-Path docker-results).Path

docker run --rm `
  --mount "type=bind,source=$resultsPath,target=/results" `
  wavekernelrt:local
```

The PowerShell form follows Docker's absolute-source-path requirement but has
not been exercised as part of this Apple Silicon workflow.

The container generates a 50-element line-array geometry, applies Taylor
shading, computes conventional-beamforming weight and look sweeps, and writes:

```text
docker-results/
|-- simple_line_array.txt
|-- line_array_geometry.png
|-- line_array_shading.png
|-- line_array_weight_sweep.png
`-- line_array_look_sweep.png
```

`--rm` removes the container after its process exits. The bind-mounted results
remain on the host, and `docker-results/` is ignored by Git.

Run the complete discovered test suite inside a fresh container:

```bash
docker run --rm \
  wavekernelrt:local \
  python -m unittest discover -s tests -v
```

The suite contains targeted unit and small integration tests. In particular, a
deterministic synthetic-signal test writes a WaveKernelRT `.ts` file, runs the
TS -> FFT -> PSD pipeline, and verifies FFT-window normalization and recovered
sinusoid amplitude. Other tests cover focused pipeline stages, file formats,
beamforming, DMR behavior, shading, steering, and plotting.

The suite does not exhaustively test every possible pipeline-stage ordering,
configuration combination, full SWellEx-96 workflow, platform, or numerical
backend. Passing it provides strong regression evidence for the covered paths,
not proof that every composable pipeline has been validated.

This workflow has been verified with Docker Desktop on Apple Silicon using a
native Linux ARM64 image. The official Python base also supports Linux AMD64;
build and test a native image for the architecture used by the deployment
system rather than treating emulated timing as representative performance.

## Quick start

The repository includes a small line-array geometry, so the following example
does not require an external dataset:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from wavekernelrt.core.geometry import Geometry

geometry_file = Path(
    "examples/basic_geometries_and_beampatterns/simple_line_array.txt"
)

geometry = Geometry(
    geometry_file,
    array_type="line",
    soundspeed_ftpers=5_000,
)

print(geometry)
geometry.plot_geometry()
plt.show()
```

Run it from the repository root after completing the installation steps.
Additional examples demonstrate shading and beampatterns for line, nonlinear
line, panel, cylindrical, and spherical arrays.

## Processing pipelines

WaveKernelRT pipelines use a source-stage-sink model:

```text
input file -> source -> processing stage(s) -> sink -> output file
```

A pipeline is described with configuration dataclasses, built into runtime
objects, and then executed block by block. For example, the SWellEx-96 workflows
include:

```text
time series -> FFT -> beamformer -> PSD -> time average -> band energy
```

Supported stages currently include channel and frequency cropping, forward and
inverse FFTs, conventional and DMR beamforming, PSD conversion, temporal
averaging, intermediate-output tees, and band-energy processing.

## Examples

Examples are grouped by the data they require:

- [Basic geometry and beampattern examples](docs/examples/basic-arrays.md) are
  self-contained and require no external data.
- The [SWellEx-96 workflow](docs/examples/swellex96.md) covers conversion,
  processing, and visualization of the horizontal-line-array data.

The SWellEx-96 recordings are not distributed with this repository. Download
them from the [SWellEx-96 experiment website](https://swellex96.ucsd.edu/index.htm):

1. Under [Arrays](https://swellex96.ucsd.edu/arrays.htm), download the HLA North
   and/or HLA South element positions.
2. Under [Events](https://swellex96.ucsd.edu/events.htm), select Event S5 or S59
   and download the matching HLA acoustic recording.
3. Arrange the downloaded files and configure their location as described in
   the [SWellEx-96 workflow](docs/examples/swellex96.md#configure-local-paths).

The conversion script produces the WaveKernelRT geometry and `.ts` input used by
the remaining SWellEx-96 examples.

## Algorithm references

The [Algorithm references](docs/references.md) page lists freely available
sources for specialized algorithms. Its
[DMR implementation mapping](docs/references.md#implementation-mapping)
distinguishes published results from local implementation choices.

## Conventions

WaveKernelRT uses an East-North-Up Cartesian coordinate system:

- `x`: right/east
- `y`: forward/north
- `z`: up
- azimuth: clockwise from north (`0 deg` north, `90 deg` east)
- elevation: positive above the horizontal plane

Geometry files use inches for element locations; internal position calculations
use meters. Time is represented as Unix nanoseconds where an absolute timestamp
is required.

See [Coordinate, unit, and time conventions](docs/concepts/conventions.md) and
[Data shapes and naming](docs/concepts/data-shapes.md) for the detailed data
contract.

## Repository layout

```text
WaveKernelRT/
|-- docs/                 Documentation landing page and writing roadmap
|-- examples/
|   |-- basic_geometries_and_beampatterns/
|   `-- swellex96/
|-- src/wavekernelrt/
|   |-- core/             Geometry, algorithms, plotting, and file utilities
|   |-- pipeline/         Configuration, blocks, construction, and execution
|   |-- processing/       Pipeline processing stages
|   |-- sinks/            Output sinks
|   `-- sources/          Input sources
|-- tests/                Unit and pipeline integration tests
|-- tools/                Offline distribution and installation utilities
|-- CONTRIBUTING.md       Contributor entry point
|-- CITATION.cff          Machine-readable software citation metadata
|-- LICENSE               BSD 3-Clause license
|-- pyproject.toml
`-- README.md
```

## Documentation

The reader-facing documentation begins at [`docs/index.md`](docs/index.md).

Current detailed documentation includes the
[coordinate and timing conventions](docs/concepts/conventions.md),
[data-shape conventions](docs/concepts/data-shapes.md),
[geometry and steering](docs/concepts/geometry-and-steering.md),
[shading and beampatterns](docs/concepts/shading-and-beampatterns.md),
[spectral processing](docs/concepts/spectral-processing.md),
[spectral normalization](docs/concepts/spectral-normalization.md),
[beamforming](docs/concepts/beamforming.md),
[band-energy detection](docs/concepts/band-energy-detection.md),
[pipeline architecture](docs/concepts/pipeline-architecture.md),
[public API](docs/reference/public-api.md),
[configuration reference](docs/reference/configuration.md),
and [binary file formats](docs/reference/file-formats.md).

Contributor workflows are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md)
and under [`docs/development/`](docs/development/).
The [dependency and packaging policy](docs/development/dependencies.md)
describes supported dependency floors, metadata ownership, and reproducibility.

## Roadmap

Near-term work is focused on performance and execution:

- Profile representative end-to-end pipelines and establish CPU and GPU
  performance baselines
- Add optional C++ backends for performance-critical beamforming and
  noise-background-estimation kernels while retaining Python reference
  implementations
- Evaluate CUDA acceleration for those kernels where measured workloads and
  data-transfer costs justify GPU execution
- Design bounded asynchronous pipeline execution so upstream FFT production
  can continue while downstream beamforming, background estimation, and
  bearing-time-record (BTR) writing process queued blocks, with explicit
  ordering and backpressure

As well as:
- Defining and testing the supported public API
- Expanding unit and integration test coverage
- Improving validation, portability, and reproducibility

Longer-term goals also include live audio input/output and additional
accelerated backends where profiling demonstrates a need.

## Contributing

Focused bug reports, documentation corrections, tests, and narrowly scoped
code changes are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
environment setup, validation commands, and contribution expectations.

## Citation and acknowledgment

If WaveKernelRT contributes to an analysis, please cite the software using the
metadata in [`CITATION.cff`](CITATION.cff) and acknowledge WaveKernelRT in
resulting publications, technical reports, and presentations. A DOI will be
added to the citation metadata if one is assigned in the future.

A suggested acknowledgment is:

> This work used WaveKernelRT, developed by Jeffrey Ballard, for acoustic-array
> signal processing and analysis.

## License

WaveKernelRT is licensed under the [BSD 3-Clause License](LICENSE). It may be
used, modified, distributed, and commercialized subject to the license's notice
and non-endorsement requirements. WaveKernelRT does not require registration,
license keys, or separate permission for commercial use.
