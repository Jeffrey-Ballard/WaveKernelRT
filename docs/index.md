# WaveKernelRT documentation

WaveKernelRT is an experimental Python framework for acoustic-array analysis
and block-oriented signal-processing pipelines.

The documentation is being developed alongside the public API and test suite.
For installation and a first geometry example, begin with
[Installation and quick start](getting-started/installation.md).

## Start here

- [Installation and quick start](getting-started/installation.md)
- [Examples overview](examples/index.md)
- [Basic geometry and beampattern examples](examples/basic-arrays.md)
- [SWellEx-96 processing workflow](examples/swellex96.md)
- [Plotting and visualization](guides/plotting.md)
- [Coordinate, unit, and time conventions](concepts/conventions.md)
- [Data shapes and naming](concepts/data-shapes.md)
- [Geometry and steering](concepts/geometry-and-steering.md)
- [Shading and beampatterns](concepts/shading-and-beampatterns.md)
- [Spectral processing](concepts/spectral-processing.md)
- [Spectral normalization](concepts/spectral-normalization.md)
- [Beamforming](concepts/beamforming.md)
- [Band-energy detection](concepts/band-energy-detection.md)
- [Pipeline architecture](concepts/pipeline-architecture.md)
- [Configuration reference](reference/configuration.md)
- [Public API](reference/public-api.md)
- [Array-analysis API reference](reference/api/array-analysis.md)
- [Plotting API reference](reference/api/plotting.md)
- [Pipeline API reference](reference/api/pipeline.md)
- [Binary file formats](reference/file-formats.md)
- [Algorithm references](references.md)
- [Contributing and repository structure](development/contributing.md)
- [Testing](development/testing.md)
- [Writing documentation](development/documentation.md)
- [Release process](development/releases.md)

## How the project is organized

WaveKernelRT separates a processing workflow into a source, one or more stages,
and a sink. Sources produce typed blocks, stages transform or aggregate those
blocks, and sinks write the resulting data products.

```text
input file -> source -> stage -> stage -> sink -> output file
```

The `core` package contains array geometry, steering, shading, beam-pattern,
visualization, and file-format utilities. The `pipeline` package describes and
builds processing workflows. Runtime stages live in `processing`, with file
sources and sinks in their corresponding packages.

## Documentation in progress

The next documentation work will cover:

- A first complete processing workflow
- Automated documentation builds and link checks

Until those pages are complete, the scripts under `examples/` are the most
representative usage references. WaveKernelRT remains experimental, so APIs and
on-disk formats may change.
