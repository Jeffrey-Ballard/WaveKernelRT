# Documentation roadmap

This file is the author-facing plan for WaveKernelRT's documentation. The
reader-facing entry point is [index.md](index.md).

The documentation should help three audiences:

1. Evaluators deciding whether the project is relevant and technically sound.
2. Users installing WaveKernelRT and completing an acoustic-processing task.
3. Contributors reviewing, testing, or extending the implementation.

## Proposed information architecture

The page names below are planned destinations. They should be added as their
content is written and reviewed; empty placeholder pages are intentionally not
created.

```text
docs/
|-- index.md
|-- references.md
|-- getting-started/
|   |-- installation.md
|   |-- first-geometry.md
|   `-- first-pipeline.md
|-- concepts/
|   |-- conventions.md
|   |-- data-shapes.md
|   |-- pipeline-architecture.md
|   |-- geometry-and-steering.md
|   |-- shading-and-beampatterns.md
|   |-- spectral-processing.md
|   |-- beamforming.md
|   `-- band-energy-detection.md
|-- how-to/
|   |-- build-a-pipeline.md
|   |-- add-a-processing-stage.md
|   |-- select-an-array-subset.md
|   |-- save-intermediate-results.md
|   `-- visualize-results.md
|-- examples/
|   |-- index.md
|   |-- basic-arrays.md
|   `-- swellex96.md
|-- reference/
|   |-- configuration.md
|   |-- file-formats.md
|   |-- public-api.md
|   `-- glossary.md
`-- development/
    |-- contributing.md
    |-- testing.md
    |-- documentation.md
    `-- releases.md
```

## Page specifications

### Getting started

These pages should get a new user to a verified result with minimal domain
knowledge.

- **Installation:** supported Python and platform versions, virtual-environment
  setup, editable installation, and an import verification.
- **First geometry:** load a bundled geometry, inspect its derived properties,
  and display it.
- **First pipeline:** process a small redistributable fixture from input to
  output without requiring the external SWellEx-96 dataset.

The first pipeline tutorial depends on creating a small example dataset or a
documented data-generation utility.

### Concepts

Concept pages explain why the system behaves as it does.

- **Conventions:** coordinate axes, bearings, elevations, input and internal
  units, timestamps, and file-specific time meanings.
- **Data shapes:** dimension letters such as `N`, `Q`, `J`, `T`, `B`, `A`, and
  `E`; axis ordering; real and complex dtypes; unit suffixes.
- **Pipeline architecture:** configuration versus runtime objects, sources,
  stages, blocks, sinks, timebase propagation, cleanup, and validation.
- **Geometry and steering:** geometry-file layout, phase centers, normals,
  baffling, array headings, subarrays, and near-/far-field steering.
- **Shading and beampatterns:** uniform, Hamming, and Taylor shading, sidelobe
  convention, beam weights, normalization, and plot interpretation.
- **Spectral processing:** FFT windows, advance/overlap, inverse FFT behavior,
  PSD scaling, cropping, and temporal averaging.
- **Beamforming:** conventional and DMR processing, required inputs,
  parameters, assumptions, and limitations.
- **Band-energy detection:** input products, band integration, optional
  background normalization, output encodings, and interpretation.

Each scientific page should include equations or references where useful,
define units and shapes, and distinguish mathematical conventions from
implementation choices.

### How-to guides

How-to pages should answer one practical question each and assume the reader
already understands the basic concepts. Every guide should state prerequisites,
show a complete focused example, and describe its output.

### Examples

The example index should list, for every script:

- Purpose and expected output
- Required local or external data
- Values and paths the user must edit
- Preceding scripts that must be run
- Approximate runtime or memory needs when significant
- A command showing how to run it from the repository root

The basic-array examples should remain self-contained. The SWellEx-96 guide must
explain that the recordings are external and document the expected directory
layout without redistributing restricted data.

### Reference

Reference pages describe exact behavior rather than teaching a workflow.

- **Configuration:** every configuration dataclass, field, type, unit, default,
  validation rule, and valid pipeline position.
- **File formats:** header and record layouts, byte order, dtypes, dimensions,
  timestamps, sequence behavior, compatibility, and read/write examples for
  `.ts`, `.bts`, `.fft`, `.psd`, `.bfft`, `.bpsd`, `.wng`, `.bfw`, and `.btr`.
- **Public API:** generated from docstrings after the supported import surface is
  explicitly chosen.
- **Glossary:** acoustic, signal-processing, and project-specific terminology.

### Development

Developer pages should document environment setup, repository structure, test
and quality commands, extension patterns, documentation style, and the release
process. `CONTRIBUTING.md` should provide a concise entry point and link here.

## Documentation standards

- Use one authoritative page for each convention; link to it instead of copying
  details into several pages.
- Use NumPy-style docstrings for public Python APIs.
- State array shapes and physical units for every scientific input and output.
- Label experimental behavior and known limitations explicitly.
- Prefer runnable examples with repository-relative paths.
- Avoid examples containing developer-specific absolute paths.
- Explain why a numerical or domain choice is made, not just which function is
  called.
- Keep private implementation details out of user tutorials unless they affect
  observable behavior.
- Test commands and code snippets where practical.

## Recommended writing order

1. **Completed:** reconcile and expand coordinate, unit, time, and shape
   conventions.
2. **Completed:** write the examples index and remove machine-specific paths.
3. **Completed:** document pipeline architecture and configuration compatibility.
4. **Completed:** specify the binary file formats.
5. **Completed:** document geometry, shading, steering, and beamforming concepts.
6. **Completed:** document FFT, PSD, averaging, inverse FFT, and band-energy concepts.
7. **Completed:** complete docstrings for the defined public API.
8. **Completed:** generate API reference pages from the reviewed docstrings.
9. **Completed:** add contributor, testing, documentation, and release guidance.
10. Add documentation builds and link/example checks to continuous integration.

## Decisions still required

- Decide which platforms are officially supported.
- Decide whether binary formats will receive explicit version fields and a
  compatibility policy.
- Select a documentation hosting target.

## Decisions completed

- WaveKernelRT uses the BSD 3-Clause License.
- Software citation metadata is maintained in the root `CITATION.cff` file.
- Publications, technical reports, and presentations are requested to
  acknowledge WaveKernelRT when it contributes to their analysis.
- The primary public import surface is defined in
  [Public API](reference/public-api.md); format-specific file utilities remain
  supported in their existing modules.
- MkDocs and mkdocstrings build the HTML documentation and generated API
  reference.

## Building the documentation

Install the package and documentation tools from the repository root:

```bash
python -m pip install -e ".[docs]"
```

Build the site with strict warning handling:

```bash
make docs
```

The generated site is written to `site/`. For a browser preview that rebuilds
when source files change, run:

```bash
make docs-serve
```

Then open <http://127.0.0.1:8000/>. The older Pandoc PDF commands remain
available as `make docs-pdf` when Pandoc and XeLaTeX are installed.
