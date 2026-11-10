# Testing

WaveKernelRT currently uses Python's standard-library `unittest` framework.
Tests include focused numerical checks and small file-backed pipeline
integrations.

## Run the complete suite

From the repository root with the development environment active:

```bash
python -m unittest discover -s tests -v
```

The command discovers files named `test*.py`. A successful run ends with
`OK`; warnings or informational messages printed by scientific and plotting
dependencies do not by themselves indicate failure.

## Run one test module or method

Use a dotted module path to narrow the run:

```bash
python -m unittest tests.test_shading -v
python -m unittest \
  tests.test_shading.TestHammingShading.test_line_array_matches_symmetric_hamming_window \
  -v
```

Run the installation smoke script separately because its filename is not part
of unittest discovery:

```bash
python tests/script_test_install.py
```

The script builds a complete offline bundle, creates a temporary virtual
environment outside the source checkout, installs exclusively from the bundled
wheelhouse with `--no-index`, and performs imports with Python's isolated mode.
This guards against incomplete offline bundles, accidentally importing from the
editable checkout, or omitting package files from the wheel. Bundle preparation
requires package-index access; installation does not. The script removes its
temporary bundle and environment when it finishes.

## Test areas

The current suite covers:

- package-level and pipeline-level public exports;
- steering-grid construction and navigation-bearing conversion;
- spatial shading for line, panel, and curved arrays;
- FFT-window power normalization and sinusoid recovery;
- beam-PSD background estimation;
- band-energy processing and complete pipeline output;
- bearing-time image behavior.

The suite does not currently define a coverage threshold, performance gate, or
cross-platform test matrix. Performance benchmarks are not currently included
in the public repository.

## Writing tests

Add a test when changing observable behavior or fixing a bug. Prefer a small
deterministic input whose expected result can be derived independently. Tests
should:

- inherit from `unittest.TestCase`;
- use `numpy.testing` for numerical arrays;
- state tolerances appropriate to the stored dtype and calculation;
- check shapes, dtypes, units, and normalization where relevant;
- use temporary directories for generated files;
- close pipeline resources, normally by running them through `run_pipeline()`;
- avoid external recordings, network access, and developer-specific paths.

Pipeline-stage changes should test both the numerical operation and builder
integration. Stateful stages also need startup, steady-state, timestamp, and
incomplete-buffer behavior where applicable.

## Documentation validation

Documentation is a separate required check:

```bash
make docs
```

This runs `mkdocs build --strict`, so broken internal links, missing generated
API objects, and MkDocs warnings fail the build.
