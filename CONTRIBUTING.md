# Contributing to WaveKernelRT

WaveKernelRT is experimental research software. Focused bug reports,
documentation corrections, tests, and narrowly scoped code changes are
welcome. For larger algorithm or file-format changes, discuss the design and
compatibility implications before investing in an implementation.

## Set up a development environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[docs]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Validate a change

Run the complete unit and integration test suite:

```bash
python -m unittest discover -s tests -v
```

Build the documentation with warnings treated as errors:

```bash
make docs
```

Code changes should include tests for observable behavior. Public-interface
changes should also update docstrings, API documentation, examples, and the
public export tests when applicable.

## Contribution expectations

- Keep each change focused and explain its motivation.
- Preserve coordinate, unit, shape, and timestamp conventions.
- Use NumPy-style docstrings for public Python interfaces.
- Include physical units and array-shape suffixes in scientific APIs.
- Do not include restricted recordings, generated output products, local
  absolute paths, credentials, or editor state.
- Do not change a binary format without documenting compatibility and migration
  consequences.
- Retain the BSD 3-Clause license notices when redistributing source.

The project currently has no automated continuous-integration gate, so local
validation is required before a change is submitted.

## Development documentation

- [Development environment and repository structure](docs/development/contributing.md)
- [Testing](docs/development/testing.md)
- [Writing documentation](docs/development/documentation.md)
- [Release process](docs/development/releases.md)
- [Coordinate, unit, and time conventions](docs/concepts/conventions.md)
- [Data shapes and naming](docs/concepts/data-shapes.md)

