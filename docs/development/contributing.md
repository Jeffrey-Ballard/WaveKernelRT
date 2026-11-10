# Development environment and repository structure

This page expands the repository-level `CONTRIBUTING.md` guide with the project
structure and extension boundaries relevant to code changes.

## Environment setup

WaveKernelRT requires Python 3.12 or newer. Create an isolated environment and
install the package in editable mode with its documentation tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[docs]"
```

Editable installation places the `src/` package on the environment's import
path while keeping the implementation in the working tree. Confirm which
interpreter and package are active with:

```bash
python -c "import sys, wavekernelrt; print(sys.executable); print(wavekernelrt.__file__)"
```

See [Dependency and packaging policy](dependencies.md) for runtime floors,
optional groups, and environment reproducibility. The separate distribution
test described under [Testing](testing.md) builds and installs an offline bundle.

## Repository structure

```text
WaveKernelRT/
|-- cpp/                   Experimental native implementations
|-- docs/                  MkDocs source pages
|-- examples/              Self-contained and external-data workflows
|-- src/wavekernelrt/
|   |-- core/              Geometry, algorithms, plotting, and file utilities
|   |-- pipeline/          Configuration, blocks, construction, and execution
|   |-- processing/        Runtime processing stages
|   |-- sinks/             Output-file sinks
|   `-- sources/           Input-file sources
|-- tests/                 Unit and pipeline integration tests
|-- tools/                 Distribution and offline-installation utilities
|-- CONTRIBUTING.md        Contributor entry point
|-- mkdocs.yml             Documentation site configuration
|-- pyproject.toml         Package metadata and dependencies
`-- Makefile               Documentation convenience commands
```

Generated documentation is written to `site/` and is not committed. Local
virtual environments, Python caches, build artifacts, and editor metadata are
also excluded through `.gitignore`.

## Scientific conventions

Changes must preserve the contracts in [Conventions](../concepts/conventions.md)
and [Data shapes and naming](../concepts/data-shapes.md). In particular:

- State physical units in names or documentation.
- Use established suffixes such as `_N3`, `_QNJ`, and `_J3` for array axes.
- Treat bearing as clockwise from north in the East-North-Up frame.
- Distinguish user-facing units from internal SI units.
- Document timestamp meaning and sequence alignment for buffered stages.

When an intentional change alters one of these rules, update the authoritative
concept page, affected file formats, examples, and tests together.

## Extending a pipeline

The configured pipeline recognizes explicit configuration classes rather than
discovering stages dynamically. Adding a stage normally requires:

1. A frozen configuration dataclass in `pipeline/configs.py`.
2. A runtime stage whose `process(block)` method yields zero or more blocks.
3. Builder logic that validates the current logical product and propagates
   metadata.
4. Any required source, sink, block, or binary-format support.
5. Unit tests for the numerical operation.
6. Pipeline tests for compatibility, timing, shapes, buffering, and cleanup.
7. Configuration, concept, API, and example documentation.

See [Pipeline architecture](../concepts/pipeline-architecture.md#extension-boundary)
for the runtime protocols and builder boundary.

## Public and internal interfaces

The names exported by `wavekernelrt.__all__` form the curated top-level public
API. Exported configuration and execution objects are also available through
`wavekernelrt.pipeline`. Adding or removing a public name requires updates to:

- `src/wavekernelrt/__init__.py` and, when applicable,
  `src/wavekernelrt/pipeline/__init__.py`;
- `tests/test_public_api.py`;
- the [Public API overview](../reference/public-api.md); and
- the appropriate generated API page.

Runtime stage classes, sources, sinks, helpers, and block implementations are
internal unless the public documentation explicitly says otherwise.

## Working with examples and data

Basic-array examples must remain self-contained. SWellEx-96 scripts may depend
on recordings downloaded by the user, but those recordings and generated data
products must not be committed. Keep paths configurable and repository-relative
where possible; never commit a developer-specific absolute path.
