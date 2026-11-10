# Dependency and packaging policy

`pyproject.toml` is the authoritative source for WaveKernelRT package metadata,
supported Python versions, runtime dependencies, optional dependency groups,
and build-backend requirements.

## Runtime dependencies

WaveKernelRT declares direct runtime dependencies with tested lower bounds:

| Dependency | Minimum | Purpose |
| --- | ---: | --- |
| NumPy | 1.26.0 | Arrays, numerical operations, FFTs, and binary data handling |
| SciPy | 1.12.0 | Windows, filtering, statistics, and signal processing |
| Matplotlib | 3.8.0 | Analysis and diagnostic visualization |

These floors align with the project's Python 3.12 minimum and are exercised by
the dedicated minimum-dependencies CI job on Python 3.12. Normal CI jobs install
current compatible releases on every supported Python minor version.

Runtime dependencies do not receive an upper bound unless a demonstrated
incompatibility requires one. An upper cap based only on an untested future
possibility prevents users from receiving compatible fixes and creates
unnecessary resolver conflicts.

When raising a minimum dependency:

1. confirm that code or supported behavior requires the newer release;
2. update `pyproject.toml` and this page together;
3. update the exact versions in the minimum-dependencies CI job;
4. run the full suite with both the new minima and current dependencies; and
5. describe the compatibility change in the release notes.

## Optional dependencies

Documentation tooling belongs to the `docs` optional-dependency group and is
installed with:

```bash
python -m pip install -e ".[docs]"
```

Tests currently use the standard library plus runtime dependencies, so there is
no separate test extra. Add optional groups according to user-visible purpose
rather than collecting every development tool into the runtime installation.

## Reproducibility and lock files

The repository intentionally does not commit the output of `pip freeze` as
`requirements.txt`. Such a snapshot describes one developer's complete
environment, including transitive and platform-specific packages; it is not a
portable statement of this library's requirements.

For ordinary development, create an isolated environment and install the
project from `pyproject.toml`. Developers may use `pip freeze` locally when
diagnosing or archiving an environment, but should not treat that output as
package metadata.

For a reproducible air-gapped deployment, use the
[offline bundle](../getting-started/offline-installation.md). Its wheelhouse and
manifest record the exact resolved artifacts for one Python, operating system,
and architecture. If another deployment workflow later requires a lock file,
generate and document it for a specific target rather than replacing the broad
library dependency declarations.

## Distribution metadata

WaveKernelRT uses standardized `pyproject.toml` project metadata and a
setuptools build backend. The license is expressed using the SPDX identifier
`BSD-3-Clause`, while `project.license-files` ensures the full `LICENSE` text is
included in source and wheel distributions.

Project version information appears in both `pyproject.toml` and
`CITATION.cff`; releases must update them together. Build artifacts under
`dist/` are generated outputs and are not committed.
