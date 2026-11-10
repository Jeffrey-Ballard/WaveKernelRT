# Release process

WaveKernelRT is currently experimental and has no automated package-publishing
or release workflow. The checklist below defines the review expected before a
maintainer creates a release; it does not imply that a public package index or
documentation host has been selected.

## Choose the release scope

Review user-visible changes since the previous release and decide whether they
affect:

- the curated Python API;
- pipeline configuration or stage compatibility;
- scientific normalization or numerical results;
- coordinate, unit, shape, or timestamp conventions;
- binary header or record layouts; or
- supported Python versions and dependencies.

Every public binary layout begins at format version 1. An incompatible layout
change must increment the affected version and make an explicit compatibility
and migration decision before release. See
[Binary file formats](../reference/file-formats.md#validation-and-compatibility-limits).

## Update versioned metadata

The software version currently appears in two authoritative metadata files:

- `[project].version` in `pyproject.toml`;
- `version` in `CITATION.cff`.

Update them together. Also update affected concept, configuration, file-format,
API, example, and installation documentation. Record important user-visible
changes in the release notes created with the release.

## Validate the candidate

From a clean development environment:

```bash
python -m unittest discover -s tests -v
make docs
python tests/script_test_install.py
```

Review generated API pages and at least one relevant end-to-end example. Tests
that require external SWellEx-96 data should be performed only when that data
is available and must not add the recordings or generated products to the
repository.

The installation smoke script builds the wheel and a complete offline bundle,
then installs from that bundle in a new temporary environment. If artifacts
will be published, also inspect their metadata and contents before publishing.
No public package-index target has yet been selected for this project.

## Tag and publish

Before creating a tag or public release, confirm:

- the target commit contains the reviewed version and documentation;
- tests and the strict documentation build pass on that commit;
- release notes identify incompatible changes and known limitations;
- `CITATION.cff` describes the released version; and
- the license and attribution files are present.

The repository does not yet prescribe a tag naming convention, supported
platform matrix, GitHub release workflow, package-index upload process, or
documentation deployment process. Those choices should be documented here
when they are adopted rather than inferred from a one-time manual command.
