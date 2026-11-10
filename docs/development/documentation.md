# Writing documentation

WaveKernelRT uses Markdown for authored pages, NumPy-style Python docstrings for
public APIs, MkDocs for site generation, and mkdocstrings for API extraction.

## Preview locally

Install the optional documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

Start the live preview server:

```bash
make docs-serve
```

Open <http://127.0.0.1:8000/>. MkDocs rebuilds when a source page or docstring
changes. Stop the server with `Ctrl+C`.

Before submitting documentation changes, run the strict build:

```bash
make docs
```

Generated HTML is written to `site/` and should not be committed.

## Choose the right page type

- **Getting started** leads a new user through a first successful result.
- **Concepts** explain scientific and architectural choices.
- **How-to guides** solve one focused practical task.
- **Examples** describe complete repository scripts and their prerequisites.
- **Reference** states exact behavior, fields, formats, and interfaces.
- **Development** documents contributor workflows.

Add a new page to `nav` in `mkdocs.yml`; otherwise it may build without being
discoverable through site navigation.

## Public docstrings

Use NumPy-style sections such as `Parameters`, `Returns`, `Raises`, `Notes`,
and `Examples`. Scientific inputs and outputs should state physical units,
array shapes, axis meanings, and dtypes when they affect behavior.

Generated API pages live under `docs/reference/api/` and identify the
implementation object with an mkdocstrings directive:

```text
::: wavekernelrt.core.geometry.Geometry
```

Application examples should still use the curated package-level import:

```python
from wavekernelrt import Geometry
```

For docstring cross-references, use mkdocstrings/autorefs Markdown rather than
Sphinx roles:

```text
[`select_elements`][wavekernelrt.core.geometry.Geometry.select_elements]
```

Syntax such as `:meth:` is not interpreted by the configured NumPy docstring
parser and will appear literally in the HTML.

## Links, equations, and code

Use relative links between documentation pages. The strict build validates
internal targets but cannot guarantee that an external website remains
available. Prefer repository documentation over external duplication when the
project owns the authoritative explanation.

Write display equations between `$$` delimiters and inline equations between
`\(` and `\)`. MathJax renders them in the browser. Use fenced code blocks with
a language label where possible:

````text
```python
from wavekernelrt import Geometry
```
````

Examples should be runnable from the repository root, avoid absolute local
paths, and explain required external data. Do not copy restricted recordings
into the documentation tree.

## PDFs

The HTML site is the primary documentation output. The older convention and
README PDF commands remain available when Pandoc and XeLaTeX are installed:

```bash
make docs-pdf
```

Generated PDFs are build products and are not the authoritative source.

