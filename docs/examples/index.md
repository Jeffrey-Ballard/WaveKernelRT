# Examples

WaveKernelRT includes self-contained array examples and data-processing examples
based on the public SWellEx-96 experiment. Run commands below from the repository
root after completing the
[installation and quick start](../getting-started/installation.md).

## Choose an example set

| Example set | External data | Purpose |
| --- | --- | --- |
| [Basic arrays](basic-arrays.md) | None | Construct geometries and inspect shading and conventional-beamforming beampatterns. |
| [SWellEx-96](swellex96.md) | Required | Convert experimental recordings, run processing pipelines, and inspect saved products. |

The basic-array examples are the best first check of a new installation. The
SWellEx-96 examples are longer workflows intended for users who have downloaded
the array files and recordings separately.

## Running examples

Activate the environment where WaveKernelRT is installed, then invoke a script
from the repository root. For example:

```bash
python examples/basic_geometries_and_beampatterns/script_example_line_array_geometry_and_beampattern.py
```

The plotting examples open interactive Matplotlib windows and finish after the
windows are closed. They do not save figures unless the script explicitly says
otherwise. The [Plotting and visualization](../guides/plotting.md) guide maps
data products and axis orientations to the available plotting functions.

## Path behavior

Bundled examples resolve bundled files relative to their own location, so they
do not depend on the current working directory or the repository's absolute
path. Configure the SWellEx-96 input and output roots once in
`examples/swellex96/example_paths.py`, as described in the
[SWellEx-96 guide](swellex96.md#configure-local-paths).
