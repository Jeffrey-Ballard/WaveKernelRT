# Installation and quick start

WaveKernelRT requires Python 3.12 or newer. The package itself is pure Python
and is expected to work on any platform supported by its dependencies.
Development and testing have primarily taken place on macOS, so feedback from
other platforms is welcome.

## Create an environment

From the root of a WaveKernelRT source checkout, create a virtual environment
and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Editable installation means changes made under `src/wavekernelrt/` are used
without reinstalling the package.

## Verify the installation

With the virtual environment active, run:

```bash
python -c "from wavekernelrt import Geometry; print('WaveKernelRT import succeeded')"
```

The command should print:

```text
WaveKernelRT import succeeded
```

## Run a first geometry example

The repository includes a small line-array geometry, so this example requires
no external data. Run it from the repository root:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from wavekernelrt import Geometry

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

Additional self-contained scripts are listed under
[Basic arrays](../examples/basic-arrays.md). File-processing examples and
their external-data requirements are listed in the
[Examples overview](../examples/index.md).

## Build this documentation

Install the optional documentation dependencies and start the local preview:

```bash
python -m pip install -e ".[docs]"
make docs-serve
```

Then open <http://127.0.0.1:8000/>. Press `Ctrl+C` in the terminal to stop the
preview server.
