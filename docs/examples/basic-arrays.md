# Basic geometry and beampattern examples

These examples require no external data. Each script generates a geometry file,
loads it with `Geometry`, displays the geometry and shading, computes a
conventional-beamforming beampattern, and displays the result.

## Prerequisites

Install WaveKernelRT from the repository and activate that Python environment:

```bash
python -m pip install -e .
```

Run each command from the repository root. No path edits are required. On a
typical development computer, each example normally completes its calculations
in seconds, although plotting and the larger cylinder example can take longer.

## Available examples

### Uniform line array

Creates a 50-element, half-wavelength-spaced line array designed for 1 kHz,
applies Taylor shading, and plots its horizontal beampattern.

```bash
python examples/basic_geometries_and_beampatterns/script_example_line_array_geometry_and_beampattern.py
```

Geometry written: `examples/basic_geometries_and_beampatterns/simple_line_array.txt`

### Center-tapered line array

Creates a 251-element nonuniform line array with dense center spacing and a
500-foot aperture. It currently demonstrates uniform shading at 400 Hz. Custom
subarray pruning for the dense center is not yet implemented.

```bash
python examples/basic_geometries_and_beampatterns/script_example_nonlinear_line_array_geometry_and_beampattern.py
```

Geometry written:
`examples/basic_geometries_and_beampatterns/simple_centertapered_line_array.txt`

### Planar panel array

Creates a 20-column by 15-row starboard-facing panel, applies separable Taylor
shading, and plots horizontal and vertical beampattern cuts at 1 kHz.

```bash
python examples/basic_geometries_and_beampatterns/script_example_panel_array_geometry_and_beampattern.py
```

Geometry written: `examples/basic_geometries_and_beampatterns/simple_panel_array.txt`

### Cylindrical array

Creates a 36-column by 20-row cylindrical array, selects the elements visible
from the look direction, applies Taylor shading to that aperture, and plots the
1 kHz beampattern.

```bash
python examples/basic_geometries_and_beampatterns/script_example_cylinder_array_geometry_and_beampattern.py
```

Geometry written: `examples/basic_geometries_and_beampatterns/simple_cylinder_array.txt`

### Spherical array section

Creates a 36-column spherical section spanning approximately -40 to 40 degrees
elevation, selects the visible aperture for an off-axis look direction, applies
Taylor shading, and plots the 1 kHz beampattern.

```bash
python examples/basic_geometries_and_beampatterns/script_example_sphere_array_geometry_and_beampattern.py
```

Geometry written: `examples/basic_geometries_and_beampatterns/simple_sphere_array.txt`

## Expected output

Each script rewrites its listed plain-text geometry file and opens multiple
Matplotlib figures. Depending on the array, these show:

- Element positions and array orientation
- Spatial-shading weights
- A bearing sweep, or bearing/elevation image with horizontal and vertical cuts

The scripts do not save plots. Add `fig.savefig(...)` before `plt.show()` when a
persistent image is needed.

Coordinate definitions, geometry-file units, and array ordering are documented
in [Conventions](../concepts/conventions.md) and
[Data shapes and naming](../concepts/data-shapes.md). The underlying scientific
choices are described in [Geometry and steering](../concepts/geometry-and-steering.md)
and [Shading and beampatterns](../concepts/shading-and-beampatterns.md).
