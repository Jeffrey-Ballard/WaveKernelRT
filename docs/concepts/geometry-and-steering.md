# Geometry and steering

Geometry connects physical sensor positions to the phase replicas used by
beampattern and beamforming calculations. This page explains the implemented
model. Coordinate axes, units, bearings, and steering-grid ordering are defined
authoritatively in [Coordinate, unit, and time conventions](conventions.md).

## Geometry model

Create a geometry from a plain-text element file:

```python
from wavekernelrt.core.geometry import Geometry

geom = Geometry(
    "examples/basic_geometries_and_beampatterns/simple_line_array.txt",
    array_type="line",
    soundspeed_ftpers=5000,
)
```

Each row contains either position alone or position followed by an element
normal:

```text
x  y  z
x  y  z  nx  ny  nz
```

Positions enter in inches and are converted to meters. The mean position is
then subtracted, so `geom.xyz_m_N3` is centered at the origin and has shape
`(N, 3)`. Translation in the input file is intentionally discarded. Optional
normal columns are dimensionless.

The supported array types are:

| `array_type` | Organization | Normal behavior |
| --- | --- | --- |
| `line` | One ordered element sequence | Normals are not needed by current paths |
| `panel_starboard` | Regular columns and vertical rows | Derived toward starboard broadside when omitted |
| `panel_port` | Regular columns and vertical rows | Derived toward port broadside when omitted |
| `cylinder` | Azimuth columns and vertical rows | Derived horizontally outward from the axis |
| `sphere` | Azimuth columns and vertical rows | Derived radially outward from the origin |

For panels, cylinders, and spheres, vertical rows must vary fastest in the
element file. Their current spacing, aperture, normal, and shading calculations
assume a complete regular row/column organization. See
[Data shapes and naming](data-shapes.md#common-geometry-and-beamforming-arrays) for the
array suffixes used below.

## Heading and relative bearings

For line and panel arrays, `array_heading_deg` is the bearing toward the first
element, inferred from element order. Cylinders and spheres use a heading of
zero. The heading is an orientation convention; reversing line-array element
order changes it by approximately 180 degrees.

Configured pipeline beamformers always construct steering vectors with
`flag_relative=True`. Geometries whose absolute heading exceeds one degree are
rotated to align their heading with north before steering is calculated, so
every bearing in `steer_rbe_rad_J3` is relative to array heading. The pipeline
does not expose a geographic steering mode.

For a known array heading, convert a desired geographic look direction to a
relative steering bearing as
`(geographic_bearing - array_heading_deg) % 360`. After processing, plotting
code can express a stored beam geographically with the inverse conversion,
`(relative_bearing + array_heading_deg) % 360`.

The lower-level `SteeringVectors` and standalone `BeamPattern` classes retain a
`flag_relative=False` option for analysis directly in the geographic
East-North-Up frame. This option is separate from configured pipeline
beamforming and its file formats.

## Derived geometry quantities

Geometry properties are calculated lazily when first requested:

| Property | Meaning |
| --- | --- |
| `dr_m` | Median adjacent spacing for a line array |
| `dcol_m`, `drow_m` | Median panel column and row spacing |
| `dring_m`, `drow_m` | Median curved-array ring and row spacing |
| `aperture_m` | End-to-end line aperture |
| `width_aperture_m`, `height_aperture_m` | Horizontal and vertical aperture |
| `design_freq` | Line-array spacing limit `c/(2d)` |
| `design_freq_azimuth` | Horizontal spacing limit `c/(2d_horizontal)` |
| `design_freq_elevation` | Vertical spacing limit `c/(2d_vertical)` |
| `far_field_range_m` | Fraunhofer-style range estimate used by steering |

The spacing-derived design frequency is a spatial-sampling guide, not a hard
processing limit. Operation above it can produce grating lobes whose exact
appearance depends on geometry and look direction.

The far-field metric uses

$$
R_{ff} = \frac{2D^2}{\lambda},
$$

where `D` is the line aperture or the diagonal of a multidimensional aperture.
The current implementation evaluates the wavelength at twice the relevant
spacing-derived design frequency. Treat `far_field_range_m` as an implementation
estimate rather than a universal near-/far-field boundary.

`geom.plot_geometry()` displays positions in feet, marks the first element, and
draws normals for non-line arrays.

## Selecting a subarray

`Geometry.select_elements(indices)` creates a `BeamformingGeometry` in the
parent coordinate system. It preserves selected positions and physical normals
but recomputes the phase center as the mean selected position.

For a line array, any nonempty in-range integer selection is accepted. For a
panel, cylinder, or sphere, the selection must contain every combination of its
chosen rows and columns in original order. For example, choosing three columns
and two rows requires all six corresponding elements. This keeps the selected
geometry reshapeable as `(Ncols, Nrows)`.

The pipeline builder uses this method after `ChannelCropConfig`. The full
geometry supplied to a beamformer must match the original source-channel count;
the builder applies the retained channel indices to both data and geometry.

## Steering grids

A steering grid has shape `(J, 3)` and columns:

1. range in meters,
2. bearing in radians,
3. elevation in radians.

`construct_rbe_X3()` is the convenience constructor for user-facing feet and
degrees:

```python
import numpy as np
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3

steer_rbe_rad_J3 = construct_rbe_X3(
    range_ft_R=np.array([1_000_000.0]),
    bearing_deg_B=np.arange(0.0, 360.0, 2.0),
    elevation_deg_E=np.array([0.0]),
)
```

The grid must contain the complete Cartesian product of its unique axes, with
range slowest, bearing next, and elevation fastest. Axes are ascending. Bearings
crossing north must be unwrapped, such as 358 through 362 degrees, rather than
jumping from 359 to 0.

## Steering-vector calculation

For element position $\mathbf{x}_n$, source position $\mathbf{s}_j$, and
phase center $\mathbf{x}_c$, WaveKernelRT calculates the relative path length

$$
\Delta r_{n,j}
= \lVert \mathbf{x}_n - \mathbf{s}_j \rVert
- \lVert \mathbf{s}_j - \mathbf{x}_c \rVert.
$$

The corresponding delay and complex replica at frequency $f_q$ are

$$
\tau_{n,j} = \frac{\Delta r_{n,j}}{c}, \qquad
a_{q,n,j} = \exp\left(-i2\pi f_q\tau_{n,j}\right).
$$

The negative exponent follows NumPy's forward-FFT convention: a positive
propagation delay contributes negative spectral phase. Beamforming uses the
conjugated replica as its multiplication weight, canceling that phase delay to
align the channels for the selected look direction.

The implementation exposes these as `time_delays_NJ` with shape `(N, J)` and
`complex_phasors_QNJ` with shape `(Q, N, J)`.

Finite ranges produce spherical-wave path differences. Very large requested
ranges are treated as far-field steering: calculation coordinates beyond
`1.5 * max(far_field_range_m, 1000 m)` are clipped to that value to avoid losing
small inter-element differences to floating-point precision. The original
steering grid is still retained in file metadata.

## Numerical precision

Steering defaults to `float32` path differences and frequency values. Before
calculation, the implementation estimates the worst phase error from aperture,
frequency, sound speed, and floating-point precision:

- above 0.5 degrees, it emits a runtime warning;
- at or above 5 degrees, it raises an error;
- `precision="float64"` is available to `SteeringVectors` and `BeamPattern`.

Changing sound speed or look directions through the steering object recomputes
delays and phasors. Pipeline CBF and DMR stages currently use `float32`
steering.

## Assumptions and limitations

- Geometry files have no embedded unit or coordinate-system metadata.
- Input positions are always interpreted as inches and recentered.
- Regular multidimensional geometry is inferred from unique `z` values and
  element order; the loader does not reconstruct arbitrary topology.
- Far-field steering is represented by a distant finite source rather than a
  separate plane-wave formula.
- The steering model includes propagation phase but no range-dependent
  spreading loss or sensor calibration response.
- Element normals affect curved-array visibility and shading, not propagation
  delay.

Runnable construction and plotting examples are listed in
[Basic geometry and beampattern examples](../examples/basic-arrays.md).
