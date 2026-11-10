# Coordinate, unit, and time conventions

This page is the authoritative reference for WaveKernelRT coordinate systems,
physical units, steering coordinates, timestamps, and sequence timing. Array
dimension letters and variable-name suffixes are defined separately in
[Data shapes and naming](data-shapes.md).

## Cartesian coordinate system

WaveKernelRT uses a right-handed East-North-Up Cartesian coordinate system:

- `x`: right or east
- `y`: forward or north
- `z`: up

For a range $r$, bearing $\theta$, and elevation $\phi$, WaveKernelRT
converts spherical coordinates to Cartesian coordinates as

$$
x = r\sin(\theta)\cos(\phi), \qquad
y = r\cos(\theta)\cos(\phi), \qquad
z = r\sin(\phi).
$$

This differs from the common mathematical convention that measures azimuth
counterclockwise from the positive x-axis.

## Bearing and elevation

Bearing, also called azimuth in parts of the API, is measured clockwise from
north in the horizontal x-y plane:

- `0 deg`: north (`+y`)
- `90 deg`: east (`+x`)
- `180 deg`: south (`-y`)
- `270 deg`: west (`-x`)

Elevation is measured from the horizontal plane. Positive elevation points
toward `+z`; negative elevation points toward `-z`.

User-facing grid builders generally accept angles in degrees. Stored steering
grids and steering-vector calculations use radians. Names containing `_deg` or
`_rad` state which representation is expected.

### Relative and geographic bearings

The geometry's `array_heading_deg` is the bearing toward the first element,
derived from element order for line and panel arrays. The configured processing
pipeline uses one bearing convention: all beamforming steering bearings are
relative to the array heading. `steer_rbe_rad_J3` therefore always contains
relative bearings for CBF and DMR stages; there is no geographic-beamforming
mode in the pipeline.

WaveKernelRT constructs the pipeline steering replicas in the relative frame
by rotating the geometry so its heading is north. A per-record heading supplied
by the input block or a stage's `heading_provider` is carried as metadata; it
does not rotate the steering grid or rebuild the beamforming weights.

Every post-beamforming runtime block and binary record carries that
`array_heading_deg` in degrees clockwise from true north. Plotting code may
express a relative beam in geographic coordinates as
`(relative_bearing + array_heading_deg) % 360`. This is a display-coordinate
choice: the stored data and beam order remain relative and unchanged. When the
heading varies, the conversion must use the heading for each record.

The lower-level `SteeringVectors` and standalone `BeamPattern` analysis classes
also expose `flag_relative=False`, which keeps input bearings in the geographic
East-North-Up frame. That option is not used by configured pipeline beamforming
and does not change the bearing convention of pipeline file formats.

## Geometry files

A geometry text file contains one element per row. The supported layouts are:

```text
x  y  z
x  y  z  nx  ny  nz
```

The first three columns are element positions in inches. The optional last
three columns are dimensionless element-normal vectors. `Geometry` converts
positions to meters and subtracts the mean position, centering the loaded array
at the origin.

If normals are omitted, WaveKernelRT derives them for panel, cylindrical, and
spherical arrays:

- `panel_starboard` and `panel_port` select opposite panel broadside directions.
- Sphere normals point radially outward from the origin.
- Cylinder normals point horizontally outward from the cylinder axis.

Line arrays do not require element normals for their current processing paths.

### Element order for multidimensional arrays

For panels, cylinders, and spheres, elements are expected to be ordered with
vertical `z` rows varying fastest and horizontal columns varying more slowly:

```text
column 0, row 0
column 0, row 1
...
column 1, row 0
column 1, row 1
...
```

This ordering allows an element vector of length `N` to be reshaped as
`(Ncols, Nrows)`. Shading, spacing, normal, and aperture calculations depend on
this convention. These calculations currently assume a regular row/column
organization for panels, cylinders, and spheres.

## Physical units

WaveKernelRT does not use one external unit system everywhere. Geometry and
some convenience constructors retain domain-specific English-unit inputs,
while calculations use SI units internally.

| Quantity | Unit |
| --- | --- |
| Geometry-file positions | inches |
| `Geometry.xyz_m_N3` and internal geometry positions | meters |
| Sound speed passed to `Geometry` | feet/second |
| `Geometry.soundspeed_mpers` and internal sound speed | meters/second |
| Range passed to `construct_rbe_X3` | feet |
| Range returned by `construct_rbe_X3` and stored in steering grids | meters |
| User-facing bearing/elevation grids | degrees |
| Bearing/elevation returned by constructors and stored in steering grids | radians |
| Frequency | hertz |
| Sample rate | samples/second (`Hz`) |
| Absolute time | integer Unix nanoseconds |
| Time intervals | samples or seconds, depending on the field |
| Shading weights | dimensionless, normally normalized to sum to one |

Unit-bearing variable names use suffixes such as `_m`, `_ft`, `_mpers`,
`_ftpers`, `_Hz`, `_deg`, `_rad`, `_ns`, and `_dB`.

## Steering grids

A steering grid has shape `(J, 3)`. Its columns are:

1. range in meters
2. bearing in radians
3. elevation in radians

Names for this representation contain `rbe_rad_J3`, where `b` means bearing.

`construct_rbe_X3` accepts range in feet and angles in degrees, then returns the
internal meter/radian representation.

A complete steering grid contains every combination of its range, bearing, and
elevation axes. It is flattened with:

1. range varying slowest
2. bearing varying next
3. elevation varying fastest

Each axis must be ascending. A sector that crosses north must use an unwrapped
bearing axis, such as `358, 359, 360, 361, 362 deg`, rather than jumping from
`359` to `0`.

## Absolute time

WaveKernelRT stores absolute timestamps as integer nanoseconds since the Unix
epoch in UTC. On disk, timestamps are unsigned 64-bit integers. In Python they
are normalized to `int` so timestamp arithmetic does not silently wrap.

The `Timebase` class maps global sample indices to timestamps:

$$
t_{ns}(s) = t_{origin,ns} +
\operatorname{round}\left(
\frac{(s-s_{origin})10^9}{f_s}
\right).
$$

`Timebase.sample_to_datetime()` returns a timezone-aware UTC `datetime`.
Header fields named `start_unix_seconds` are floating-point conveniences and
may not preserve nanosecond precision; use `start_unix_nanoseconds` for exact
work.

## File timestamp meanings

The header timestamp identifies the first record, but its position within that
record depends on the data product:

| Format | Meaning of `start_unix_nanoseconds` |
| --- | --- |
| `.ts` | Time of the first sample in the first channel time-series record |
| `.bts` | Time of the first sample in the first beam time-series record |
| `.fft`, `.psd` | Center time of the first spectral frame |
| `.bfft`, `.bpsd`, `.wng` | Center time of the first beam-frequency frame |
| `.btr` | Time assigned to the first band-energy output record |

TS and BTS records advance by `Npt_per_seq / sample_rate_hz`. Spectral,
beam-frequency, and BTR records advance by `advance / sample_rate_hz`.

When time series are converted to spectra, the first spectral timestamp is
shifted by `nfft / 2` samples. PSD averaging moves the first output timestamp to
the midpoint of the averaged input record centers and multiplies `advance` by
the number of records averaged. Band-energy processing preserves the incoming
timestamp unless background time averaging is enabled. With a background time
window, the first output timestamp is the selected center record; an
even-length window uses the later of its two middle records.

`generate_time_array_from_header(header)` applies these record-spacing rules and
returns floating-point Unix seconds. It is convenient for plotting, but exact
timestamp comparisons should remain in integer nanoseconds.

## Runtime block timing

Current runtime blocks carry a zero-based `sequence` number. `FFTBlock` and
`BFBlock` also carry `advance`. Post-beamforming `TimeBlock`, `BFBlock`, and
`BTRBlock` instances carry `array_heading_deg`; FFT blocks may also carry it
when heading metadata is already available. Individual runtime blocks do **not** carry
`sample_start`, `sample_center`, `sample_end`, an absolute timestamp, or a
`Timebase` instance.

The pipeline builder keeps a `Timebase` and the evolving first-output timestamp
in `PipelineDataBlock` while it constructs the stages and output header. After a
file is written, record times should be reconstructed from the header according
to the table above.

## Indexing and ordering

- Python APIs and examples use zero-based indices.
- Array suffixes state axis order; axes should not be inferred from a file
  extension alone.
- Bearing axes intended for storage must be strictly increasing, so circular
  axes should be unwrapped when necessary.
- Frequency axes contain nonnegative, strictly increasing, one-sided FFT-bin
  frequencies in hertz. Frequency cropping may retain only part of that grid.

The exact binary layouts are documented in the
[binary file-format reference](../reference/file-formats.md).
