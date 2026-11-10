# Data shapes and naming

WaveKernelRT uses variable-name suffixes to make physical units and NumPy axis
order visible at the call site. This page defines that notation and the
canonical shapes exchanged by the pipeline.

## Naming pattern

Names generally follow this pattern:

```text
quantity[_unit]_<axes>
```

For example:

- `xyz_m_N3`: Cartesian positions in meters, shape `(N, 3)`
- `freq_Q`: frequency coordinates, shape `(Q,)`
- `data_QJ`: data with frequency first and beam second, shape `(Q, J)`
- `steer_rbe_rad_J3`: steering range/bearing/elevation coordinates with angular
  values in radians, shape `(J, 3)`

The trailing axis suffix describes array order, not multiplication. `QN` means
axis 0 is frequency and axis 1 is sensor/channel. Transposing the data should
therefore change the suffix from `QN` to `NQ`.

## Common axis letters

| Letter | Usual meaning |
| --- | --- |
| `A` | Azimuth or bearing coordinate |
| `B` | Frequency band |
| `C` | Horizontal geometry column |
| `D` | Retained dominant mode or eigenspace dimension |
| `E` | Elevation coordinate |
| `J` | Beam, look direction, or steering direction |
| `N` | Sensor, array element, or input channel |
| `Q` | Frequency bin |
| `T` | Temporal axis: samples within a record or a sequence/record stack |
| `3` | Cartesian or range/bearing/elevation coordinate triplet |
| `1` | Explicit singleton axis retained for broadcasting |
| `X`, `K` | Generic or flattened coordinate/direction count |

Some letters are context-dependent in private implementation details. For
example, `R` may mean a steering-range axis or a geometry-row axis. The complete
variable name, function documentation, and shape remain authoritative.

Lowercase axis letters often identify a subset of the corresponding uppercase
axis. For example, `q` may omit the first and last bins of `Q`, and `c` may
denote only selected geometry columns. This is an implementation convention,
not a separate physical dimension.

## Units and qualifiers

Unit and representation qualifiers appear before the axis suffix:

| Suffix component | Meaning |
| --- | --- |
| `_m` | meters |
| `_ft` | feet |
| `_mpers` | meters per second |
| `_ftpers` | feet per second |
| `_Hz` | hertz |
| `_deg` | degrees |
| `_rad` | radians |
| `_ns` | nanoseconds |
| `_dB` | decibels |

`_dB` identifies a logarithmic value but does not by itself define the
conversion. Power and PSD quantities use `10*log10`; amplitude responses such
as beampattern magnitude use `20*log10`. The producing function should state
which quantity is represented.

## Canonical pipeline blocks

| Block | Data member | Shape | Meaning |
| --- | --- | --- | --- |
| `TimeBlock` | `data_NT` | `(N, T)` | Sensor/channel time series |
| `TimeBlock` used for BTS | `data_JT` | `(J, T)` | Beam time series; alias of `data_NT` |
| `FFTBlock` | `data_QN` | `(Q, N)` | Channel spectra or channel PSD |
| `BFBlock` | `data_QJ` | `(Q, J)` | Beam spectra or beam PSD |
| `BTRBlock` | `data_BAE` | `(B, A, E)` | Band-by-bearing-by-elevation record |

The `FFTBlock` and `BFBlock` classes may contain complex spectra or real PSD
values. The surrounding file type and pipeline stage determine which semantic
quantity is present.

Post-beamforming blocks additionally carry one scalar `array_heading_deg` for
the represented record. It is metadata rather than another data-array axis.

## Common geometry and beamforming arrays

| Name or pattern | Shape | Meaning |
| --- | --- | --- |
| `xyz_m_N3` | `(N, 3)` | Element positions in meters |
| `normals_N3` | `(N, 3)` | Element-normal unit vectors |
| `phase_center_m_13` | `(1, 3)` | Phase center with singleton element axis |
| `steer_rbe_rad_J3` | `(J, 3)` | Steering range, bearing, and elevation |
| `time_delays_NJ` | `(N, J)` | Per-element delay for each direction |
| `complex_phasors_QNJ` | `(Q, N, J)` | Steering phasors |
| `shade_N` | `(N,)` or vector equivalent | Shared element shading |
| `shade_JN` | `(J, N)` | Per-beam element shading for curved arrays |
| `weight_QNJ` | `(Q, N, J)` | Beampattern-oriented beamformer weights |
| `cbf_weight_QJN` | `(Q, J, N)` | Processing-stage CBF weights |
| One BFW weight record | `(Q, J, N)` | Stored multiplication coefficients; element axis is contiguous |
| One WNG record | `(Q, J)` | Dimensionless linear white-noise gain |
| `bands_B2` | `(B, 2)` | Inclusive lower/upper frequency bounds in Hz |

The two weight layouts above intentionally reflect their consumers. Always use
the suffix rather than assuming every weight tensor uses the same order.

## File-reader return shapes

File readers expose both individual records and record stacks. Their temporal
axis placement is format-specific:

| Reader result | Shape | Notes |
| --- | --- | --- |
| TS data | <code style="white-space: nowrap;">(N,T)</code> | Requested records concatenate along sample axis `T` |
| BTS data | <code style="white-space: nowrap;">(J,T)</code> | Requested records concatenate along sample axis `T` |
| One FFT/PSD record | <code style="white-space: nowrap;">(Q,N)</code> | Same order as `FFTBlock.data_QN` |
| Multiple FFT/PSD records | <code style="white-space: nowrap;">(T,Q,N)</code> | Record index is the leading axis |
| One BFFT/BPSD/WNG record | <code style="white-space: nowrap;">(Q,J)</code> | Same order as `BFBlock.data_QJ` |
| Multiple BFFT/BPSD/WNG records | <code style="white-space: nowrap;">(Q,J,T)</code> | Record index is the trailing axis |
| Multiple BFW records | <code style="white-space: nowrap;">(Q,J,N,T)</code> | Record index is the trailing axis |
| Selected BTR elevations/records | <code style="white-space: nowrap;">(E,B,A,T)</code> | Elevation and record axes are preserved |

Post-beamforming readers return a second value, `headings_T`, containing one
array heading per selected record. Single-record BF and BFW readers instead
return one scalar heading with the selected data.

These layouts are not currently uniform across all multi-record readers. Code
that loads files directly should name the returned array with its full suffix
and should not rely on `squeeze()`, which can silently remove meaningful
singleton axes.

## Steering-grid flattening

If a steering grid has `R` ranges, `A` bearings, and `E` elevations, then
`J = R*A*E`. The `(J, 3)` representation is ordered with elevation fastest,
bearing next, and range slowest:

```text
r0, a0, e0
r0, a0, e1
...
r0, a1, e0
...
r1, a0, e0
```

This ordering allows a direction axis to be reshaped to `(R, A, E)` without
reordering values.

## Geometry flattening

For regular panels, cylinders, and spheres, `N = C*R`, where `C` is the number
of horizontal columns and `R` is the number of vertical rows. Rows vary fastest:

```python
values_CR = values_N.reshape(Ncols, Nrows)
```

This rule is used by spatial shading and curved-array visibility masks.

## Dtypes

The pipeline's nominal real and complex array aliases are `float32` and
`complex64`. TS and BTS samples are stored as `float32`. Spectral and
beam-frequency files store the dtype identified in their headers; the current
pipeline writes complex FFT/BFFT products and real PSD/BPSD products.
Band-energy pipelines write BTR values as `float32` decibels or azimuth ranks
as `uint8`, depending on configuration.

Some public utilities accept a wider numeric dtype and validate or convert it
at an I/O boundary. Shape, units, and dtype are separate parts of the data
contract.

## Examples

```python
# 64 sensors, 4096 time samples
data_NT.shape == (64, 4096)

# 2049 one-sided FFT bins for the same 64 sensors
data_QN.shape == (2049, 64)

# 2049 frequency bins and 201 beams
data_QJ.shape == (2049, 201)

# 5 energy bands, 360 bearings, 1 elevation
data_BAE.shape == (5, 360, 1)
```

When adding an API, include the axis suffix in scientific array names and state
the shape explicitly in its docstring.
