# Shading and beampatterns

Spatial shading controls how array elements contribute to a beam. A beampattern
then evaluates the response of those weighted elements across arrival
directions. This is distinct from the time-domain FFT windows described in
[Spectral normalization](spectral-normalization.md).

## Supported spatial shading

`ShadingConfig` supports three recipes:

| Window | Parameters | Purpose |
| --- | --- | --- |
| `uniform` | None | Equal active-element weights and narrow main response |
| `hamming` | None | Symmetric Hamming taper with reduced sidelobes |
| `taylor` | `sll_dB`, `nbar` | Adjustable near-in sidelobe taper |

`sll_dB` is supplied as a positive attenuation, such as `30.0` for a nominal
-30 dB Taylor sidelobe level. `nbar` is an integer of at least two. These
parameters are ignored for uniform and Hamming shading.

All generated shading vectors are real, nonnegative, and normalized so their
active weights sum to one. This is amplitude normalization across elements; it
is unrelated to FFT-window power normalization.

## Shading by geometry type

### Line arrays

Uniform shading assigns equal weights. Hamming and Taylor use symmetric tapers.
For an unevenly spaced line, the implementation interpolates a sampled taper
over cumulative element position rather than treating element index as uniform
distance.

### Panel arrays

Panel shading is separable. A column taper and row taper are multiplied to form
`shade_CR`, then flattened with rows varying fastest. The final element vector
is normalized to sum to one.

### Cylinders and spheres

Curved arrays are treated as horizontally baffled. Each beam gets its own
visible aperture based on the horizontal element-normal direction. A column is
active when

$$
\mathbf{n}_c \mathbin{\cdot} \mathbf{u}_{look}
> \cos(60^\circ).
$$

Thus, the default aperture includes columns whose outward normals lie within
60 degrees of the beam bearing. Range and elevation do not change this
horizontal visibility decision; every retained vertical row in an active
column participates.

Uniform shading distributes equal weights across the visible elements.
Hamming and Taylor shading apply separable tapers over the visible columns and
rows, set invisible elements to zero, and normalize each beam independently.
The result has shape `(J, N)`.

This per-beam treatment is why curved arrays must use a `ShadingConfig` rather
than one manually supplied shared vector in `BeamPattern`.

## Shared and per-beam shapes

| Geometry | Shading representation | Shape |
| --- | --- | --- |
| Line or panel | One vector shared by every beam | `(N,)` or vector equivalent |
| Cylinder or sphere | One visible-aperture vector per beam | `(J, N)` |

The pipeline stores the same distinction in BFFT, BPSD, and BTS headers:
`Nshades` is either `1` or `Jbeams`.

## Conventional beam weights

Let $a_{q,n,j}$ be the steering replica and $s_{n,j}$ the spatial shading.
WaveKernelRT constructs multiplication weights

$$
w_{q,n,j}
= \frac{s_{n,j}a^*_{q,n,j}}
       {\sum_n s_{n,j}a^*_{q,n,j}a_{q,n,j}}.
$$

The denominator normalizes the response to one at the steering direction.
Because the replicas have unit magnitude, generated sum-normalized shading
normally makes that denominator one already. The explicit normalization also
supports valid custom shared shading whose sum is not one.

The weights appear as `(Q, N, J)` in `BeamPattern` and `(Q, J, N)` in the
streaming CBF stage. The differing suffixes reflect their consumers.

## Constructing a beampattern

```python
from wavekernelrt.core.beampattern import BeamPattern
from wavekernelrt.pipeline.configs import ShadingConfig

bp = BeamPattern(
    geom=geom,
    freq=1000.0,
    Nbearings=181,
    Nelevations=1,
    focus_range_ft=1_000_000.0,
)
bp.set_shading_config(
    ShadingConfig(window_type="taylor", sll_dB=30.0, nbar=4)
)

weights_QNJ = bp.cbf_weight_QNJ
```

`freq` may be one frequency or a one-dimensional frequency vector. If omitted,
the geometry's spacing-derived design frequency is used. If bearing or
elevation counts are omitted, `BeamPattern` estimates a dense plotting grid
from aperture and wavelength. Supplying the counts explicitly makes plot
resolution reproducible.

The default `focus_range_ft=100000` approximates a far-field response for many
example arrays. A smaller finite range evaluates near-field focusing through
the same spherical-wave steering calculation.

For line and panel geometries, `set_shading(shade_N)` also accepts a custom
shared real, finite, nonnegative vector with positive total weight. Curved
arrays require `set_shading_config()` so visibility can be recomputed per beam.

## Two beampattern sweeps

WaveKernelRT exposes two complementary evaluations. The word before *sweep*
identifies what changes:

| Evaluation | Held fixed | Swept | Question answered |
| --- | --- | --- | --- |
| Weight sweep | One signal-arrival direction | A bank of beam weights | How does each beam respond to a source at this location? |
| Look sweep | One set of beam weights | Signal-arrival direction | What is the spatial response of this one beam? |

Another way to remember the distinction is:

- **Weight sweep:** move the receiving beam past a stationary source.
- **Look sweep:** hold the receiving beam stationary and move the source.

### Weight sweep

`compute_beampattern_weight_sweep_QNJ()` accepts weights with shape `(Q, N,
J)` and one fixed arrival direction. It evaluates every beam `J` against that
same arrival replica. This is convenient for displaying a complete bank of
conventional beams and seeing which beams respond to a source at a specified
location.

For fixed arrival direction $\theta_0$, it evaluates

$$
B_{weight}(j) =
\left|\sum_n w_{n,j}a_n(\theta_0)\right|.
$$

### Look sweep

`compute_beampattern_look_sweep_JQ()` accepts one fixed weight set with shape
`(Q, N)` and a grid of arrival directions. It applies those same weights to
every arrival replica. This is the usual answer to “what does the beampattern
of these weights look like?”

For fixed weights $w_n$, it evaluates

$$
B_{look}(\theta) =
\left|\sum_n w_na_n(\theta)\right|.
$$

The look sweep is therefore the natural choice for inspecting one externally
computed or adaptive weight vector, including a DMR weight vector at a
particular time, frequency, and steering direction. A weight sweep of a DMR
beam bank answers a different question: how all those independently computed
beams respond to one fixed arrival.

### When the two plots match

For conventional weights formed with shared real shading,

$$
w_n(\phi) = s_na_n^*(\phi),
$$

a weight sweep past an arrival at $\theta_0$ and a look sweep using the beam
steered to $\theta_0$ produce complex-conjugate sums. Their magnitudes are
therefore equal when both evaluations use the same frequency, direction grid,
focus range, geometry, shading, and normalization.

This equivalence is not guaranteed for adaptive weights. It may also differ
when the active aperture or shading changes with direction, as it can for
curved and baffled arrays.

Both return response magnitude. Plotting converts magnitude to decibels with

$$
B_{dB} = 20\log_{10}\left(\max(|B|, 10^{-12})\right).
$$

The standard plots display down to -60 dB. This floor is a presentation choice,
not a change to the returned linear response.

For curved arrays, arrival replicas from invisible columns are masked during
beampattern evaluation, matching the horizontal baffling model used for
shading.

## Interpreting results

- Uniform shading generally gives the narrowest main response but the highest
  sidelobes of the supported recipes.
- Hamming and Taylor shading trade main-response width for lower sidelobes.
- Beam-grid spacing affects displayed scalloping but does not change the
  underlying array aperture.
- A spacing-derived design frequency does not guarantee an alias-free response
  for every multidimensional geometry and look direction.
- A plotted response is normalized array response, not an absolute acoustic
  calibration.
- Curved-array results depend on element normals and the fixed 60-degree
  visibility cutoff.

The self-contained scripts in
[Basic geometry and beampattern examples](../examples/basic-arrays.md) exercise
line, panel, cylindrical, and spherical cases.
