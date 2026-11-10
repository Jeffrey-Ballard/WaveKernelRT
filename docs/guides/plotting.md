# Plotting and visualization

WaveKernelRT provides plotting helpers for channel spectra and PSDs,
beamformed spectra and PSDs, white-noise gain (WNG), bearing-time records
(BTR), and prepared two-dimensional arrays. The supported public plotting
namespace is:

```python
from wavekernelrt.core import analysis_images
```

The functions return Matplotlib objects without calling `plt.show()`. This
lets the caller change labels, titles, limits, or annotations and then either
display or save the figure.

## Choose a plotting function

The suffix on a spectral image function identifies the displayed plane. Its
first letter is the vertical row axis and its second letter is the horizontal
column axis. The dimension omitted from the suffix is selected or reduced.

| Product | Input shape | Displayed plane | Functions |
| --- | --- | --- | --- |
| Channel FFT | `(T, Q, N)` | Frequency × channel | `fft_spectrum_image_QN`, `fft_spectrum_image_NQ` |
| Channel FFT | `(T, Q, N)` | Frequency × time | `fft_spectrum_image_QT`, `fft_spectrum_image_TQ` |
| Channel FFT | `(T, Q, N)` | Channel × time | `fft_spectrum_image_NT`, `fft_spectrum_image_TN` |
| Beam FFT | `(Q, J, T)` | Frequency × beam | `bfft_spectrum_image_QJ`, `bfft_spectrum_image_JQ` |
| Beam FFT | `(Q, J, T)` | Frequency × time | `bfft_spectrum_image_QT`, `bfft_spectrum_image_TQ` |
| Beam FFT | `(Q, J, T)` | Beam × time | `bfft_spectrum_image_JT`, `bfft_spectrum_image_TJ` |
| Channel PSD | `(T, Q, N)` | Same six channel planes | `psd_image_QN`, `psd_image_NQ`, `psd_image_QT`, `psd_image_TQ`, `psd_image_NT`, `psd_image_TN` |
| Beam PSD | `(Q, J, T)` | Same six beam planes | `bpsd_image_QJ`, `bpsd_image_JQ`, `bpsd_image_QT`, `bpsd_image_TQ`, `bpsd_image_JT`, `bpsd_image_TJ` |
| WNG | `(Q, J, T)` | Frequency × beam snapshot | `wng_image_QJ` |
| WNG | `(Q, J, T)` | Time history at one frequency and beam | `wng_line_T` |
| Bearing-time record | `(T, A)` | Time × bearing | `btr_image` |
| Prepared scalar data | `(R, C)` | Arbitrary rows × columns | `image_data_RC` |

Here `T` is time or record, `Q` is frequency, `N` is channel, `J` is beam,
and `A` is bearing. See [Data shapes and naming](../concepts/data-shapes.md)
for the project-wide conventions.

All spectral and WNG wrappers expect three-dimensional input, including
singleton dimensions. For example, retain the time dimension when reading one
beam record:

```python
data_QJT, headings_T = bf_read_multiple_sequence(header, [sequence])
```

If data came from a single-record reader as `(Q, J)`, restore the singleton
time dimension with `data_QJ[:, :, None]` before plotting.

## Relative and geographic bearings

Beam axes produced by the processing pipeline are always relative to the
array. Post-beamforming readers return the corresponding per-record array
headings separately as `headings_T`; changing that heading metadata does not
change the stored beam data or order.

Geographic bearing is a plotting coordinate derived for record `idxT` as:

```python
geographic_bearing_A = (relative_bearing_A + headings_T[idxT]) % 360
```

Use the heading belonging to each record when headings vary over time. This
conversion changes plot coordinates or labels only. The underlying beam axis
remains relative; plotting code may reorder coordinates around the 0/360-degree
wrap when a monotonically increasing geographic axis is desired.

## Select or reduce the omitted dimension

Each spectral image omits one input dimension. The corresponding `idxT`,
`idxQ`, `idxN`, or `idxJ` argument selects zero-based positions along that
dimension. It accepts an integer, a `range`, or a one-dimensional integer
sequence. These are array positions, not timestamps, frequencies, channel
IDs, or beam angles.

Selection and reduction interact as follows:

| Index argument | `collapse_func` | Result |
| --- | --- | --- |
| Omitted | Omitted | Display the first position |
| Supplied | Omitted | Display the first selected position |
| Omitted | Supplied | Reduce all positions |
| Supplied | Supplied | Reduce only the selected positions |

Supported plotting reductions are:

| `collapse_func` | Meaning |
| --- | --- |
| `None` | Select the first applicable position |
| `"mean"` | Arithmetic mean in linear units |
| `"median"` | Median in linear units |
| `"min"` | Minimum |
| `"max"` | Maximum |
| `"mean_dB"` | Convert to decibels, average, and return to linear units before final display conversion |

Reduction happens before display conversion. Thus `collapse_func="mean"`
with a decibel display means “average the linear values, then convert that
average to decibels.” For spectrum phase, mean and median requests use circular
statistics so values around the phase wrap boundary are handled correctly.
Titles record the selected indices or reduction.

## Display controls

Inputs to the spectrum, PSD, and WNG wrappers are linear values. BTR values are
displayed directly because a BTR may already contain decibels or `uint8` ranks.

| Plot family | Important controls |
| --- | --- |
| FFT/BFFT spectrum | `plot_type="complex"`, `"amplitude"`, or `"phase"`; `amp_disp="dB"` or `"linear"`; `phs_disp="wrapped"` or `"unwrapped"` |
| PSD/BPSD | `mag_disp="dB"` or `"linear"` |
| WNG | `wng_disp="dB"` or `"linear"`; optional `constraint_wng` |
| BTR | Direct value display; `pixels_per_sample` controls initial logical-pixel sizing; `colorbar` is optional |

Spectrum amplitude limits use `amp_vmin` and `amp_vmax`; phase limits use
`phs_vmin` and `phs_vmax` in radians. PSD and ordinary WNG images use `vmin`
and `vmax` in the selected display units. Wrapped phase defaults to the range
0 through `2*pi`. Unwrapped phase follows the frequency axis after reduction;
a plot without frequency cannot unwrap phase and emits a warning before using
wrapped phase.

WNG constraint images show
`constraint_dB - achieved_WNG_dB`, clipped to -10 through 10 dB. Negative
values pass the constraint, positive values fail it, and zero lies exactly on
the constraint. `constraint_wng` is supplied in linear units as a scalar,
shape `(J,)`, or shape `(Q, J)`.

Common options include `title`, `cmap`, and physical coordinates such as
`freq`, `channels`, `beams`, and `times`. Missing coordinates use zero-based
positions. Time coordinates are used as supplied; set `time_label` to describe
seconds, minutes, or another chosen unit. Channel IDs label equally spaced
bins, while beam angles retain their physical spacing.

## Example: channel FFT through time

This displays amplitude and unwrapped phase for one channel. Frequency is
vertical, time is horizontal, and `idxN` chooses the omitted channel axis.

```python
fig, axes, meshes = analysis_images.fft_spectrum_image_QT(
    spectrum_TQN,
    plot_type="complex",
    freq=freq_Hz_Q,
    times=times_s_T,
    idxN=channel_index,
    amp_disp="dB",
    phs_disp="unwrapped",
    time_label="Seconds from first spectrum",
)

fig.savefig("channel_fft.png", dpi=200)
```

Complex spectrum plots return amplitude and phase axes and meshes in that
order. Single-component spectrum, PSD, WNG, BTR, and generic scalar images
return one axis and one mesh. `wng_line_T` returns its WNG trace as a line.

## Public lower-level drawing functions

The lower-level drawing functions are public for applications that need custom
wrappers:

- `image_data_RC` draws an arbitrary prepared scalar plane and optionally adds
  a colorbar.
- `create_spectrum_image` converts prepared amplitude and phase components to
  display units and draws them on supplied axes.
- `create_psd_image` converts prepared linear PSD values and draws them on a
  supplied axis.
- `set_beam_ticks` applies 30-degree angular ticks without expanding current
  plot limits.

These functions do not perform the high-level axis selection and reduction
done by the named FFT, BFFT, PSD, and BPSD wrappers.

See the [Plotting API reference](../reference/api/plotting.md) for complete
signatures and parameter behavior. Runnable applications are cataloged in the
[SWellEx-96 examples](../examples/swellex96.md).
