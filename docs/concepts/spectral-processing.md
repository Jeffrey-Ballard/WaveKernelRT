# Spectral processing

WaveKernelRT converts channel or beam time series into one-sided complex
spectra, optionally crops the frequency grid, converts spectra to power
spectral density (PSD), and averages PSD records in time. Complex spectra can
also be transformed back into time series.

The common processing paths are:

```text
ts  -> FFT -> fft  -> PSD -> psd  -> time average -> psd
bts -> FFT -> bfft -> PSD -> bpsd -> time average -> bpsd

fft  -> inverse FFT -> ts
bfft -> inverse FFT -> bts
```

Channel spectra and PSD use shape `(Q, N)`. Beam spectra and PSD use shape
`(Q, J)`, where `Q` is frequency, `N` is channel, and `J` is beam. See
[Data shapes and naming](data-shapes.md) for the complete suffix convention.

## Forward FFT framing

For sample rate $f_s$, FFT length $N_{FFT}$, and advance $H$:

$$
\Delta f = \frac{f_s}{N_{FFT}}, \qquad
Q = \left\lfloor\frac{N_{FFT}}{2}\right\rfloor + 1.
$$

Each frame contains `nfft` consecutive samples. Successive frame starts are
separated by `advance` samples, so the overlap is

$$
N_{overlap} = N_{FFT} - H.
$$

`advance` must be positive and cannot exceed `nfft`; the current stage does
not create gapped frames. Frames are emitted as soon as the internal buffer
contains `nfft` samples. Samples remaining in an incomplete final frame are
not emitted.

The first spectral timestamp is the center of the first frame, implemented as
`nfft / 2` samples after the first time-series sample. Later timestamps are
separated by `advance / sample_rate_hz` seconds. Spectral sequence numbers
start at zero and increase by one per frame.

## Analysis windows

`TStoFFTConfig.window_name` selects one of these symmetric windows:

| Name | Behavior |
| --- | --- |
| `boxcar` | No taper |
| `tukey` | Tukey alpha is `1 - advance / nfft` |
| `hann` or `hanning` | Hann taper with zero endpoints |
| `hamming` | Hamming taper with nonzero endpoints |

Before application, every window is normalized to unit mean-square and
converted to `float32`. The exact normalized coefficients applied to the
samples are stored in spectral-product headers. The normalization, equivalent
noise bandwidth (ENBW), and sinusoid-amplitude recovery rules are described in
[Spectral normalization](spectral-normalization.md).

## One-sided complex spectrum

For power-normalized window $w[t]$, WaveKernelRT first calculates

$$
Y_0[q] = \frac{1}{f_s}
\sum_{t=0}^{N_{FFT}-1} x[t]w[t]
\exp\left(-i\frac{2\pi qt}{N_{FFT}}\right).
$$

It then multiplies bins with a distinct negative-frequency partner by
$\sqrt{2}$. DC is never doubled. For an even `nfft`, Nyquist is also not
doubled; for an odd `nfft`, every bin above DC is doubled.

The stored complex value therefore has the input unit multiplied by seconds.
The scaling preserves the contribution of the omitted negative frequencies
when squared magnitude is later converted to a one-sided PSD.

## PSD conversion

`FFTtoPSDConfig` converts complex spectra to real `float32` PSD using the frame
integration period

$$
T = \frac{N_{FFT}}{f_s}
$$

and

$$
P[q] = \frac{|Y[q]|^2}{T}.
$$

If the input time series has unit `U`, PSD has unit `U^2/Hz`. The PSD scaling
uses the FFT frame duration, not the frame advance. Overlap changes record
spacing and statistical dependence, but it does not change the scale of an
individual PSD estimate.

Because the analysis window has unit mean-square, broadband mean-square level
is recovered in expectation by

$$
x_{rms}^2 \approx \sum_q P[q]\,\Delta f.
$$

The same expression can be restricted to selected frequency bins for
band-limited power. A discrete sinusoid requires either ENBW correction of its
peak bin or integration over its leaked main response, as described in
[Spectral normalization](spectral-normalization.md#recovering-a-sinusoid-amplitude).

PSD conversion discards phase and is not reversible.

## Frequency cropping

`FreqCropConfig` retains every existing frequency bin between two inclusive
bounds. The requested bounds must lie inside the current grid and contain at
least one bin. Bounds that fall between bins select the first bin at or above
the lower bound and the last bin at or below the upper bound.

Cropping does not change `nfft`, sample rate, frame advance, or frequency-bin
spacing. It can be applied to `fft`, `bfft`, `psd`, or `bpsd` products.

Complex cropped spectra can still be inverse transformed: the inverse stage
restores retained values to their original FFT-bin positions and fills all
missing bins with zero. The result is consequently a band-limited time series.

## Temporal PSD averaging

`PSDAvgConfig` forms a nonoverlapping arithmetic mean of $K$ adjacent PSD
records:

$$
\overline{P}_m[q] = \frac{1}{K}
\sum_{k=0}^{K-1} P_{mK+k}[q].
$$

Set $K$ directly with `seq_to_avg`, or request a duration with
`time_to_avg_secs`. The builder converts a requested duration to the nearest
integer record count using the current record spacing:

$$
K = \operatorname{round}\left(
\frac{t_{avg}}{H/f_s}
\right).
$$

A request that rounds below one record is rejected. Only complete groups are
emitted, so an incomplete group at the end of the stream is discarded.

The averaged output advance is $KH$, and output sequence numbers restart at
zero. Its first timestamp is the midpoint of the input record centers in the
first group. For even $K$, that midpoint lies halfway between the two middle
input centers.

When the original FFT frames overlap, adjacent PSD estimates are correlated.
Averaging $K$ records still computes the arithmetic mean above, but it does not
necessarily provide $K$ statistically independent estimates.

## Inverse FFT reconstruction

`FFTtoTSConfig` accepts complex `fft` or `bfft` data. The inverse stage:

1. Maps stored frequencies back to the original one-sided FFT bins.
2. Fills missing bins with zero.
3. Removes the forward transform's $\sqrt{2}$ one-sided scaling.
4. Applies `irfft` and multiplies by the sample rate.
5. Divides by each nonzero analysis-window coefficient.
6. Retains the center portion contributed by each frame and buffers it into
   output time-series records.

The first reconstructed frame also supplies the samples before the first
center section so output begins at the original start time. Later frames
normally contribute `advance` central samples. `Npt_per_seq` controls the
number of samples written in each output record and defaults to `advance`.

An exact round trip requires the original complex spectrum, FFT length,
advance, sample rate, frequency grid, and applied window. Important limitations
are:

- PSD cannot be inverted because its phase has been discarded.
- Cropped spectra reconstruct a filtered signal because omitted bins are zero.
- Samples where the analysis window equals zero cannot be recovered and are
  written as zero. This affects Hann endpoints.
- Incomplete trailing input or output buffers are not flushed as padded
  records.
- Spectral modifications can make division by small window coefficients
  sensitive near frame edges.

## Configuration example

```python
from wavekernelrt.pipeline.configs import (
    FFTtoPSDConfig,
    FreqCropConfig,
    PSDAvgConfig,
    TStoFFTConfig,
)

stages = (
    TStoFFTConfig(nfft=4096, advance=2048, window_name="hann"),
    FreqCropConfig(band_to_keep_Hz=(10.0, 800.0)),
    FFTtoPSDConfig(),
    PSDAvgConfig(time_to_avg_secs=5.0),
)
```

The exact configuration fields, validation rules, and compatible input types
are listed in the [Configuration reference](../reference/configuration.md#transform-and-selection-stages).
