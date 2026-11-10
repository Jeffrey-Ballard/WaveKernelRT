# Spectral normalization

WaveKernelRT uses power-normalized analysis windows for time-series-to-FFT
processing. This convention makes integrated PSD levels directly useful while
retaining enough information in each FFT or PSD header to recover sinusoid
levels.

For FFT framing, one-sided transform scaling, PSD conversion, temporal
averaging, and inverse reconstruction, see
[Spectral processing](spectral-processing.md).

## Analysis-window normalization

For an unnormalized length-`N` window `w`, define its power gain as

```text
G_power = mean(w**2) = sum(w**2) / N
```

Before applying the window, `TStoFFT_Stage` constructs

```text
w_power = w / sqrt(G_power)
```

so that

```text
mean(w_power**2) = 1
```

The normalized `float32` coefficients in `w_power` are both applied to the time
series and stored as `window_T` in FFT, PSD, BFFT, and BPSD headers. Reading the
header therefore gives the exact window used by the forward transform.

This normalization preserves the mean-square level of stationary broadband
signals in expectation. Consequently, a one-sided PSD can be integrated as

```text
power = sum(PSD) * df
```

where `df = sample_rate_hz / nfft`. No separate window-power correction is
needed. Restrict the sum to a frequency band when band-limited power is wanted.

## Equivalent noise bandwidth

Compute the equivalent noise bandwidth directly from the stored normalized
window:

```text
window_mean = mean(window_T)
ENBW_bins = mean(window_T**2) / window_mean**2
ENBW_hz = ENBW_bins * df
```

Because `window_T` is power-normalized, `mean(window_T**2)` is approximately
one, but using the stored values in the full expression accounts for `float32`
rounding.

For a symmetric Hann window, `ENBW_bins` approaches `1.5` as `N` grows. It is
slightly larger at finite `N`; for example, a 4096-point symmetric Hann window
has an ENBW of approximately `1.500366` bins.

## Recovering a sinusoid amplitude

For an isolated, bin-centered sinusoid away from DC and Nyquist, recover peak
amplitude from its one-sided PSD peak with

```text
A_peak = sqrt(2 * PSD_peak * df * ENBW_bins)
```

For a power-normalized Hann window this is commonly written

```text
A_peak = sqrt(2 * PSD_peak * df * 1.5)
```

The factor of two converts the one-sided sinusoid power to squared peak
amplitude. DC and Nyquist require different handling because those bins are not
doubled by the one-sided transform.

An off-bin sinusoid leaks into neighboring bins, so the peak-bin formula has
scalloping error. For the pure-tone case, integrate enough bins to include its
leakage and compute

```text
A_peak = sqrt(2 * sum(PSD_tone_bins) * df)
```

This integrated estimate also includes any noise or other signals in the
selected bins. The windowing examples under `examples/windowing/` demonstrate
both estimates.
