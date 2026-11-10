# Configuration reference

Pipeline configurations are frozen dataclasses from
`wavekernelrt.pipeline.configs`. Use `dataclasses.replace()` to derive a modified
configuration rather than assigning to a field.

```python
from dataclasses import replace

updated_fft = replace(original_fft, advance=1024)
```

Configuration validation occurs in two places. Some scalar rules are checked
when a dataclass is constructed; compatibility with preceding stages and
input-derived metadata is checked by `build_pipeline()`. Building opens files
and creates output files, so it should not be used as a side-effect-free config
checker.

## Pipeline-level configuration

### `InputConfig`

| Field | Type | Meaning |
| --- | --- | --- |
| `file` | `Path` | Existing input file |
| `type` | `str` | Input product type |

Accepted input types are `ts`, `bts`, `fft`, `bfft`, and `bpsd`, optionally
with a leading period. Use lowercase and make the value agree with the filename
extension. `.psd` and `.btr` cannot currently serve as configured sources.

### `OutputConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `file` | `Path` | Required | Output file; an existing file is overwritten when the sink opens |
| `type` | `str` | Required | `ts`, `bts`, `fft`, `psd`, `bfft`, `bpsd`, or `btr` |
| `check_finite` | `bool` | `False` | Check each written block for NaN or infinite values |

The type is case-insensitive and may have a leading period. Its normalized value
must match both the filename extension and the logical type produced by the
pipeline. Finite-value checking adds a full-array check at each write and may
cost time on large products.

### `PipelineConfig`

| Field | Type | Meaning |
| --- | --- | --- |
| `input` | `InputConfig` | Source description |
| `stages` | `tuple[StageConfig, ...]` | Ordered stage descriptions; an empty tuple is allowed |
| `output` | `OutputConfig` | Final sink description |

An empty stage tuple copies a supported input product to the matching output
format. The input and output files should be different paths; the builder opens
the input and output during the same operation and does not protect against
overwriting the input path.

A complete TS-to-PSD configuration looks like this:

```python
from pathlib import Path

from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.configs import (
    FFTtoPSDConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PSDAvgConfig,
    TStoFFTConfig,
)
from wavekernelrt.pipeline.run_pipeline import run_pipeline

config = PipelineConfig(
    input=InputConfig(file=Path("input.ts"), type="ts"),
    stages=(
        TStoFFTConfig(nfft=4096, advance=2048, window_name="tukey"),
        FFTtoPSDConfig(),
        PSDAvgConfig(time_to_avg_secs=5.0),
    ),
    output=OutputConfig(file=Path("output.psd"), type="psd"),
)

pipeline = build_pipeline(config)
run_pipeline(pipeline.source, pipeline.stages, pipeline.sink)
```

## Transform and selection stages

### `TStoFFTConfig`

Converts channel or beam timeseries into one-sided spectra.

| Field | Type | Default | Unit and rule |
| --- | --- | --- | --- |
| `nfft` | positive integer | `4096` | Samples per FFT; determines `Q = floor(nfft/2) + 1` |
| `advance` | positive integer | `2048` | Samples between frame centers; must not exceed `nfft` |
| `window_name` | `str` | `"tukey"` | `tukey`, `boxcar`, `hann`/`hanning`, or `hamming` |

Window names are case-insensitive. The Tukey alpha is
`1 - advance / nfft`; the other tapers are symmetric. Before application, the
window is divided by the square root of its mean-square value. This gives it
unit power gain and preserves stationary broadband power in expectation. The
normalized `float32` coefficients actually applied are stored in FFT and PSD
headers. A sinusoid peak still requires ENBW correction; see
[Spectral normalization](../concepts/spectral-normalization.md).

`ts` becomes `fft`, while `bts` becomes `bfft`.

### `FFTtoTSConfig`

Reconstructs time series from complex channel or beam spectra. Values left as
`None` inherit the current pipeline metadata.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nfft` | `int | None` | `None` | Original FFT length |
| `advance` | `int | None` | `None` | Samples contributed by each spectral frame |
| `Npt_per_seq` | `int | None` | `None` | Samples in each output record; defaults to `advance` |
| `window_T` | array-like or `None` | `None` | Length-`nfft` analysis window; inherited when omitted |

The stage restores retained frequencies to their original FFT bins and fills
missing bins with zero, so inverse transformation remains possible after
frequency cropping. It divides out nonzero window samples and writes zero where
the original window is zero. PSD input is rejected because phase has been lost.

### `FreqCropConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `band_to_keep_Hz` | two-value tuple | `None` | Inclusive lower and upper frequency bounds in hertz |

Although the field defaults to `None`, a two-value finite real tuple is required
when the stage is used. Bounds must be ordered and lie inside the current
frequency grid. Every existing bin between the bounds is retained. The stage
accepts `fft`, `psd`, `bfft`, and `bpsd` without changing the logical type.

### `ChannelCropConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `idx_channels_to_keep` | 1-D integer array-like | `None` | Nonempty zero-based source-column positions |

`None` is not valid when the stage is used. Indices must be in range. For line
arrays they may select or reorder arbitrary elements, although unique ascending
indices are the clearest convention. Panel, cylinder, and sphere selections
must form a complete rectangular row/column subset in original order if a later
beamformer uses the geometry.

Channel cropping accepts `ts`, `fft`, and `psd`. It does not crop beam products.

### `FFTtoPSDConfig`

This configuration has no fields. It converts complex squared magnitude to
single-sided PSD using the FFT integration period `nfft / sample_rate_hz`:

```text
fft  -> psd
bfft -> bpsd
```

The result is `float32`. The preceding TS-to-FFT stage has already
power-normalized its analysis window, so this stage applies no additional
window correction.

### `PSDAvgConfig`

Exactly one averaging field must be supplied.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `seq_to_avg` | positive integer or `None` | `None` | Number of adjacent PSD records per nonoverlapping average |
| `time_to_avg_secs` | positive finite real or `None` | `None` | Requested duration, converted to a rounded record count |

The time form is divided by the current record spacing
`advance / sample_rate_hz` and rounded to the nearest integer. A duration that
rounds below one record is rejected. The output `advance` is multiplied by the
group size. Only complete groups are emitted.

Forward and inverse transforms, PSD scaling, cropping, averaging, and timestamp
behavior are described together in
[Spectral processing](../concepts/spectral-processing.md).

## Beamforming configuration

### `ShadingConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `window_type` | `str` | `"taylor"` | `uniform`, `hamming`, or `taylor` |
| `sll_dB` | positive finite real | `30.0` | Positive Taylor sidelobe attenuation in decibels |
| `nbar` | integer >= 2 | `4` | Taylor near-in sidelobe count parameter |

Window names are currently case-sensitive. Taylor parameters are ignored for
uniform and Hamming shading. Line and panel arrays use one shared element
vector. Cylinders and spheres recompute a visible, normalized shading vector
for every steering direction.

### `FFTtoCBFConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `geom` | `Geometry` | Required | Full geometry corresponding to the original source channels |
| `shading` | `ShadingConfig` | Required | Spatial window definition |
| `steer_rbe_rad_J3` | array-like `(J, 3)` | Required | Range in meters; relative bearing and elevation in radians |
| `output_file` | `Path | None` | `None` | Reserved; currently unused by the builder |
| `heading_provider` | callable or `None` | `None` | Supplies per-record array-heading metadata in degrees clockwise from true north; otherwise use the block heading or geometry heading |

The stage accepts channel `fft` input and produces `bfft`. The source must have
at least two channels. The full geometry's element count must equal the original
source-channel count; the builder applies any preceding channel crop to the
geometry before it computes steering replicas and shading.

Pipeline steering bearings are always relative to array heading.
`heading_provider` changes only the heading stored with each output record; it
does not rotate `steer_rbe_rad_J3` or rebuild the beamforming weights.

### `DMRElementConfig`

DMR has the same `geom`, `shading`, and `steer_rbe_rad_J3` fields as CBF, plus:

| Field | Type | Unit or meaning |
| --- | --- | --- |
| `Dmax` | positive integer | Maximum retained dominant-mode count and design rank used to derive the covariance training reference |
| `Dmin` | positive integer | Minimum retained dominant-mode count; must not exceed `Dmax` |
| `wngc_val_dB` | finite real | White-noise-gain constraint target in decibels |
| `wngc_tol_dB` | nonnegative finite real | White-noise-gain constraint tolerance in decibels |
| `eig_thresh` | finite real in `(0, 1]` | Cumulative eigen-energy threshold used in mode selection |
| `sinr_loss_dB` | positive finite real, default `1.0` | Nominal SINR-loss target used to calculate the covariance training snapshots |
| `write_weights` | Boolean, default `False` | Write the coefficients used for every DMR-stage output record to a BFW sidecar file |
| `weight_band_Hz` | two-value tuple or `None`, default `None` | Inclusive stored frequency band; `None` stores every current frequency bin |
| `output_weight_file` | `Path | None`, default `None` | Explicit `.bfw` destination; when omitted, derive it from the final output and input paths |
| `weight_seq_stride` | positive integer, default `1` | Store weights for DMR input sequences `0, stride, 2*stride, ...` |
| `write_wng` | Boolean, default `False` | Write achieved WNG to a sidecar whose header includes `wngc_val_dB` and `wngc_tol_dB` |
| `wng_band_Hz` | two-value tuple or `None`, default `None` | Inclusive stored frequency band; `None` stores every current frequency bin |
| `output_wng_file` | `Path | None`, default `None` | Explicit `.wng` destination; when omitted, derive it from the final output and input paths |
| `wng_seq_stride` | positive integer, default `1` | Store WNG for DMR input sequences `0, stride, 2*stride, ...` |
| `heading_provider` | callable or `None`, default `None` | Supplies per-record array-heading metadata in degrees clockwise from true north; otherwise use the block heading or geometry heading |

DMR accepts channel `fft` and produces `bfft`. It requires at least two active
elements for every beam. The startup count is
`ceil(Dmax/(1 - 10**(-sinr_loss_dB/10)))`; during those records the stage emits
CBF output while initializing the covariance matrix, and later records use
DMR. The steady-state exponential coefficient is `2/(snapshots + 1)` so its
variance-equivalent sample count matches the startup reference under independent
stationary sampling.

Weight recording includes the CBF weights used during covariance training and
any CBF fallback used for a zero-energy covariance group. Adaptive records
contain the already-conjugated DMR multiplication coefficients documented in
[Beamformer weights](file-formats.md#beamformer-weights-bfw). When
`weight_band_Hz` is set, DMR still processes and emits every current frequency;
only the BFW sidecar is cropped. Its bounds are inclusive, must lie within the
current spectrum, and must select at least one FFT bin.

For `complex64`, the record payload is
`8 * Nstored_freq * Jbeams * Nchan` bytes per sequence, so restricting
`weight_band_Hz` reduces file size in direct proportion to the selected-bin
count.

WNG recording is independent of weight recording. WNG uses a dedicated file
layout with `float32` `(Q, J)` records and stores `wngc_val_dB` and
`wngc_tol_dB` in its header. Values are dimensionless linear WNG,
`1 / (w^H w)`, for the coefficients actually used. Startup and zero-energy
fallback records therefore contain CBF WNG, while adaptive records contain the
achieved DMR WNG. `wng_band_Hz` limits only this sidecar and follows the same
inclusive validation rules as `weight_band_Hz`.

The two sequence strides are independent. A stride of `10` stores source DMR
sequences `0, 10, 20, ...`. Each sidecar retains the center timestamp of source
sequence zero and stores `advance = source_advance * stride`. Consequently,
stored record `k` has center time
`start_unix_nanoseconds + k * advance / sample_rate_hz`, so readers can recover
the correct time axis without knowing which records were skipped.

The weight path is resolved in this order:

1. Use `output_weight_file` when it is provided.
2. When the final pipeline output is BFFT or BPSD, replace its suffix with
   `.bfw`; for example, `beams.bfft` becomes `beams.bfw`.
3. For any other final product, use `<input-stem>_dmr.bfw` in the final output
   directory; for example, input `event.fft` and final output
   `results/detections.btr` produce `results/event_dmr.bfw`.

WNG follows the same priority using `.wng`: an explicit `output_wng_file`,
then replace the suffix of a final BFFT/BPSD output (`beams.bfft` becomes
`beams.wng`), otherwise use `<input-stem>_dmr.wng` in the final output
directory.

The implemented algorithm is described in [Algorithm references](../references.md).

## Tee configuration

### `TeeConfig`

| Field | Type | Meaning |
| --- | --- | --- |
| `output` | `OutputConfig` | Intermediate file sink |

The tee output type and extension must match the logical type at its position in
the stage tuple. Each incoming block is written and then yielded unchanged. For
example, a tee immediately after `TStoFFTConfig` in a channel pipeline must use
an `.fft` output; a tee after CBF or DMR must use `.bfft`.

## Band-energy configuration

### `BackgroundEstimateConfig`

This optional configuration defines the background used to normalize beam PSD
before band integration. Each operation can be enabled independently.

| Field | Type | Default | Unit or meaning |
| --- | --- | --- | --- |
| `bkgd_time_seq_avg` | positive integer or `None` | `None` | Records in the sliding time-average window |
| `bkgd_sector_beam_size` | positive integer or `None` | `None` | Bearing-neighborhood width in beams |
| `full_circle_flag` | `bool` | `False` | Wrap bearing filtering across the circular seam |
| `freq_smoothing_window` | 1-D array-like or `None` | `None` | Frequency-axis convolution weights |
| `car_parms_WP` | list of `(side, guard, threshold_dB)` tuples or `None` | `None` | Sequential frequency-axis clip-and-replace definitions: sideband width per side, excluded central width, and upper-peak threshold in dB |

Processing optionally averages linear power over time, converts to decibels,
applies clip-and-replace filters, smooths across frequency, and takes a
bearing-neighborhood median. A time window delays the first output until the
window is full; the later middle record supplies the timestamp for an even
window.

Repeated observations, nearby beams, and nearby-frequency background data are
motivated by Abraham (2019), Section 9.3.1, pp. 683–685. That reference does
not prescribe this configuration's decibel-domain smoothing, bearing median,
or clip-and-replace rules. See the
[noise-normalized detector reference mapping](../references.md#abraham-2019).

Each clip-and-replace filter compares a bin with the mean of its two guarded
frequency sidebands and replaces the bin only when it exceeds that mean by
more than `threshold_dB`. These are Hampel-like decision filters, but they do
not use the median/MAD rule of a standard Hampel filter. See [Relationship to
Hampel
filtering](../concepts/band-energy-detection.md#relationship-to-hampel-filtering).

### `BandEnergyConfig`

Band-energy processing accepts beam PSD (`bpsd`) and produces bearing-time
records (`btr`).

| Field | Type | Default | Unit or meaning |
| --- | --- | --- | --- |
| `bands_B2` | list of `(low, high)` pairs | Required | Inclusive frequency bands in hertz |
| `background` | `BackgroundEstimateConfig` or `None` | `None` | Optional PSD background normalization |
| `output` | `"float32"` or `"rank_uint8"` | `"float32"` | Stored output representation |

Without a background, each band is the PSD integral weighted by frequency-bin
width. With a background, each PSD bin is divided by its estimated background
in linear power before integration. Float32 output stores `10*log10` of the
integrated value: direct output is relative to one input power unit and
normalized output is relative to 1 Hz. Ranked output maps azimuth ranks to the
full uint8 range independently for every band and elevation.

The normalization-before-integration order follows Abraham (2019), Section
9.3, p. 683, Eqs. (9.190)–(9.191). WaveKernelRT's bin-width integration and
rank encoding are implementation choices; the citation does not imply an
exact false-alarm calibration.

The stage supports exactly one steering range while preserving all configured
azimuths and elevations. Every band must contain at least one frequency bin.
See [Band-energy detection](../concepts/band-energy-detection.md) and the
[SWellEx-96 Energy Detect example](../examples/swellex96.md#energy-detect-bearing-time-records).

## Compatibility summary

| Configuration | Input | Output |
| --- | --- | --- |
| `ChannelCropConfig` | `ts`, `fft`, `psd` | Same |
| `TStoFFTConfig` | `ts`, `bts` | `fft`, `bfft` |
| `FFTtoTSConfig` | `fft`, `bfft` | `ts`, `bts` |
| `FreqCropConfig` | `fft`, `psd`, `bfft`, `bpsd` | Same |
| `FFTtoPSDConfig` | `fft`, `bfft` | `psd`, `bpsd` |
| `FFTtoCBFConfig` | `fft` | `bfft` |
| `DMRElementConfig` | `fft` | `bfft` |
| `PSDAvgConfig` | `psd`, `bpsd` | Same |
| `BandEnergyConfig` | `bpsd` | `btr` |
| `TeeConfig` | Any writable current type | Same |

For lifecycle, buffering, timestamp propagation, and extension rules, see
[Pipeline architecture](../concepts/pipeline-architecture.md).
