# Pipeline architecture

WaveKernelRT processes file-backed acoustic data as a linear stream of typed
blocks. A pipeline has one source, an ordered sequence of zero or more stages,
and one sink:

```text
InputConfig -> source -> stage -> stage -> ... -> sink <- OutputConfig
                    \---- blocks flow left to right ----/
```

The pipeline is intentionally linear. `TeeConfig` can write an intermediate
product while allowing the same block to continue, but it does not create an
independently scheduled branch.

## Configuration versus runtime objects

The public configuration layer consists of frozen dataclasses in
`wavekernelrt.pipeline.configs`. A `PipelineConfig` describes the input file,
ordered stage configurations, and output file without containing open files or
processing buffers.

`build_pipeline(config)` turns that description into a `BuiltPipeline`:

| Configuration object | Runtime object |
| --- | --- |
| `InputConfig` | File source |
| Each stage config | Processing stage, sometimes with an internal buffer |
| `OutputConfig` | File sink |
| `PipelineConfig` | `BuiltPipeline(source, stages, sink)` |

Building is not a pure validation operation. It opens the input file, constructs
the stages, opens the final output file, and opens any tee outputs. Opening a
sink creates or overwrites its target file. If construction fails, resources
opened during the build are closed, although an output file that was already
created may remain.

`PipelineDataBlock` is an internal build-time metadata object. Despite its name,
it is not passed through runtime stages. The builder mutates it while tracking
the current logical product type, frequency grid, channels, timebase, shading,
steering grid, and output-header values.

## Runtime interfaces

The runtime follows three small protocols:

```python
Source:  __iter__() -> Iterator[Block]
Stage:   process(block) -> Iterable[Block]
Sink:    write(block); close()
```

`run_pipeline()` iterates over source blocks and lazily passes each stream
through every stage. A stage may yield:

- no blocks while it fills a buffer,
- one block for a one-to-one transform, or
- several blocks when one input contains enough data for multiple outputs.

Every block leaving the final stage is written to the sink before the source
iteration continues. The runner is synchronous and single-process; it does not
provide a queue, worker pool, or backpressure mechanism beyond ordinary Python
iteration.

## Runtime blocks

The logical product type is tracked separately from the Python block class.
Complex spectra and real PSD values deliberately share the same container.

| Block class | Data | Shape | Logical products |
| --- | --- | --- | --- |
| `TimeBlock` | `data_NT` / `data_JT` | `(N, T)` / `(J, T)` | `ts`, `bts` |
| `FFTBlock` | `data_QN` | `(Q, N)` | `fft`, `psd` |
| `BFBlock` | `data_QJ` | `(Q, J)` | `bfft`, `bpsd` |
| `BTRBlock` | `data_BAE` | `(B, A, E)` | `btr` |

`FFTBlock` and `BFBlock` carry `advance` in samples. All runtime blocks carry a
zero-based `sequence` value. Post-beamforming blocks also carry the per-record
`array_heading_deg`; beam axes remain relative. Blocks do not currently carry
an absolute timestamp or `Timebase`; timing is prepared by the builder and
written to file headers.
See [Conventions](conventions.md#runtime-block-timing) for the exact timestamp
rules.

## Stage compatibility

Compatibility is validated in stage order while the pipeline is built. The
type names below are logical product types, not merely filename extensions.

| Stage configuration | Accepted input | Produced output |
| --- | --- | --- |
| `ChannelCropConfig` | `ts`, `fft`, `psd` | Same type |
| `TStoFFTConfig` | `ts`, `bts` | `fft`, `bfft` respectively |
| `FFTtoTSConfig` | `fft`, `bfft` | `ts`, `bts` respectively |
| `FreqCropConfig` | `fft`, `psd`, `bfft`, `bpsd` | Same type |
| `FFTtoPSDConfig` | `fft`, `bfft` | `psd`, `bpsd` respectively |
| `FFTtoCBFConfig` | `fft` | `bfft` |
| `DMRElementConfig` | `fft` | `bfft` |
| `PSDAvgConfig` | `psd`, `bpsd` | Same type |
| `BandEnergyConfig` | `bpsd` | `btr` |
| `TeeConfig` | Any currently writable type | Same type |

The final `OutputConfig.type` and filename extension must both match the type
produced by the last stage. A tee output must similarly match the type at the
tee's exact position.

Common valid chains include:

```text
ts  -> TStoFFT -> fft
fft -> FFTtoTS -> ts
ts  -> TStoFFT -> FFTtoPSD -> PSDAvg -> psd
ts  -> TStoFFT -> FFTtoCBF/DMR -> FFTtoTS -> bts
ts  -> TStoFFT -> FFTtoCBF/DMR -> FFTtoPSD -> PSDAvg -> BandEnergy -> btr
bts -> TStoFFT -> bfft
```

The builder does not implicitly insert conversion stages. For example, a PSD
cannot be sent to `FFTtoTSConfig`, and channel FFT data cannot be written as a
beam-frequency file without a beamforming stage.

## File sources and sinks

Current file sources support:

| Input type | Runtime block | Notes |
| --- | --- | --- |
| `ts` | `TimeBlock` | Channel time series |
| `bts` | `TimeBlock` | Beam time series via the `data_JT` alias |
| `fft` | `FFTBlock` | Complex channel spectra |
| `bfft` | `BFBlock` | Complex beam spectra |
| `bpsd` | `BFBlock` | Real beam PSD |

There is currently no source for `.psd` or `.btr` files, even though the
pipeline can write both formats. Those files can be read with their direct
file-utility functions but cannot begin a configured pipeline.

Sinks support `ts`, `bts`, `fft`, `psd`, `bfft`, `bpsd`, and `btr`. They write a
header with an initial record count of zero, count successfully written blocks,
and update the header count when closed.

## Buffering and trailing data

Several stages are stateful:

- `TStoFFT_Stage` accumulates samples and emits every complete `nfft` frame,
  retaining overlap according to `advance`.
- `FFTtoTS_Stage` reconstructs samples and emits fixed `Npt_per_seq` records.
- `PSDAvg_Stage` emits one record for each complete averaging group.
- `BFtoBandEnergy_Stage` emits immediately without background time averaging.
  With a configured background time window, it fills that window before its
  first output and then produces a sliding output for each subsequent record.
- `FFTtoDMRElement_Stage` uses CBF during covariance startup, then changes to
  DMR after the snapshot count derived from `Dmax` and `sinr_loss_dB` has
  initialized the covariance estimate.

Stages do not currently have an end-of-stream flush protocol. Incomplete FFT
frames, inverse-FFT output records, and PSD averaging groups remaining when the
source ends are discarded. This behavior is significant when comparing input
and output record counts.

## Timebase propagation

The builder creates one `Timebase` from the input header and adjusts the first
output timestamp as stages are added:

- TS/BTS input timestamps identify the first sample.
- FFT/BFFT/BPSD input timestamps identify the center of the first frame.
- TS-to-FFT moves the first timestamp forward by `nfft / 2` samples.
- FFT-to-TS moves it back by `nfft / 2` samples.
- PSD averaging selects the midpoint of the grouped input centers and multiplies
  `advance` by the group size.
- Background-normalized band energy selects the middle record of its first time
  window; even windows use the later middle record.

Frequency and channel crops do not change timing. CBF, DMR, FFT-to-PSD, and tee
stages preserve the incoming timestamp and advance.

## Channel selection and geometry

`ChannelCropConfig` uses zero-based positions in the source data, not stored
channel labels. The builder retains the corresponding original source indices.
When a later CBF or DMR stage is built, it checks that the supplied full
geometry has one element per original source channel, selects the retained
elements, and recomputes the selected phase center and shading.

For panels, cylinders, and spheres, a selected subarray must contain every
combination of its selected rows and columns in the original row-fastest order.
Arbitrary sparse curved-array subsets are rejected. At least two channels must
remain for beamforming.

## Tees and cleanup

A `TeeConfig` builds a sink using the metadata available at that point. At
runtime, `TeeStage` writes each block to that sink and yields the unchanged
block downstream. The tee sink is closed with the other stages.

`run_pipeline()` uses an `ExitStack` so stages with `close()`, the final sink,
and the source are closed even when processing raises an exception. Sink
`close()` operations are idempotent. Users who build a pipeline but do not pass
it to `run_pipeline()` are responsible for closing its source and sinks.

## Extension boundary

The runtime protocols are intentionally small, but `build_pipeline()` only
recognizes the configuration classes listed above. A new configured stage
therefore requires:

1. a configuration dataclass,
2. a runtime stage whose `process()` yields an iterable of blocks,
3. builder logic that validates and updates `PipelineDataBlock`, and
4. tests for type compatibility, metadata propagation, buffering, and cleanup.

A custom source, stage, and sink may also be composed directly and passed to
`run_pipeline()` without using the configuration builder.
