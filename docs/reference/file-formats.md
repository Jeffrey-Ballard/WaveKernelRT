# Binary file formats

WaveKernelRT uses compact binary files for time-series, spectral,
beam-frequency, and bearing-time products. This page documents the layouts
implemented by the current readers and writers in `wavekernelrt.core`.

These formats are experimental and have no compatibility guarantee. Every
format begins with an explicit format-version field; the first publicly
documented layouts are version 1. Preserve the software version or commit used
to create important datasets.

## Format families

| Extension | Product | Header utility | Record shape | Pipeline dtype |
| --- | --- | --- | --- | --- |
| `.ts` | Channel time series | `ts_file_utils` | <code style="white-space: nowrap;">(Nchan,Npt_per_seq)</code> | `float32` |
| `.bts` | Beam time series | `bts_file_utils` | <code style="white-space: nowrap;">(Jbeams,Npt_per_seq)</code> | `float32` |
| `.fft` | Channel spectrum | `fft_file_utils` | <code style="white-space: nowrap;">(Nfreq,Nchan)</code> | `complex64` |
| `.psd` | Channel PSD | `fft_file_utils` | <code style="white-space: nowrap;">(Nfreq,Nchan)</code> | `float32` |
| `.bfft` | Beam spectrum | `bf_file_utils` | <code style="white-space: nowrap;">(Nfreq,Jbeams)</code> | `complex64` |
| `.bpsd` | Beam PSD | `bf_file_utils` | <code style="white-space: nowrap;">(Nfreq,Jbeams)</code> | `float32` |
| `.wng` | Beam white-noise gain | `wng_file_utils` | <code style="white-space: nowrap;">(Nfreq,Jbeams)</code> | `float32` |
| `.bfw` | Beamformer weights | `bf_weight_file_utils` | <code style="white-space: nowrap;">(Nfreq,Jbeams,Nchan)</code> | `complex64` |
| `.btr` | Bearing-time record | `btr_file_utils` | <code style="white-space: nowrap;">(Nbands,Nazim,Nelev)</code> | `float32` or `uint8` |

The file extension is not stored in most binary headers. In particular, `.fft`
and `.psd` share one layout, as do `.bfft` and `.bpsd`. Their stored dtype and
the surrounding workflow determine whether values are complex spectra or real
PSD values. WNG has a dedicated layout so its constraint configuration travels
with the achieved values. Renaming a file does not convert its contents.

## Shared binary conventions

### Byte order marker

Every file begins with the unsigned 32-bit value `0x12345678`. Writers use the
host machine's native byte order. Readers inspect the marker as both big- and
little-endian and then use the detected order for the remaining multibyte
fields and records.

The one-byte dtype code has no byte-order distinction. Multibyte record dtypes,
including complex values, use the byte order detected from the marker.

### Scalar encodings

The layout notation in this page uses:

| Notation | NumPy dtype | Bytes | Use |
| --- | --- | ---: | --- |
| `u1` | `uint8` | 1 | Dtype code |
| `u4` | `uint32` | 4 | Byte-order marker |
| `u8` | `uint64` | 8 | Header size and Unix timestamp |
| `f4` | `float32` | 4 | Sample rates, counts, coordinates, and header arrays |

Dimension and record-count fields are stored as `float32`, not integer fields.
The writer accepts only integer values that remain exact after conversion to
`float32`. `Nseq` may be zero at file creation in a pipeline and will be
overwritten at close; dimensions must be positive.

All timestamps are unsigned 64-bit integer nanoseconds since the Unix epoch.
The `start_unix_seconds` key returned by readers is derived for convenience and
is not a separate on-disk field. Use `start_unix_nanoseconds` when exact epoch
precision matters.

### Dtype codes

FFT-family, beam-frequency, and BTR headers store one of these codes:

| Code | NumPy dtype | Bytes | Code | NumPy dtype | Bytes |
| ---: | --- | ---: | ---: | --- | ---: |
| 1 | `uint8` | 1 | 8 | `int64` | 8 |
| 2 | `uint16` | 2 | 9 | `float16` | 2 |
| 3 | `uint32` | 4 | 10 | `float32` | 4 |
| 4 | `uint64` | 8 | 11 | `float64` | 8 |
| 5 | `int8` | 1 | 12 | `complex64` | 8 |
| 6 | `int16` | 2 | 13 | `complex128` | 16 |
| 7 | `int32` | 4 | 14 | `bool` | 1 |

The code table describes what the generic headers can represent. Current
configured pipelines write only the dtypes shown in the format-family table.
Individual record writers also enforce real/complex and casting rules suitable
for their product.

### Post-beamforming bearings and headings

BTS, BFFT, BPSD, WNG, BFW, and BTR steering/bearing axes are always relative
to the array. Their version-1 records begin with one `float32`
`array_heading_deg`: degrees clockwise from true north, normalized to
`[0, 360)`, for the center time of that record. Readers return
`bearing_frame="relative"` in the header dictionary and return the selected
record headings alongside the data. Geographic conversion is a display-time
operation; stored data and beam order are not rotated.

### Record count and file closure

Pipeline sinks create a header with `Nseq = 0`, count records as they are
written, and overwrite `Nseq` when the sink closes. A file left open or
interrupted before cleanup may therefore contain data after a zero record count.
Use `run_pipeline()`, which closes sources and sinks even when processing raises
an exception.

Readers verify that the file is large enough for the declared header and record
count. Extra trailing bytes are currently permitted. There is no checksum,
footer, or recovery scan for records not included in `Nseq`.

### Array serialization

Arrays are serialized in NumPy C order. The rightmost axis is contiguous and
varies fastest on disk. File records have no padding or per-record delimiter:

```text
file = header + record[0] + record[1] + ... + record[Nseq - 1]
```

For any fixed-size format:

```text
record_offset(i) = byte_header + i * bytes_per_record
```

## Channel time series: `.ts`

A TS file stores real channel samples and a vector of channel labels or IDs.

### TS header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `Nseq` | `f4` | 1 | Number of records |
| 12 | `sample_rate_hz` | `f4` | 1 | Positive sample rate in hertz |
| 16 | `Npt_per_seq` | `f4` | 1 | Samples per channel in one record |
| 20 | `Nchan` | `f4` | 1 | Number of channels |
| 24 | `channels` | `f4` | `Nchan` | Finite channel labels/IDs |
| `24 + 4*Nchan` | `start_unix_nanoseconds` | `u8` | 1 | Time of the first sample |

```text
byte_header = 32 + 4*Nchan
bytes_per_record = 4*Nchan*Npt_per_seq
```

### TS records

Each record is `float32` with shape `(Nchan, Npt_per_seq)`. Time samples are
contiguous within each channel; channels vary more slowly:

```text
channel 0 samples, channel 1 samples, ..., channel Nchan-1 samples
```

`ts_read_sequence()` accepts one or more record indices and concatenates the
requested records along the sample axis, returning `(Nchan, T)`.

The header timestamp is the time of the first sample in record zero. Records
advance by `Npt_per_seq / sample_rate_hz` seconds.

## Beam time series: `.bts`

A BTS file stores real samples for steered beams plus the steering grid and
element shading that produced them.

### BTS header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `Nseq` | `f4` | 1 | Number of records |
| 12 | `sample_rate_hz` | `f4` | 1 | Positive sample rate in hertz |
| 16 | `Npt_per_seq` | `f4` | 1 | Samples per beam in one record |
| 20 | `Jbeams` | `f4` | 1 | Number of beams |
| 24 | `Nchan` | `f4` | 1 | Number of shaded input elements |
| 28 | `Nshades` | `f4` | 1 | `1` or `Jbeams` |
| 32 | `steer_rbe_rad_J3` | `f4` | `3*Jbeams` | Range/bearing/elevation rows |
| `32 + 12*Jbeams` | `shade` | `f4` | `Nshades*Nchan` | Shading rows |
| `32 + 12*Jbeams + 4*Nshades*Nchan` | `start_unix_nanoseconds` | `u8` | 1 | Time of first sample |

```text
byte_header = 40 + 12*Jbeams + 4*Nshades*Nchan
bytes_per_record = 4 + 4*Jbeams*Npt_per_seq
```

Each steering row contains range in meters, bearing in radians, and elevation
in radians. The grid contains every range/bearing/elevation combination, with
range varying slowest and elevation fastest. Bearings crossing north are
stored unwrapped so the bearing axis remains ascending.

`shade` has shape `(1, Nchan)` for one shared element shading or
`(Jbeams, Nchan)` for per-beam shading. Weights are finite and nonnegative, and
every shading row contains at least one positive weight.

### BTS records

Each record starts with `array_heading_deg`, followed by `float32` samples with
shape `(Jbeams, Npt_per_seq)`. Samples are contiguous within each beam.
`bts_read_sequence()` returns `(data_JT, headings_T)` and concatenates requested
sample records along `T`.

The header timestamp is the time of the first sample in record zero. Records
advance by `Npt_per_seq / sample_rate_hz` seconds.

## Channel frequency products: `.fft` and `.psd`

Channel FFT and PSD files share the same header and record layout. The current
pipeline uses `complex64` records for `.fft` and `float32` records for `.psd`.

### FFT/PSD header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `sample_rate_hz` | `f4` | 1 | Positive sample rate in hertz |
| 12 | `Nseq` | `f4` | 1 | Number of records |
| 16 | `Nchan` | `f4` | 1 | Number of channels |
| 20 | `nfft` | `f4` | 1 | Original FFT length |
| 24 | `Nfreq` | `f4` | 1 | Stored frequency count |
| 28 | `advance` | `f4` | 1 | Samples between record centers |
| 32 | `byte_header` | `u8` | 1 | Total header size in bytes |
| 40 | `start_unix_nanoseconds` | `u8` | 1 | Center time of first frame |
| 48 | dtype code | `u1` | 1 | Record dtype from the shared code table |
| 49 | `window_T` | `f4` | `nfft` | Applied time-domain window |
| <code style="white-space: nowrap;">49+4*nfft</code> | `freq` | `f4` | `Nfreq` | Stored frequencies in hertz |
| <code style="white-space: nowrap;">49+4*(nfft+Nfreq)</code> | `channels` | `f4` | `Nchan` | Channel labels/IDs |

```text
byte_header = 49 + 4*(nfft + Nfreq + Nchan)
bytes_per_record = Nfreq*Nchan*dtype_bytes
```

`window_T` always has the original `nfft` samples, even after frequency
cropping. For files created by the current TS-to-FFT stage, it is the exact
`float32` power-normalized window applied to the time series. Its mean-square
value is approximately one. See [Spectral normalization](../concepts/spectral-normalization.md)
for PSD integration and sinusoid-level recovery.

`freq` contains nonnegative, strictly increasing frequencies that map to bins
of the original one-sided FFT grid. It may contain the full grid, a cropped
contiguous band, skipped bins, or one bin; therefore `Nfreq` need not equal
`floor(nfft/2) + 1`.

### FFT/PSD records

Each record has shape `(Nfreq, Nchan)` and the dtype selected by the header.
Channels are contiguous within each frequency bin. Complex scalars use NumPy's
complex representation, with real and imaginary components in the detected
file byte order.

`fft_read_single_sequence()` returns `(Nfreq, Nchan)`.
`fft_read_multiple_sequence()` returns `(T, Nfreq, Nchan)`, with requested
records on the leading axis.

The header timestamp is the center of spectral record zero. Records advance by
`advance / sample_rate_hz` seconds. `secs_per_seq` is derived by the reader and
is not stored in this format.

## Beam-frequency products: `.bfft` and `.bpsd`

Beam FFT and beam PSD files share one layout. In addition to the FFT metadata,
they store beam steering coordinates and the element shading used for
beamforming. The current pipeline uses `complex64` for `.bfft` and `float32`
for `.bpsd`.

### BFFT/BPSD header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `sample_rate_hz` | `f4` | 1 | Positive sample rate in hertz |
| 12 | `Nseq` | `f4` | 1 | Number of records |
| 16 | `Jbeams` | `f4` | 1 | Number of beams |
| 20 | `nfft` | `f4` | 1 | Original FFT length |
| 24 | `Nfreq` | `f4` | 1 | Stored frequency count |
| 28 | `advance` | `f4` | 1 | Samples between record centers |
| 32 | `Nchan` | `f4` | 1 | Number of shaded input elements |
| 36 | `Nshades` | `f4` | 1 | `1` or `Jbeams` |
| 40 | `secs_per_seq` | `f4` | 1 | `advance / sample_rate_hz` |
| 44 | `byte_header` | `u8` | 1 | Total header size in bytes |
| 52 | `start_unix_nanoseconds` | `u8` | 1 | Center time of first frame |
| 60 | dtype code | `u1` | 1 | Record dtype from the shared code table |
| 61 | `window_T` | `f4` | `nfft` | Applied time-domain window |
| <code style="white-space: nowrap;">61+4&#42;nfft</code> | `shade` | `f4` | `Nshades*Nchan` | Shared or per-beam shading |
| <code style="white-space: nowrap;">61+4&#42;(nfft+Nshades&#42;Nchan)</code> | `freq` | `f4` | `Nfreq` | Stored frequencies in hertz |
| <code style="white-space: nowrap;">61+4&#42;(nfft+Nshades&#42;Nchan+Nfreq)</code> | `steer_rbe_rad_J3` | `f4` | `3*Jbeams` | Steering rows |

```text
byte_header = 61 + 4*(nfft + Nshades*Nchan + Nfreq + 3*Jbeams)
bytes_per_record = 4 + Nfreq*Jbeams*dtype_bytes
```

Window, frequency, shading, and steering-grid rules are the same as in the
FFT/PSD and BTS sections. The reader verifies that stored `secs_per_seq` agrees
with `advance / sample_rate_hz` within float32 rounding tolerance.

### BFFT/BPSD records

Each record starts with `array_heading_deg`, followed by data with shape
`(Nfreq, Jbeams)`. Beams are contiguous within each frequency bin.
`bf_read_single_sequence()` returns `(data_QJ, heading_deg)`.
`bf_read_multiple_sequence()` returns `(data_QJT, headings_T)`.

The header timestamp is the center of beam-frequency record zero. Records
advance by `advance / sample_rate_hz` seconds.

## White-noise gain: `.wng`

WNG is a dedicated version-1 format. Its axes and records follow the BF
conventions, while its header also stores the relative constraint and
tolerance used by DMR.

### WNG header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `sample_rate_hz` | `f4` | 1 | Positive sample rate in hertz |
| 12 | `Nseq` | `f4` | 1 | Number of records |
| 16 | `Jbeams` | `f4` | 1 | Number of beams |
| 20 | `nfft` | `f4` | 1 | Original FFT length |
| 24 | `Nfreq` | `f4` | 1 | Stored frequency count |
| 28 | `advance` | `f4` | 1 | Samples between record centers |
| 32 | `Nchan` | `f4` | 1 | Number of shaded input elements |
| 36 | `Nshades` | `f4` | 1 | `1` or `Jbeams` |
| 40 | `secs_per_seq` | `f4` | 1 | `advance / sample_rate_hz` |
| 44 | `wngc_val_dB` | `f4` | 1 | Desired WNG relative to unshaded CBF WNG |
| 48 | `wngc_tol_dB` | `f4` | 1 | Nonnegative relaxation of desired WNG |
| 52 | `byte_header` | `u8` | 1 | Total header size in bytes |
| 60 | `start_unix_nanoseconds` | `u8` | 1 | Center time of first frame |
| 68 | dtype code | `u1` | 1 | Must encode `float32` |
| 69 | `window_T` | `f4` | `nfft` | Applied time-domain window |
| <code style="white-space: nowrap;">69+4&#42;nfft</code> | `shade` | `f4` | `Nshades*Nchan` | Shared or per-beam shading |
| <code style="white-space: nowrap;">69+4&#42;(nfft+Nshades&#42;Nchan)</code> | `freq` | `f4` | `Nfreq` | Stored frequencies in hertz |
| <code style="white-space: nowrap;">69+4&#42;(nfft+Nshades&#42;Nchan+Nfreq)</code> | `steer_rbe_rad_J3` | `f4` | `3*Jbeams` | Steering rows |

```text
byte_header = 69 + 4*(nfft + Nshades*Nchan + Nfreq + 3*Jbeams)
bytes_per_record = 4 + Nfreq*Jbeams*4
```

The DMR feasibility threshold is the unshaded CBF WNG multiplied by
`10**((wngc_val_dB - wngc_tol_dB)/10)`. With the current unit-magnitude
steering replicas, unshaded CBF WNG is the number of active elements for each
beam; active elements are the nonzero entries in `shade`.

### WNG records

Each record starts with `array_heading_deg`, followed by a `float32` array with
shape `(Nfreq, Jbeams)`. `wng_read_single_sequence()` returns
`(wng_QJ, heading_deg)` and `wng_read_multiple_sequence()` returns
`(wng_QJT, headings_T)`. `wng_constraint_linear_J()` combines the stored
constraint fields and active shading apertures into the effective absolute
linear constraint for every beam.

Records contain dimensionless linear white-noise gain, not decibels or power
spectral density. Each value is `1 / (w^H w)` for the unity-response weights
actually used for that record, frequency, and beam. Use
`10*np.log10(wng_QJ)` to express it in decibels. DMR startup and zero-energy
fallback records contain the WNG of the CBF coefficients used for those
outputs; adaptive records contain the achieved WNG returned by the loading
search.

When DMR WNG output uses a sequence stride, record zero still corresponds to
DMR input sequence zero and the stored `advance` is the input spectral advance
multiplied by that stride. The ordinary header timestamp reconstruction
therefore remains valid for `.wng` files.

## Beamformer weights: `.bfw`

Beamformer-weight files store the time-varying complex coefficients used to
map channel spectra to beam spectra. The BFW format begins with an explicit
unsigned 32-bit format version after the byte-order marker. The only supported
version is version 1.

### BFW header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `sample_rate_hz` | `f4` | 1 | Positive sample rate in hertz |
| 12 | `Nseq` | `f4` | 1 | Number of records |
| 16 | `Jbeams` | `f4` | 1 | Number of beams |
| 20 | `Nchan` | `f4` | 1 | Number of input elements/channels |
| 24 | `nfft` | `f4` | 1 | Original FFT length |
| 28 | `Nfreq` | `f4` | 1 | Stored frequency count |
| 32 | `advance` | `f4` | 1 | Samples between record centers |
| 36 | `secs_per_seq` | `f4` | 1 | `advance / sample_rate_hz` |
| 40 | `byte_header` | `u8` | 1 | Total header size in bytes |
| 48 | `start_unix_nanoseconds` | `u8` | 1 | Center time of first weight record |
| 56 | dtype code | `u1` | 1 | Complex record dtype from the shared code table |
| 57 | `window_T` | `f4` | `nfft` | Applied time-domain window |
| <code style="white-space: nowrap;">57+4&#42;nfft</code> | `channels` | `f4` | `Nchan` | Input channel labels/IDs in weight order |
| <code style="white-space: nowrap;">57+4&#42;(nfft+Nchan)</code> | `freq` | `f4` | `Nfreq` | Stored frequencies in hertz |
| <code style="white-space: nowrap;">57+4&#42;(nfft+Nchan+Nfreq)</code> | `steer_rbe_rad_J3` | `f4` | `3*Jbeams` | Steering rows |

```text
byte_header = 57 + 4*(nfft + Nchan + Nfreq + 3*Jbeams)
bytes_per_record = 4 + Nfreq*Jbeams*Nchan*dtype_bytes
```

### BFW records and weight convention

Each record starts with `array_heading_deg`. Its weights are serialized in C
order with shape `(Nfreq, Jbeams, Nchan)`, or
`(Q, J, N)`. Elements are contiguous, followed by beams and then frequency.
`bf_weight_read_single_sequence()` returns `(weights_QJN, heading_deg)` and
`bf_weight_read_multiple_sequence()` returns `(weights_QJNT, headings_T)`.

The stored values are the **already-conjugated multiplication coefficients**,
not an unconjugated mathematical weight vector. If an algorithm derives
`w_QJN` and defines its output using `x @ conj(w)`, the file stores
`conj(w_QJN)`. A loaded record is therefore applied directly:

```python
beamformed_QJ = np.einsum("qn,qjn->qj", data_QN, weights_QJN)
```

For the `(Q, N, J)` convention accepted by `BeamPattern`, transpose the loaded
array. This normally creates a view rather than copying the weight data:

```python
weights_QNJ = weights_QJN.transpose(0, 2, 1)
```

The timestamp and sequence spacing follow the spectral records whose weights
were used. Channel labels define the required input-element order; steering
rows define the beam order. The geometry itself is not embedded and must match
the geometry supplied separately for beampattern analysis or beamforming.
When weight output uses a sequence stride, record zero corresponds to DMR input
sequence zero and stored `advance` is multiplied by the stride, preserving the
standard timestamp relationship for every retained record.

## Bearing-time record: `.btr`

Band-energy pipelines write values indexed by frequency band, azimuth, and
elevation. Float32 records store band power or a background-normalized statistic
in decibels. Unsigned 8-bit records store relative azimuth ranks. The statistic
and its literature basis are defined in
[Band-energy detection](../concepts/band-energy-detection.md).

### BTR header

| Offset | Field | Encoding | Count | Meaning |
| ---: | --- | --- | ---: | --- |
| 0 | byte-order marker | `u4` | 1 | `0x12345678` |
| 4 | format version | `u4` | 1 | `1` |
| 8 | `sample_rate_hz` | `f4` | 1 | Source sample rate in hertz |
| 12 | `Nseq` | `f4` | 1 | Number of records |
| 16 | `nfft` | `f4` | 1 | Source FFT length |
| 20 | `Nbands` | `f4` | 1 | Number of frequency bands |
| 24 | `Nazim` | `f4` | 1 | Number of azimuth coordinates |
| 28 | `Nelev` | `f4` | 1 | Number of elevation coordinates |
| 32 | `advance` | `f4` | 1 | Source samples between output records |
| 36 | `secs_per_seq` | `f4` | 1 | `advance / sample_rate_hz` |
| 40 | `byte_header` | `u8` | 1 | Total header size in bytes |
| 48 | `start_unix_nanoseconds` | `u8` | 1 | Time assigned to first output record |
| 56 | dtype code | `u1` | 1 | Record dtype from the shared code table |
| 57 | `bands_B2` | `f4` | `2*Nbands` | Inclusive lower/upper bounds in hertz |
| <code style="white-space: nowrap;">57+8&#42;Nbands</code> | `azim_deg_A` | `f4` | `Nazim` | Strictly increasing relative azimuths in degrees |
| <code style="white-space: nowrap;">57+8&#42;Nbands+4&#42;Nazim</code> | `elev_deg_E` | `f4` | `Nelev` | Strictly increasing elevations in degrees |

```text
byte_header = 57 + 8*Nbands + 4*Nazim + 4*Nelev
bytes_per_record = 4 + Nbands*Nazim*Nelev*dtype_bytes
```

Band rows contain inclusive `[lower_hz, upper_hz]` bounds. Bounds are
nonnegative and ordered within each row; bands may overlap and need not align
with FFT bins. Azimuth and elevation axes must each be strictly increasing.
Circular azimuth axes must be unwrapped before storage if needed.

The first BTR timestamp is the time assigned to the first band-energy output.
Without background time averaging it is inherited from the input. With a time
window it identifies the selected center record. Records advance by
`advance / sample_rate_hz`; the reader verifies stored `secs_per_seq` against
that relationship.

### BTR records

Each record starts with `array_heading_deg`. The Python writer accepts
`(Nbands, Nazim, Nelev)`. For the common singleton
elevation case it also accepts `(Nbands, Nazim)`. Before serialization it
transposes the record to `(Nelev, Nbands, Nazim)`, so azimuth varies fastest on
disk, followed by band, then elevation.

`btr_read_sequence()` supports selected records and selected elevations. It
returns `(data_EBAT, headings_T)`. The data retain singleton elevation and
record axes.

## Reading files directly

Header readers return a dictionary containing decoded fields, derived timing
values, and an open file object under `header["fid"]`. Close that file object
after reading records:

```python
import numpy as np

from wavekernelrt.core.fft_file_utils import (
    fft_read_header,
    fft_read_multiple_sequence,
)

header = fft_read_header("output.psd")
try:
    sequence_indices = np.arange(header["Nseq"])
    psd_TQN = fft_read_multiple_sequence(header, sequence_indices)
finally:
    header["fid"].close()
```

Read-index arguments may be a scalar or one-dimensional integer array. Requested
order and duplicate indices are preserved, and an empty selection is allowed.

## Writing files directly

Header writers open the target in `"wb"` mode, so an existing file is
immediately truncated. Supply the final record count when writing a complete
file directly:

```python
import numpy as np

from wavekernelrt.core.ts_file_utils import ts_write_header, ts_write_sequence

samples_NT = np.zeros((2, 4096), dtype=np.float32)

header = ts_write_header(
    "example.ts",
    Nseq=1,
    sample_rate_hz=4096.0,
    start_unix_nanoseconds=0,
    Npt_per_seq=4096,
    channels=np.array([0, 1]),
)
try:
    ts_write_sequence(header, samples_NT, check_finite=True)
finally:
    header["fid"].close()
```

The optional `check_finite=True` record-writer argument rejects NaN or infinite
values before and after conversion to the output dtype. Pipeline sinks expose
this through `OutputConfig.check_finite`. It is normally set to `False` in a pipeline
for computationally speed purposes.

## Validation and compatibility limits

Current readers validate the marker, header field shapes and ranges, expected
header size where stored, and minimum file size implied by `Nseq`. They do not
authenticate files or protect against deliberate resource-exhaustion inputs.
Treat files from untrusted sources accordingly.

Every current layout stores format version 1. The current layouts do not store:

- a format name or product-type identifier,
- software version or provenance,
- calibration units or scale factors,
- a checksum or integrity digest,
- arbitrary user metadata, or
- a footer confirming clean closure.

An incompatible layout change must increment the affected format version and
make an explicit compatibility decision: retain an older reader, provide a
conversion path, or reject the older version with a clear error. Applications
should use the provided WaveKernelRT readers rather than duplicating these
layouts in independent parsers.
