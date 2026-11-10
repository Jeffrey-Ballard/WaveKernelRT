"""Reusable input validation and binary header preparation for WaveKernelRT.

The validate_header_* helpers should be called before opening an output file. They return validated values
for both the returned header dictionary and serialization; they perform no I/O
and do not modify input arrays.

Overlap between formats:
* TS, BTS, FFT, BF, BTR: float32 integer counts, positive sample rate,
  uint64 Unix nanoseconds, and finite real float32 arrays.
* TS, FFT: channel vectors; FFT, BF: frequency and window vectors.
* BTS, BF: shared/per-beam shading and ordered steering grids.
* BF, BTR: a stored positive float32 sequence duration.

Layout sizes, frequency-bin relationships, and BTR band/angle conventions
remain the responsibility of each format. Dtype codes already live in
data_types.py. These helpers do not constitute complete reader validation:
readers must also check truncation and file sizes, and validate decoded
float32 count fields before converting them to Python integers.
"""

from numbers import Integral, Real

import numpy as np
import numpy.typing as npt


def validate_header_count(value: int, name: str, *, minimum: int = 1) -> int:
    """Return a Python integer exactly representable in a float32 header field.

    Use minimum=0 for Nseq; dimensions and stored header byte counts use 1.
    Integer-valued floats and booleans are rejected at the writer interface.
    Values above 2**24 are allowed when they are still exactly representable.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")

    value = int(value)

    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")

    if abs(value) > int(np.finfo(np.float32).max):
        raise ValueError(f"{name} must be exactly representable as float32")

    # Compare Python ints so the comparison itself cannot round the input.
    if int(np.float32(value)) != value:
        raise ValueError(f"{name} must be exactly representable as float32")

    return value


def validate_header_positive_float32(value: Real, name: str) -> np.float32:
    """Return a finite positive float32 scalar (e.g. sample_rate_hz).

    Ordinary float32 rounding is allowed. Reject overflow, nonfinite inputs,
    and positive values that round to zero. No application-specific minimum
    sample rate is imposed. Also usable for stored secs_per_seq fields.
    """

    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not 0 < value <= float(np.finfo(np.float32).max)):

        raise ValueError(f"{name} must be finite, positive, and representable as float32")
    
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.float32(value)

    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must remain finite and positive as float32")

    return result


def validate_header_float32(value: Real, name: str, *, nonnegative: bool = False) -> np.float32:
    """Return one finite float32 scalar, optionally requiring a nonnegative value."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")

    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.float32(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite float32 value")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")

    return result


def validate_header_timestamp(start_unix_nanoseconds: int) -> int:
    """Return exact Python integer Unix nanoseconds within the uint64 range.

    Reject floats rather than trying to recover precision already lost in a
    floating-point epoch timestamp. Zero is the Unix epoch, not a missing value.
    """

    if (isinstance(start_unix_nanoseconds, (bool, np.bool_))
        or not isinstance(start_unix_nanoseconds, Integral)
        or not 0 <= start_unix_nanoseconds <= int(np.iinfo(np.uint64).max)):
        raise ValueError("start_unix_nanoseconds must be an integer in the uint64 range")
    
    return int(start_unix_nanoseconds)


def validate_header_array(values: npt.ArrayLike, name: str,  *, shape: tuple[int | None, ...]) -> npt.NDArray[np.float32]:
    """Return a nonempty finite real float32 copy with the requested shape.

    None denotes a variable-length axis: (None,) for channels/frequencies,
    (nfft,) for a window, (None, 2) for BTR bands. All axes must be nonempty.
    Reject strings, objects, booleans, complex values, and conversion overflow.
    Normal rounding and underflow to zero are allowed; callers enforce semantic
    constraints such as positive shading or distinct frequency coordinates.
    """
    
    array = np.asarray(values)
    if array.ndim != len(shape) or array.size == 0 or any(expected is not None and actual != expected for actual,
        expected in zip(array.shape, shape)):
        raise ValueError(f"{name} must be nonempty with shape {shape}")
    
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real numbers")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = array.astype(np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must remain finite as float32")

    return result


def validate_header_shading(shade: npt.ArrayLike, Jbeams: int) -> npt.NDArray[np.float32]:
    """Return BTS/BF shading as (1, N) or (Jbeams, N) float32 weights.

    A shared (N,) vector is expanded to (1, N). Weights must be nonnegative
    before conversion and each vector must retain a positive weight afterward.
    This validates stored shading, not the beamformer's minimum element count.
    """

    Jbeams = validate_header_count(Jbeams, "Jbeams")

    array = np.asarray(shade)
    if array.ndim == 1:
        array = array[None, :]

    result = validate_header_array(array, "shade", shape=(None, None))

    if result.shape[0] not in (1, Jbeams):
        raise ValueError("Shading must have shape (N,), (1, N), or (Jbeams, N)")

    if np.any(array < 0) or not np.all(np.any(result > 0, axis=1)):
        raise ValueError("Shading must be nonnegative with a positive weight in each vector")

    validate_header_count(result.shape[0], "Nshades")
    validate_header_count(result.shape[1], "Nchan")

    return result


def validate_header_beam_grid(steer_rbe_rad_J3: npt.ArrayLike) -> npt.NDArray[np.float32]:

    """Return a BTS/BF grid valid both before and after float32 serialization.

    Columns are range (meters), azimuth (radians), elevation (radians).
    Reuse validate_beam_grid's complete-grid and ascending-axis conventions.
    Checking the serialized grid catches distinct directions rounded together.
    """

    array = np.asarray(steer_rbe_rad_J3)
    result = validate_header_array(array, "steer_rbe_rad_J3", shape=(None, 3))

    validate_beam_grid(array)
    validate_beam_grid(result)
    validate_header_count(result.shape[0], "Jbeams")

    return result


def validate_beam_grid(beams_J3):

    """Validate a complete, ordered range/azimuth/elevation grid.
    Must pass a validation check for compliance with WaveKernelRT conventions
        
    Columns are range in meters, azimuth in radians, elevation in radians.
    Axes must be ascending, with range slowest and elevation fastest.
    Azimuths crossing north must be unwrapped, e.g. 358 through 362 degrees.
    """

    beams = np.asarray(beams_J3)
    
    if beams.ndim != 2 or beams.shape[1] != 3 or beams.shape[0] == 0:
        raise ValueError("Beam grid must have nonempty shape (J, 3)")

    if not np.issubdtype(beams.dtype, np.number) or np.iscomplexobj(beams):
        raise ValueError("Beam coordinates must be real numbers")

    if not np.all(np.isfinite(beams)):
        raise ValueError("Beam coordinates must be finite")
        
    ranges_m = np.unique(beams[:,0])
    azim_A = np.unique(beams[:,1])
    elev_E = np.unique(beams[:,2])
     
    R, A, E = ranges_m.size, azim_A.size, elev_E.size

    if beams.shape[0] != R * A * E:
        raise ValueError(
            "Beam grid must contain every range/azimuth/elevation "
            "combination exactly once")

    grid = beams.reshape(R, A, E, 3)

    ordered = (
        np.all(grid[..., 0] == ranges_m[:, None, None])
        and np.all(grid[..., 1] == azim_A[None, :, None])
        and np.all(grid[..., 2] == elev_E[None, None, :]))

    if not ordered:
        raise ValueError(
            "Beam grid must use ascending axes with range slowest, "
            "azimuth next, and elevation fastest. "
            "Use unwrapped azimuths for sectors crossing north.")

    return ranges_m, azim_A, elev_E

def validate_header_frequencies(freq: npt.ArrayLike, sample_rate_hz: Real, nfft: int) -> npt.NDArray[np.float32]:

    """Validate actual one-sided FFT bin frequencies in Hz.

    sample_rate_hz and nfft must already be validated header values.

    Cropped grids, skipped bins, and single-bin grids are allowed.
    Return the float32 frequencies without snapping or relabeling them.
    """

    freq = validate_header_array(freq, "freq", shape=(None,))

    if np.any(freq < 0):
        raise ValueError("freq must be nonnegative")

    if np.any(freq[1:] <= freq[:-1]):
        raise ValueError("freq must be strictly increasing")

    # Use the same metadata comparison precision as FFTtoTS_Stage.
    # The returned frequencies and signal data remain float32.
    freq_check = freq.astype(np.float64)
    df = float(sample_rate_hz) / nfft
    bin_numbers = np.rint(freq_check / df)

    if np.any(bin_numbers > nfft // 2):
        raise ValueError("freq contains values outside the original FFT grid")

    if np.any(bin_numbers[1:] <= bin_numbers[:-1]):
        raise ValueError("Multiple frequencies map to the same FFT bin")

    expected_freq = bin_numbers * df

    # Allow float32 header rounding, capped at 1% of a bin spacing.
    tolerance_hz = np.minimum(0.01 * df, 4 * np.finfo(np.float32).eps * np.maximum(expected_freq, df))

    if np.any(np.abs(freq_check - expected_freq) > tolerance_hz):
        raise ValueError("freq does not match the original FFT bin frequencies")

    return freq

def validate_header_bands(bands_B2: npt.ArrayLike) -> npt.NDArray[np.float32]:

    """Return finite, nonnegative inclusive frequency bounds in Hz.

    Each row contains [lower, upper]. Bounds need not align with FFT bins.
    Overlapping bands and equal bounds are allowed.
    """

    bands = validate_header_array(bands_B2, "bands_B2", shape=(None, 2))

    if np.any(bands < 0):
        raise ValueError("Frequency band bounds must be nonnegative")

    if np.any(bands[:, 0] > bands[:, 1]):
        raise ValueError("Lower band bounds must not exceed upper bounds")

    return bands


def validate_header_axis(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float32]:

    """Return a finite, strictly increasing float32 coordinate vector.

    A single coordinate is allowed. Circular coordinates must be unwrapped
    before calling; this function does not sort or reorder coordinates.
    """

    axis = validate_header_array(values, name, shape=(None,))

    if np.any(axis[1:] <= axis[:-1]):
        raise ValueError(f"{name} must be strictly increasing with no duplicates")

    return axis

def validate_read_indices(values: npt.ArrayLike, size: int, name: str) -> npt.NDArray:

    """Validate scalar or one-dimensional integer read indices.

    size is the already-validated number of available records or elevations.
    Preserve requested order and duplicates. Empty selections are allowed.
    """

    indices = np.atleast_1d(np.asarray(values))

    if indices.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")

    if indices.size == 0:
        return np.empty(0, dtype=np.intp)

    if indices.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")

    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError(f"{name} must satisfy 0 <= index < {size}")

    return indices

def validate_array_heading_deg(value):
    """Return a finite heading normalized to degrees clockwise from north."""

    heading = np.asarray(value)
    if (heading.ndim != 0 or not np.issubdtype(heading.dtype, np.number) or np.iscomplexobj(heading) or not np.isfinite(heading)):
        raise ValueError("array_heading_deg must be a finite real scalar")

    return float(heading) % 360.0

