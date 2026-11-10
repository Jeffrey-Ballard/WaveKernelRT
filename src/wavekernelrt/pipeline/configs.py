"""Declarative configuration objects for file-backed processing pipelines."""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal
import numpy.typing as npt
from numbers import Integral, Real
from math import isfinite

from wavekernelrt.core.geometry import Geometry


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Describe an existing pipeline input file.

    Parameters
    ----------
    file : pathlib.Path
        Existing input-file path.
    type : str
        Logical input type: ``"ts"``, ``"bts"``, ``"fft"``, ``"bfft"``, or
        ``"bpsd"``, optionally with a leading period. Type compatibility and
        file existence are checked by
        [`build_pipeline`][wavekernelrt.pipeline.build_pipeline.build_pipeline].
    """
    file: Path
    type: str

@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Describe a pipeline output file.

    Parameters
    ----------
    file : pathlib.Path
        Output path. Opening the sink creates or overwrites this file.
    type : str
        Logical output type: ``"ts"``, ``"bts"``, ``"fft"``, ``"psd"``,
        ``"bfft"``, ``"bpsd"``, or ``"btr"``. A leading period is accepted.
    check_finite : bool, default=False
        Check every output array for NaN or infinity before writing. This adds
        a full-array check to each write and slows down run time.

    Raises
    ------
    ValueError
        If ``check_finite`` is not Boolean. Type, extension, and pipeline
        compatibility are checked when the pipeline is built.
    """
    file: Path
    type: str
    check_finite: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.check_finite, bool):
            raise ValueError("check_finite must be True or False")


@dataclass(frozen=True, slots=True)
class TStoFFTConfig:
    """Configure conversion from time series to one-sided spectra.

    Parameters
    ----------
    nfft : positive int, default=4096
        Samples in each FFT frame. The one-sided output has
        ``floor(nfft/2) + 1`` frequency bins.
    advance : positive int, default=2048
        Samples between frame centers. Must not exceed ``nfft``.
    window_name : {"tukey", "boxcar", "hann", "hanning", "hamming"}, default="tukey"
        Case-insensitive symmetric analysis-window name. Tukey alpha is
        ``1 - advance/nfft``; boxcar applies no taper.

    Raises
    ------
    ValueError
        If counts or the window name are invalid.

    Notes
    -----
    The selected window is normalized to unit mean square before application,
    preserving stationary broadband power in expectation. Exact normalized
    ``float32`` coefficients are stored in FFT and PSD headers. Sinusoid
    recovery still requires the window's equivalent noise bandwidth.

    ``ts`` input becomes ``fft``; ``bts`` becomes ``bfft``. Hann has zero
    endpoints, so inverse processing cannot recover those samples by dividing
    out the window; Hamming has nonzero endpoints.
    """

    nfft: int = 4096
    advance: int = 2048
    window_name: str = "tukey"

    def __post_init__(self) -> None:
        for name in ("nfft", "advance"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")

            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.advance > self.nfft:
            raise ValueError("advance must not exceed nfft")

        if (not isinstance(self.window_name, str)
                or self.window_name.lower() not in ("tukey", "boxcar", "hann", "hanning", "hamming")):
            raise ValueError("window_name must be 'tukey', 'boxcar', 'hann', 'hanning', or 'hamming'")

@dataclass(frozen=True, slots=True)
class FFTtoTSConfig:
    """Configure reconstruction of time series from complex spectra.

    Parameters
    ----------
    nfft : int or None, default=None
        Original FFT length. Inherited from current metadata when omitted.
    advance : int or None, default=None
        Samples contributed by each spectral frame. Inherited when omitted.
    Npt_per_seq : int or None, default=None
        Samples per output record. Defaults to the effective ``advance``.
    window_T : array-like or None, default=None
        Length-``nfft`` analysis window. Inherited when omitted.

    Notes
    -----
    The builder validates resolved values and rejects PSD input because its
    phase has been lost. Missing frequency bins are filled with zero. Nonzero
    window coefficients are divided out; samples at zero coefficients remain
    zero. ``fft`` becomes ``ts`` and ``bfft`` becomes ``bts``.
    """
    nfft: int | None = None
    advance: int | None = None
    Npt_per_seq: int | None = None
    window_T: npt.ArrayLike | None = None

@dataclass(frozen=True, slots=True)
class FreqCropConfig:
    """Configure an inclusive frequency-band selection.

    Parameters
    ----------
    band_to_keep_Hz : tuple or None, default=None
        ``(lower, upper)`` frequency bounds in hertz. A finite ordered
        two-value tuple within the current grid is required when this stage is
        used.

    Notes
    -----
    The stage accepts ``fft``, ``psd``, ``bfft``, and ``bpsd`` and preserves
    the logical product type.
    """
    band_to_keep_Hz: tuple | None = None

@dataclass(frozen=True, slots=True)
class ChannelCropConfig:
    """Configure selection of source-channel positions.

    Parameters
    ----------
    idx_channels_to_keep : array-like or None, default=None
        Nonempty one-dimensional zero-based integer positions. A value is
        required when the stage is used.

    Notes
    -----
    The stage accepts channel ``ts``, ``fft``, and ``psd`` products. If a
    panel or curved geometry is later beamformed, retained elements must form
    a complete rectangular row/column subset in original order.
    """
    idx_channels_to_keep: npt.ArrayLike | None = None

@dataclass(frozen=True, slots=True)
class FFTtoPSDConfig:
    """Configure conversion from complex spectra to power spectral density.

    This fieldless stage maps ``fft`` to ``psd`` and ``bfft`` to ``bpsd``.
    Output is single-sided ``float32`` PSD scaled by the FFT integration
    period. Analysis-window power normalization has already occurred in the
    preceding time-series-to-FFT stage.
    """

    pass

@dataclass(frozen=True, slots=True)
class PSDAvgConfig:
    """Configure nonoverlapping averaging of adjacent PSD records.

    Parameters
    ----------
    seq_to_avg : positive int or None, default=None
        Exact number of adjacent records in each average.
    time_to_avg_secs : positive float or None, default=None
        Requested duration in seconds. The builder converts this to a rounded
        record count using the current record spacing.

    Raises
    ------
    ValueError
        If exactly one field is not supplied, or the supplied value is not
        finite and positive.

    Notes
    -----
    Only complete groups are emitted. The stage accepts both ``psd`` and
    ``bpsd`` and preserves the logical product type.
    """
    seq_to_avg: int | None = None
    time_to_avg_secs: float | None = None

    def __post_init__(self) -> None:
        if (self.seq_to_avg is None) == (self.time_to_avg_secs is None):
            raise ValueError("Exactly one of seq_to_avg or time_to_avg_secs must be specified.")

        if self.seq_to_avg is not None:
            if (isinstance(self.seq_to_avg, bool)
                or not isinstance(self.seq_to_avg, Integral)
                or self.seq_to_avg <= 0):

                raise ValueError("seq_to_avg must be a positive integer")

        if self.time_to_avg_secs is not None:
            if (isinstance(self.time_to_avg_secs, bool)
                or not isinstance(self.time_to_avg_secs, Real)
                or not isfinite(self.time_to_avg_secs)
                or self.time_to_avg_secs <= 0):

                raise ValueError("time_to_avg_secs must be a finite, positive real scalar")

@dataclass(frozen=True, slots=True)
class ShadingConfig:
    """Configure spatial shading recomputed for the selected geometry.

    Parameters
    ----------
    window_type : {"uniform", "hamming", "taylor"}, default="taylor"
        Case-sensitive spatial-window recipe.
    sll_dB : positive float, default=30.0
        Taylor sidelobe attenuation in decibels, expressed as a positive
        value. Ignored for uniform and Hamming shading.
    nbar : int, default=4
        Taylor near-in sidelobe count. Must be at least two and is ignored for
        uniform and Hamming shading.

    Raises
    ------
    ValueError
        If the recipe is unsupported or active Taylor parameters are invalid.

    Notes
    -----
    Line and panel arrays use one shared vector. Cylinders and spheres
    recompute a visible, sum-normalized vector for every steering direction.
    """

    window_type: str = "taylor"
    sll_dB: float = 30.0
    nbar: int = 4

    def __post_init__(self) -> None:
        if self.window_type not in ("uniform", "hamming", "taylor"):
            raise ValueError("Shading window_type must be 'uniform', 'hamming', or 'taylor'")

        if self.window_type == "taylor":
            if (isinstance(self.sll_dB, bool)
                or not isinstance(self.sll_dB, Real)
                or not isfinite(self.sll_dB)
                or self.sll_dB <= 0):
                raise ValueError("sll_dB must be finite and positive")

            if (isinstance(self.nbar, bool)
                or not isinstance(self.nbar, Integral)
                or self.nbar < 2):
                raise ValueError("nbar must be an integer of at least 2")
            
@dataclass(frozen=True, slots=True)
class FFTtoCBFConfig:
    """Configure conventional beamforming of channel spectra.

    Parameters
    ----------
    geom : wavekernelrt.Geometry
        Full geometry corresponding to the original input channels. The
        builder applies any preceding channel selection.
    shading : wavekernelrt.ShadingConfig
        Spatial-shading recipe.
    steer_rbe_rad_J3 : array-like, shape (J, 3)
        Complete ordered steering grid: range in meters, then relative bearing
        and elevation in radians.
    output_file : pathlib.Path or None, default=None
        Reserved for future use; currently ignored by the builder.
    heading_provider : callable or None, default=None
        Optional function called with each input FFT block. It must return the
        array heading metadata in degrees clockwise from true north. When
        omitted, the block heading is used when present, otherwise
        ``geom.array_heading_deg``. It does not change the steering grid or
        beamforming weights.

    Notes
    -----
    The stage accepts channel ``fft`` and produces ``bfft``. Steering bearings
    are always relative to array heading. At least two active channels are
    required.
    """
    geom: Geometry
    shading: ShadingConfig
    steer_rbe_rad_J3: npt.ArrayLike
    output_file: Path | None = None
    heading_provider: object | None = None

    def __post_init__(self) -> None:
        if self.heading_provider is not None and not callable(self.heading_provider):
            raise ValueError("heading_provider must be callable or None")

@dataclass(frozen=True, slots=True)
class TeeConfig:
    """Configure an intermediate file output without consuming the stream.

    Parameters
    ----------
    output : OutputConfig
        Intermediate sink. Its logical type and extension must match the
        product at this exact point in the stage sequence.

    Notes
    -----
    Every incoming block is written to the tee sink and then yielded unchanged
    to the next stage.
    """
    output: OutputConfig

@dataclass(frozen=True, slots=True)
class BackgroundEstimateConfig:
    """Configure a beam-PSD background estimate.

    Parameters
    ----------
    bkgd_time_seq_avg : positive int or None, default=None
        Records in the sliding time-average window. When omitted, process each
        input record immediately without time averaging.
    bkgd_sector_beam_size : positive int or None, default=None
        Bearing-neighborhood width in input beams. When omitted, skip bearing
        filtering.
    full_circle_flag : bool, default=False
        Apply circular rather than bounded bearing-neighborhood filtering when
        ``bkgd_sector_beam_size`` is provided.
    freq_smoothing_window : array-like or None, default=None
        One-dimensional frequency-axis convolution weights, normally
        normalized to sum to one. When omitted, skip frequency smoothing.
    car_parms_WP : list of tuple or None, default=None
        Sequential frequency-axis clip-and-replace definitions
        ``(side, guard, threshold_dB)``. Each filter estimates a local level
        from two ``side``-bin sidebands separated by a ``guard``-bin central
        exclusion and replaces only values more than ``threshold_dB`` above
        that level. This is a Hampel-like decision rule, not a median/MAD
        Hampel filter. When omitted or empty, skip clip-and-replace filtering.

    Notes
    -----
    Processing order is fixed: optionally time average in linear power,
    convert to decibels, optionally apply clip-and-replace, optionally smooth
    along frequency, then optionally take the bearing-neighborhood median.
    Each configurable operation is independent. With every field omitted,
    each current input record is returned in decibels without buffering or
    additional filtering.

    Abraham (2019), Sec. 9.3.1, pp. 683--685, motivates repeated,
    nearby-bearing, and nearby-frequency background observations. Its linear
    arithmetic-mean result applies only under the model stated there; this
    configuration's dB-domain smoothing, bearing median, and clip-and-replace
    rules are implementation choices. See ``docs/references.md``.
    """
    bkgd_time_seq_avg: int | None = None
    bkgd_sector_beam_size: int | None = None
    full_circle_flag: bool = False
    freq_smoothing_window: npt.ArrayLike | None = None
    car_parms_WP: (list[tuple[int|float, int|float, int|float]] | None) = None

    def __post_init__(self) -> None:
        if self.bkgd_time_seq_avg is not None:
            if (isinstance(self.bkgd_time_seq_avg, bool)
                    or not isinstance(self.bkgd_time_seq_avg, Integral)
                    or self.bkgd_time_seq_avg <= 0):
                raise ValueError("bkgd_time_seq_avg must be a positive integer or None")

        if self.bkgd_sector_beam_size is not None:
            if (isinstance(self.bkgd_sector_beam_size, bool)
                    or not isinstance(self.bkgd_sector_beam_size, Integral)
                    or self.bkgd_sector_beam_size <= 0):
                raise ValueError("bkgd_sector_beam_size must be a positive integer or None")

        if not isinstance(self.full_circle_flag, bool):
            raise ValueError("full_circle_flag must be True or False")

@dataclass(frozen=True, slots=True)
class BandEnergyConfig:
    """Configure conventional band-energy processing of beam PSD records.

    Parameters
    ----------
    bands_B2 : list of tuple
        Inclusive ``(low, high)`` frequency bands in hertz. Band power is the
        PSD sum weighted by the frequency-bin widths.
    background : BackgroundEstimateConfig or None, default=None
        When provided, normalize each PSD bin by its estimated background
        before integrating. When omitted, integrate the PSD directly.
    output : {"float32", "rank_uint8"}, default="float32"
        Emit ``10*log10`` of the integrated values as ``float32``, or replace
        the linear integrated values with bearing ranks mapped to ``uint8``.

    Notes
    -----
    This stage accepts ``bpsd`` and produces bearing-time records (``btr``).
    Direct ``float32`` output is band power in dB relative to one input power
    unit. When ``background`` is provided, the PSD is divided by the estimated
    background before integration and the stored value is the resulting
    statistic in dB relative to 1 Hz. A ``1e-40`` linear floor is applied
    before conversion. Exactly one steering range is supported; azimuth and
    elevation axes are preserved.

    Per-bin normalization followed by a linear sum follows Abraham (2019),
    Sec. 9.3, p. 683, Eqs. 9.190--9.191. The implementation also weights by
    bin width to integrate the PSD ratio over hertz. See
    ``docs/references.md`` for the full citation and statistical scope.
    """
    bands_B2: list[tuple[int|float, int|float]]
    background: BackgroundEstimateConfig | None = None
    output: Literal["float32", "rank_uint8"] = "float32"

    def __post_init__(self) -> None:
        if (self.background is not None
                and not isinstance(self.background, BackgroundEstimateConfig)):
            raise ValueError(
                "background must be a BackgroundEstimateConfig or None")

        if self.output not in ("float32", "rank_uint8"):
            raise ValueError("output must be 'float32' or 'rank_uint8'")

@dataclass(frozen=True, slots=True)
class DMRElementConfig:
    """Configure element-space dominant-mode rejection beamforming.

    Parameters
    ----------
    geom : wavekernelrt.Geometry
        Full geometry corresponding to the original input channels.
    shading : wavekernelrt.ShadingConfig
        Spatial-shading recipe used by the conventional and adaptive weights.
    steer_rbe_rad_J3 : array-like, shape (J, 3)
        Complete ordered steering grid: range in meters, then relative bearing
        and elevation in radians.
    Dmax, Dmin : positive int
        Maximum and minimum retained dominant-mode counts. ``Dmin`` must not
        exceed ``Dmax``.
    wngc_val_dB : finite float
        White-noise-gain constraint target in decibels.
    wngc_tol_dB : nonnegative finite float
        White-noise-gain constraint tolerance in decibels.
    eig_thresh : float
        Cumulative eigen-energy threshold in ``(0, 1]`` used for mode
        selection.
    sinr_loss_dB : positive finite float, default=1.0
        Nominal SINR-loss design target used with ``Dmax`` to derive the
        covariance training snapshot count.
    write_weights : bool, default=False
        Write the multiplication weights actually used for each output record
        to a separate BFW file.
    weight_band_Hz : tuple or None, default=None
        Optional inclusive ``(low, high)`` frequency band to store. When
        omitted, store weights for every frequency bin.
    output_weight_file : pathlib.Path or None, default=None
        Explicit BFW destination. When omitted, the builder derives the path
        from the final output and input filenames.
    weight_seq_stride : positive int, default=1
        Store one weight record for every this-many DMR input sequences,
        beginning with sequence zero.
    write_wng : bool, default=False
        Write the achieved white-noise gain used for every DMR-stage output
        record to a separate WNG file.
    wng_band_Hz : tuple or None, default=None
        Optional inclusive ``(low, high)`` frequency band to store. When
        omitted, store WNG for every frequency bin.
    output_wng_file : pathlib.Path or None, default=None
        Explicit WNG destination. When omitted, the builder derives the path
        from the final output and input filenames.
    wng_seq_stride : positive int, default=1
        Store one WNG record for every this-many DMR input sequences,
        beginning with sequence zero.
    heading_provider : callable or None, default=None
        Optional function called with each input FFT block. It must return the
        array heading metadata in degrees clockwise from true north. When
        omitted, the block heading is used when present, otherwise
        ``geom.array_heading_deg``. It does not change the steering grid or
        beamforming weights.

    Raises
    ------
    ValueError
        If integer counts, mode bounds, eigen threshold, or WNG parameters are
        invalid.

    Notes
    -----
    The stage accepts channel ``fft`` and produces ``bfft``. Steering bearings
    are always relative to array heading. It emits CBF output during covariance
    initialization, then switches to DMR.

    Enabling ``write_weights`` can create extremely large BFW sidecar files.
    Each stored ``complex64`` record has a payload of
    ``8 * Nstored_freq * Jbeams * Nchan`` bytes. Use ``weight_band_Hz`` and
    ``weight_seq_stride`` to limit frequency and time storage before a long
    run; these settings affect only the diagnostic sidecar, not DMR processing
    or beam output.

    For more in-depth coverage of covariance startup and updating, dominant
    mode selection, the white-noise-gain constraint, diagnostic sidecars, and
    implementation assumptions, see
    [Beamforming](../../concepts/beamforming.md#dominant-mode-rejection).
    """
    geom: Geometry
    shading: ShadingConfig
    steer_rbe_rad_J3: npt.ArrayLike
    Dmax: int
    Dmin: int
    wngc_val_dB: float
    wngc_tol_dB: float
    eig_thresh: float
    sinr_loss_dB: float = 1.0
    write_weights: bool = False
    weight_band_Hz: tuple | None = None
    output_weight_file: Path | None = None
    weight_seq_stride: int = 1
    write_wng: bool = False
    wng_band_Hz: tuple | None = None
    output_wng_file: Path | None = None
    wng_seq_stride: int = 1
    heading_provider: object | None = None

    def __post_init__(self) -> None:
        if self.heading_provider is not None and not callable(self.heading_provider):
            raise ValueError("heading_provider must be callable or None")

        if not isinstance(self.write_weights, bool):
            raise ValueError("write_weights must be True or False")

        if not isinstance(self.write_wng, bool):
            raise ValueError("write_wng must be True or False")

        for name in ("weight_seq_stride", "wng_seq_stride"):
            value = getattr(self, name)
            if (isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        for name in ("Dmin", "Dmax"):
            value = getattr(self, name)

            if (isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        if self.Dmin > self.Dmax:
            raise ValueError("Dmin must not exceed Dmax")

        if (isinstance(self.eig_thresh, bool)
            or not isinstance(self.eig_thresh, Real)
            or not isfinite(self.eig_thresh)
            or not 0 < self.eig_thresh <= 1):
            raise ValueError("eig_thresh must be finite and in (0, 1]")

        for name in ("wngc_val_dB", "wngc_tol_dB"):
            value = getattr(self, name)
            if (isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")

        if self.wngc_tol_dB < 0:
            raise ValueError("wngc_tol_dB must be nonnegative")

        if (isinstance(self.sinr_loss_dB, bool)
            or not isinstance(self.sinr_loss_dB, Real)
            or not isfinite(self.sinr_loss_dB)
            or self.sinr_loss_dB <= 0):
            raise ValueError(
                "sinr_loss_dB must be a positive finite real number")


StageConfig = (
    TStoFFTConfig
    | FFTtoTSConfig
    | FreqCropConfig
    | ChannelCropConfig
    | FFTtoPSDConfig
    | PSDAvgConfig
    | FFTtoCBFConfig
    | TeeConfig
    | BandEnergyConfig
    | DMRElementConfig)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Declare a complete source-stage-sink pipeline.

    Parameters
    ----------
    input : InputConfig
        Existing input-file description.
    stages : tuple of StageConfig
        Ordered stage configurations. An empty tuple copies a supported input
        product to a matching output format.
    output : OutputConfig
        Final output-file description.

    Notes
    -----
    Configuration objects are frozen. Use ``dataclasses.replace`` to make a
    modified copy. Cross-stage compatibility is validated by
    [`build_pipeline`][wavekernelrt.pipeline.build_pipeline.build_pipeline].
    """
    input: InputConfig
    stages: tuple[StageConfig, ...]
    output: OutputConfig


@dataclass(slots=True)
class BuiltPipeline:
    """Runtime objects produced by
    [`build_pipeline`][wavekernelrt.pipeline.build_pipeline.build_pipeline].

    Attributes
    ----------
    source : Any
        Iterable file source with a ``close()`` method.
    stages : list
        Ordered runtime stages. Each exposes ``process(block)`` and may expose
        ``close()``.
    sink : Any
        Final file sink with ``write(block)`` and ``close()`` methods.

    Notes
    -----
    This object owns open resources but is not a context manager. Pass its
    fields to [`run_pipeline`][wavekernelrt.pipeline.run_pipeline.run_pipeline],
    which closes them even when processing raises. If it is never run, the
    caller must close them.
    """
    source: Any
    stages: list[Any]
    sink: Any
