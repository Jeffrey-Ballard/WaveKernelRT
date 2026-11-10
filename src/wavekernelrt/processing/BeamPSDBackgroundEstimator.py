"""Reusable background estimation for beam power-spectral-density records."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d,percentile_filter

from wavekernelrt.core.general_utils import clip_and_replace_dB,make_clip_and_replace_filters


@dataclass(frozen=True,slots=True)
class BackgroundEstimate:
    """A centered beam PSD record and its estimated background."""

    center_psd_QJ: npt.NDArray
    background_dB_QJ: npt.NDArray


class BeamPSDBackgroundEstimator:
    """Estimate a configurable background from beam PSD records.

    Records are accumulated in a centered sliding window. Once the window is
    full, each call returns its center record and the background estimate.

    Processing order is fixed: optionally time average in linear power,
    convert to decibels, optionally apply guarded-mean clip-and-replace
    decision filters along frequency, optionally smooth along frequency, then
    optionally take the bearing-neighborhood median. Clip-and-replace is
    Hampel-like but uses a fixed one-sided dB threshold and a guarded mean,
    rather than a two-sided MAD threshold and median replacement.
    Passing only ``nbeams`` returns each current record in decibels without
    buffering or additional filtering.

    Abraham (2019), Sec. 9.3.1, pp. 683--685, motivates repeated,
    nearby-bearing, and nearby-frequency auxiliary observations for estimating
    a noise background. Equations 9.192--9.194 support an arithmetic mean of
    linear powers under their independent equal-mean model. The decibel-domain
    convolution, bearing median, and clip-and-replace rules here are separate
    implementation choices; they are not those equations' sample-mean
    estimator. See ``docs/references.md`` for the full citation and scope.
    """

    def __init__(self,nbeams: int,bkgd_time_seq_avg: int | None = None,bkgd_sector_beam_size: int | None = None,
        full_circle_flag: bool = False,freq_smoothing_window: npt.ArrayLike | None = None,
        car_parms_WP: npt.ArrayLike | None = None) -> None:

        if (isinstance(nbeams,bool) or not isinstance(nbeams,Integral) or nbeams <= 0):
            raise ValueError("nbeams must be a positive integer")

        if bkgd_time_seq_avg is not None:
            if (isinstance(bkgd_time_seq_avg,bool) or not isinstance(bkgd_time_seq_avg,Integral) or bkgd_time_seq_avg <= 0):
                raise ValueError("bkgd_time_seq_avg must be a positive integer or None")

        if bkgd_sector_beam_size is not None:
            if (isinstance(bkgd_sector_beam_size,bool) or not isinstance(bkgd_sector_beam_size,Integral)
                or not 1 <= bkgd_sector_beam_size <= nbeams):
                raise ValueError("bkgd_sector_beam_size must be an integer between 1 and nbeams")

        if not isinstance(full_circle_flag,bool):
            raise ValueError("full_circle_flag must be True or False")

        smoothing_window = None
        if freq_smoothing_window is not None:
            smoothing_window = np.asarray(freq_smoothing_window)
            if (smoothing_window.ndim != 1 or smoothing_window.size == 0 or np.iscomplexobj(smoothing_window)
                or not np.all(np.isfinite(smoothing_window))):
                raise ValueError("freq_smoothing_window must be a nonempty 1D array of finite real values")

        self.bkgd_time_seq_avg = None if bkgd_time_seq_avg is None else int(bkgd_time_seq_avg)
        self.nbeams = int(nbeams)
        self.bkgd_sector_beam_size = None if bkgd_sector_beam_size is None else int(bkgd_sector_beam_size)
        self.full_circle_flag = full_circle_flag
        self.freq_smoothing_window = smoothing_window
        self._origin = None if smoothing_window is None else -1 if smoothing_window.size % 2 == 0 else 0
        self.clip_and_replace_filters = None if car_parms_WP is None else make_clip_and_replace_filters(car_parms_WP)

        self._oldest_t = 0
        self._buffer_QJT: npt.NDArray | None = None
        self._center_offset = None if self.bkgd_time_seq_avg is None else self.bkgd_time_seq_avg // 2
        self._count = 0

    def process(self,data_QJ: npt.ArrayLike) -> BackgroundEstimate | None:
        """Add one beam PSD record and return an estimate when ready."""
        data = np.asarray(data_QJ)
        if data.ndim != 2 or data.shape[1] != self.nbeams:
            raise ValueError(
                "Beam PSD records must have shape "
                f"(Nfreq, {self.nbeams}); received {data.shape}")

        if self.bkgd_time_seq_avg is None:
            center_psd_QJ = data.copy()
            background_linear_QJ = data
        else:
            self._append(data)
            self._count += 1

            if self._count < self.bkgd_time_seq_avg:
                return None

            if (self._buffer_QJT is None or self._center_offset is None):  # pragma: no cover
                raise RuntimeError("Background buffer was not initialized")

            center_t = (self._oldest_t + self._center_offset) % self.bkgd_time_seq_avg
            center_psd_QJ = self._buffer_QJT[:,:,center_t].copy()
            # Abraham (2019), Sec. 9.3.1, Eqs. 9.192--9.194, supports an
            # arithmetic mean of linear auxiliary powers under its iid model.
            background_linear_QJ = np.mean(self._buffer_QJT,axis=2)

        background_dB_QJ = 10 * np.log10(np.maximum(background_linear_QJ,1e-40))

        if self.clip_and_replace_filters:
            background_dB_QJ = clip_and_replace_dB(background_dB_QJ,self.clip_and_replace_filters)

        if self.freq_smoothing_window is not None:
            # Abraham (2019), pp. 684--685, motivates sharing nearby-frequency
            # background observations; this dB-domain kernel is local policy.
            background_dB_QJ = convolve1d(background_dB_QJ,self.freq_smoothing_window,axis=0,mode="reflect",origin=self._origin)

        if self.bkgd_sector_beam_size is not None:
            # Abraham (2019), pp. 683--684, motivates nearby beams as auxiliary
            # data; the bearing-neighborhood median is local policy.
            background_dB_QJ = self._filter_sector_background(background_dB_QJ)

        return BackgroundEstimate(center_psd_QJ=center_psd_QJ,background_dB_QJ=background_dB_QJ)

    def _append(self,data_QJ: npt.NDArray) -> None:
        if self.bkgd_time_seq_avg is None:  # pragma: no cover - guarded by process
            raise RuntimeError("Time averaging is disabled")

        if self._buffer_QJT is None:
            self._buffer_QJT = np.empty((*data_QJ.shape,self.bkgd_time_seq_avg),dtype=data_QJ.dtype)

        self._buffer_QJT[:,:,self._oldest_t] = data_QJ
        self._oldest_t = (self._oldest_t + 1) % self.bkgd_time_seq_avg

    def _filter_sector_background(self,background_QJ: npt.NDArray) -> npt.NDArray:
        width = self.bkgd_sector_beam_size
        if width is None:  # pragma: no cover - guarded by process
            return background_QJ
        nbeams = background_QJ.shape[1]

        if self.full_circle_flag:
            return percentile_filter(background_QJ,percentile=50,size=(1,width),mode="wrap",origin=0)

        # With origin=0, even widths include one more beam to the left.
        left = width // 2
        right = width - left - 1
        result_QJ = percentile_filter(background_QJ,percentile=50,size=(1,width),mode="nearest",origin=0)

        # Replace padded boundary results using only available beams.
        for j in range(nbeams):
            if j >= left and j + right < nbeams:
                continue

            start = max(0,j - left)
            stop = min(nbeams,j + right + 1)
            neighborhood = background_QJ[:,start:stop]

            # Match percentile_filter's convention by selecting the upper
            # middle value for even neighborhood sizes.
            k = neighborhood.shape[1] // 2
            result_QJ[:,j] = np.partition(neighborhood,k,axis=1)[:,k]

        return result_QJ
