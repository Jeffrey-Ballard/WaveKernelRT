"""Conventional frequency-band energy processing for beam PSD records."""

from collections.abc import Iterator
from collections import deque
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from wavekernelrt.core.general_utils import unpack_beams_J3
from wavekernelrt.pipeline.blocks import BFBlock,BTRBlock
from wavekernelrt.processing.BeamPSDBackgroundEstimator import BeamPSDBackgroundEstimator


class BFtoBandEnergy_Stage:
    """Integrate beam PSD values over one or more frequency bands.

    Direct processing integrates PSD over frequency using per-bin frequency
    widths. When a background estimator is supplied, the centered PSD is
    first divided by the estimated background in linear power and the
    resulting dimensionless spectrum is integrated over frequency. That
    normalized statistic has units of hertz rather than power before output
    encoding.

    This normalization-before-integration order follows Abraham (2019),
    Sec. 9.3, p. 683, Eqs. 9.190--9.191. The stage's frequency-bin-width
    factors are numerical PSD-integration weights; they are an implementation
    extension of the book's linear sum. See ``docs/references.md`` for the
    full citation, assumptions, and limits of the detector rationale.

    ``float32`` output stores ``10*log10`` of the integrated value, floored at
    ``1e-40`` before conversion. Direct output is therefore relative to one
    input power unit; background-normalized output is relative to 1 Hz.
    ``rank_uint8`` ranks the integrated linear values instead.

    One steering range is supported because BTR records have band, azimuth,
    and elevation axes but no range axis. Multiple azimuths and elevations are
    supported. Ranked output is computed independently across azimuth for each
    band and elevation.
    """

    def __init__(self,freq: npt.ArrayLike,bands_B2: npt.ArrayLike,steer_rbe_rad_J3: npt.ArrayLike,
        output: Literal["float32","rank_uint8"] = "float32",
        background_estimator: BeamPSDBackgroundEstimator | None = None) -> None:

        self.freq_Q = self._validate_frequency_grid(freq)
        self.df_Q = np.gradient(self.freq_Q)
        self.bands_B2,self.band_weights_BQ = self._make_band_weights(bands_B2)

        range_m_R,azim_rad_A,elev_rad_E = unpack_beams_J3(steer_rbe_rad_J3)
        if range_m_R.size != 1:
            raise NotImplementedError("Band energy output supports exactly one steering range")

        self.azim_deg_A = np.rad2deg(azim_rad_A)
        self.elev_deg_E = np.rad2deg(elev_rad_E)
        self.nazimuths = self.azim_deg_A.size
        self.nelevations = self.elev_deg_E.size
        self.nbeams = self.nazimuths * self.nelevations

        if output not in ("float32","rank_uint8"):
            raise ValueError("output must be 'float32' or 'rank_uint8'")
        self.output = output

        if (background_estimator is not None and not isinstance(background_estimator,BeamPSDBackgroundEstimator)):
            raise ValueError("background_estimator must be a BeamPSDBackgroundEstimator or None")
        if (background_estimator is not None and background_estimator.nbeams != self.nbeams):
            raise ValueError("Background estimator beam count does not match the steering grid")
        self.background_estimator = background_estimator
        heading_window = 1
        if (background_estimator is not None and background_estimator.bkgd_time_seq_avg is not None):
            heading_window = background_estimator.bkgd_time_seq_avg
        self._heading_buffer = deque(maxlen=heading_window)
        self._sequence = 0

    def process(self,block: BFBlock) -> Iterator[BTRBlock]:
        """Process one beam PSD block, yielding when any buffer is ready."""
        data_QJ = self._validate_psd(block.data_QJ)
        if block.array_heading_deg is None:
            raise ValueError("Beam PSD blocks require array_heading_deg")
        self._heading_buffer.append(block.array_heading_deg)

        if self.background_estimator is None:
            spectrum_QJ = data_QJ
        else:
            estimate = self.background_estimator.process(data_QJ)
            if estimate is None:
                return

            background_dB_QJ = np.asarray(estimate.background_dB_QJ,dtype=np.float64)
            if (background_dB_QJ.shape != data_QJ.shape or not np.all(np.isfinite(background_dB_QJ))):
                raise ValueError("Background estimate must match the PSD shape and contain finite values")
            background_linear_QJ = np.power(10.0,background_dB_QJ / 10.0)
            if (not np.all(np.isfinite(background_linear_QJ)) or np.any(background_linear_QJ <= 0)):
                raise ValueError("Background estimate cannot be represented as positive finite linear power")
            # Abraham (2019), Eq. 9.190: normalize each bin in linear power.
            spectrum_QJ = np.asarray(estimate.center_psd_QJ,dtype=np.float64) / background_linear_QJ
            if not np.all(np.isfinite(spectrum_QJ)):
                raise ValueError("Background-normalized spectrum must contain finite values")

        # Abraham (2019), Eq. 9.191, sums linear normalized bin powers. These
        # weights additionally account for each PSD bin's width in hertz.
        values_BJ = self.band_weights_BQ @ spectrum_QJ
        values_BAE = values_BJ.reshape(self.bands_B2.shape[0],self.nazimuths,self.nelevations)
        if not np.all(np.isfinite(values_BAE)):
            raise ValueError("Integrated band values must be finite")

        if self.output == "float32":
            output_BAE = (10 * np.log10(np.maximum(values_BAE,1e-40))).astype(np.float32)
            if not np.all(np.isfinite(output_BAE)):
                raise ValueError("Band energy dB values exceed the finite float32 range")
        else:
            output_BAE = self._rank_azimuths(values_BAE)

        yield BTRBlock(data_BAE=output_BAE,sequence=self._sequence,array_heading_deg=self._heading_buffer[len(self._heading_buffer) // 2])
        self._sequence += 1

    def _validate_frequency_grid(self,freq: npt.ArrayLike) -> npt.NDArray:
        freq_Q = np.asarray(freq)
        if (freq_Q.ndim != 1 or freq_Q.size < 2 or np.iscomplexobj(freq_Q) or not np.all(np.isfinite(freq_Q))):
            raise ValueError("freq must contain at least two finite real values")

        spacing = np.diff(freq_Q.astype(np.float64))
        if np.any(spacing <= 0):
            raise ValueError("freq must be strictly increasing")

        return freq_Q.astype(np.float64)

    def _make_band_weights(self,bands_B2: npt.ArrayLike) -> tuple[npt.NDArray,npt.NDArray]:
        bands = np.asarray(bands_B2)
        if (bands.ndim != 2 or bands.shape[0] == 0 or bands.shape[1] != 2 or np.iscomplexobj(bands)
            or not np.all(np.isfinite(bands))):
            raise ValueError("bands_B2 must be a nonempty finite real array with shape (Nbands, 2)")
        if np.any(bands[:,0] > bands[:,1]):
            raise ValueError("Each frequency band must have low <= high")

        masks_BQ = ((self.freq_Q[np.newaxis,:] >= bands[:,0,np.newaxis])
            & (self.freq_Q[np.newaxis,:] <= bands[:,1,np.newaxis]))
        if np.any(~np.any(masks_BQ,axis=1)):
            raise ValueError("Every frequency band must contain at least one bin")

        weights_BQ = masks_BQ.astype(np.float64) * self.df_Q[np.newaxis,:]
        return bands.astype(np.float64),weights_BQ

    def _validate_psd(self,data_QJ: npt.ArrayLike) -> npt.NDArray:
        data = np.asarray(data_QJ)
        expected_shape = (self.freq_Q.size,self.nbeams)
        if data.shape != expected_shape:
            raise ValueError(
                f"Beam PSD data must have shape {expected_shape}; "
                f"received {data.shape}")
        if np.iscomplexobj(data):
            raise ValueError("Band energy requires real PSD values")
        if not np.all(np.isfinite(data)):
            raise ValueError("Band energy requires finite PSD values")
        if np.any(data < 0):
            raise ValueError("Band energy requires nonnegative PSD values")

        return data

    def _rank_azimuths(self,values_BAE: npt.NDArray) -> npt.NDArray[np.uint8]:
        ranks_BAE = rankdata(values_BAE,method="min",axis=1) - 1

        if self.nazimuths == 1:
            return np.zeros(values_BAE.shape,dtype=np.uint8)

        return np.rint(ranks_BAE * (255.0 / (self.nazimuths - 1))).astype(np.uint8)
