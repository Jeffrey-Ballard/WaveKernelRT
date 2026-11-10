from collections.abc import Iterator
from numbers import Integral,Real

import numpy as np
import numpy.typing as npt

from wavekernelrt.pipeline.blocks import TimeBlock,FFTBlock,BFBlock

# Store a power-preserving single-sided spectrum.
# T = nfft / sample_rate_hz; frequency-bin width df = 1/T.
#
# Without a window, a bin-centered sinusoid of peak amplitude A has
# abs(Y) = A*T/sqrt(2), excluding DC and Nyquist.
#
# PSD = abs(Y)**2 / T.
# The sinusoid's PSD bin value is A**2*T/2, and its integrated
# power is sum(PSD)*df = A**2/2.
#
# For zero-mean white noise with standard deviation std, interior
# bins are std*sqrt(2*T/fs), and expected
# PSD = std**2/(fs/2). DC and Nyquist are not doubled.
#
# The analysis window is power-normalized on construction:
# mean(window_T**2) = 1. This preserves stationary broadband power in
# expectation, so integrated PSD does not need a window-energy correction.
# The normalized window is also the window stored in FFT/PSD headers.
#
# A bin-centered sinusoid still requires an equivalent-noise-bandwidth
# correction. For a tone away from DC and Nyquist:
# enbw_corr = mean(window_T**2) / mean(window_T)**2
# A_peak = sqrt(2 * PSD_peak * df * enbw_corr)
#
# A power-normalized Hann window has enbw_corr approximately 1.5.

class TStoFFT_Stage:
    def __init__(self,nfft: int,advance: int,sample_rate_hz: float,file_type: str,
        window_T: npt.ArrayLike | None = None) -> None:
        self.nfft = nfft
        self.advance = advance
        self.sample_rate_hz = sample_rate_hz
        self.file_type = file_type

        for name in ("nfft","advance"):
            value = getattr(self,name)

            if isinstance(value,bool) or not isinstance(value,Integral):
                raise ValueError(f"{name} must be an integer")

            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.advance > self.nfft:
            raise ValueError("advance must not exceed nfft")

        if (isinstance(sample_rate_hz,bool) or not isinstance(sample_rate_hz,Real) or not np.isfinite(sample_rate_hz)
            or sample_rate_hz <= 0):
            raise ValueError("sample_rate_hz must be a finite, positive real scalar")

        if window_T is None:
            window_T = np.ones(nfft)
        else:
            window_T = np.asarray(window_T)

        if window_T.shape != (self.nfft,):
            raise ValueError("window_T must be a 1-D array of length nfft")

        if (not np.issubdtype(window_T.dtype,np.number) or np.iscomplexobj(window_T)):
            raise ValueError("window_T must contain real numeric values")

        if not np.all(np.isfinite(window_T)):
            raise ValueError("window_T must contain finite values")

        # float32 is the on-disk window representation. Apply these same
        # coefficients during processing so the stored header is exact.
        self.window_T = (window_T / np.sqrt(np.mean(window_T**2))).astype(np.float32)

        # Created when the first TimeBlock arrives because the number
        # of sensors and dtype are not known beforehand.
        self._buffer_NT: npt.NDArray | None = None

        # Absolute sample index represented by buffer[:, 0].
        self._buffer_sample_start: int | None = None

        self._frame = 0

    def process(self,block: TimeBlock) -> Iterator[FFTBlock | BFBlock]:
        # Building a buffer in case the FFT length is small or shorter than the
        # the size of the data sent in
        self._append(block)

        while self._buffer_NT.shape[1] >= self.nfft:
            # Copy into an independent variable, so you do not modify it in place with a view.
            # if you're doing overlapped data this will cause weirdness if
            # you don't make it a copy
            frame_NT = self._buffer_NT[:,:self.nfft].copy()
            frame_NT *= self.window_T[None,:]

            Y_NQ = np.fft.rfft(frame_NT,n=self.nfft,axis=1)/self.sample_rate_hz

            # but still need to correct for the negative frequencies
            if self.nfft % 2 == 0:
                # Even: exclude DC and Nyquist
                Y_NQ[:,1:-1] *= np.sqrt(2)
            else:
                # Odd: exclude only DC
                Y_NQ[:,1:] *= np.sqrt(2)

            match self.file_type:
                case ".fft" | "fft":
                    yield FFTBlock(data_QN=Y_NQ.T,advance=self.advance,sequence=self._frame,array_heading_deg=block.array_heading_deg)
                case ".bfft" | "bfft":
                    yield BFBlock(data_QJ=Y_NQ.T,advance=self.advance,sequence=self._frame,array_heading_deg=block.array_heading_deg)

            self._frame += 1

            # Advance, retaining the overlap.
            self._buffer_NT = self._buffer_NT[:,self.advance:]
            self._buffer_sample_start += self.advance

    def _append(self,block: TimeBlock) -> None:
        if self._buffer_NT is None:
            # Copy because the source may reuse or modify its array.
            self._buffer_NT = block.data_NT.copy()
            self._buffer_sample_start = 0
            return

        self._buffer_NT = np.concatenate((self._buffer_NT,block.data_NT),axis=1)
