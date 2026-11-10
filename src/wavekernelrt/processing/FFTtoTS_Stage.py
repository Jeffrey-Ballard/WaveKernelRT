from collections.abc import Iterator
from numbers import Integral,Real

import numpy as np
import numpy.typing as npt

from wavekernelrt.pipeline.blocks import TimeBlock,FFTBlock,BFBlock

class FFTtoTS_Stage:
    def __init__(self,freq: npt.ArrayLike,nfft: int,advance: int,sample_rate_hz: float,Npt_per_seq: int,
        window_T: npt.ArrayLike | None = None) -> None:

        self.nfft = nfft
        self.advance = advance
        self.sample_rate_hz = sample_rate_hz
        self.Npt_per_seq = Npt_per_seq

        for name in ("nfft","advance","Npt_per_seq"):
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
            self.window_T = np.ones(nfft)
        else:
            self.window_T = np.asarray(window_T)

        if self.window_T.shape != (self.nfft,):
            raise ValueError("window_T must be a 1-D array of length nfft")

        if (not np.issubdtype(self.window_T.dtype,np.number) or np.iscomplexobj(self.window_T)):
            raise ValueError("window_T must contain real numeric values")

        if not np.all(np.isfinite(self.window_T)):
            raise ValueError("window_T must contain finite values")

        self.mask_T = self.window_T != 0

        # map the frequencies in the file to the full band
        full_freq = np.fft.rfftfreq(nfft,d=1 / sample_rate_hz)
        df = sample_rate_hz / nfft
        self._nfreq_full = full_freq.size

        freq = np.asarray(freq,dtype=np.float64)

        if freq.ndim != 1 or freq.size == 0:
            raise ValueError("freq must be a nonempty 1-D array")

        if not np.all(np.isfinite(freq)):
            raise ValueError("freq must contain finite values")

        if np.any(np.diff(freq) <= 0):
            raise ValueError("freq must be strictly increasing")

        bin_numbers = np.rint(freq / df)

        if np.any((bin_numbers < 0) | (bin_numbers >= self._nfreq_full)):
            raise ValueError("freq contains values outside the original FFT grid")

        self._freq_indices = bin_numbers.astype(np.intp)

        if np.unique(self._freq_indices).size != freq.size:
            raise ValueError("Multiple frequencies map to the same FFT bin")

        # Allow float32 header rounding, but never a substantial bin displacement.
        expected_freq = full_freq[self._freq_indices]
        tolerance_hz = np.minimum(0.01 * df,4 * np.finfo(np.float32).eps * np.maximum(expected_freq,df))

        if np.any(np.abs(freq - expected_freq) > tolerance_hz):
            raise ValueError("freq does not match the original FFT bin frequencies")

        # Buffer to write out time series in specific block sizes
        # of sensors and dtype are not known beforehand.
        self._buffer_NT: npt.NDArray | None = None

        # Absolute sample index represented by buffer[:, 0].
        self._buffer_sample_start: int | None = None

        self._nsensors: int | None = None
        self._frame = 0
        self._count = 0

    def process(self,block: FFTBlock | BFBlock) -> Iterator[TimeBlock]:
        # Undo the single-sided sqrt(2) scaling before irfft,
        # which reconstructs the negative-frequency contribution.
        # copy the data so we don't change things inline for other
        # stages that may use this block in the future
        data_QN = block.data  # Also works for BFBlock's frequency × beam data.

        if data_QN.ndim != 2 or data_QN.shape[0] != self._freq_indices.size:
            raise ValueError("Input spectrum shape does not match configured freq")

        if not np.iscomplexobj(data_QN):
            raise ValueError("Inverse FFT requires complex spectra, not PSD data")

        # Restore retained bins to their original positions.
        # Missing bins stay zero; the incoming block is not modified.
        Y_NQ = np.zeros((data_QN.shape[1],self._nfreq_full),dtype=data_QN.dtype)
        Y_NQ[:,self._freq_indices] = data_QN.T

        if self.nfft % 2 == 0:
            # Even: exclude DC and Nyquist
            Y_NQ[:,1:-1] /= np.sqrt(2)
        else:
            # Odd: exclude only DC
            Y_NQ[:,1:] /= np.sqrt(2)

        x_NT = np.fft.irfft(Y_NQ,n=self.nfft,axis=1) * self.sample_rate_hz

        # Remove the window where recovery is possible.
        # Preserve positions of unrecoverable samples by filling them with zero.
        x_NT[:,self.mask_T] /= self.window_T[self.mask_T]
        x_NT[:,~self.mask_T] = 0

        # handle the first sequence to hold onto the start
        idx1 = int(np.floor(self.nfft / 2 + self.advance / 2))
        if self._count==0:
            idx0 = 0
        else:
            idx0 = int(np.floor(self.nfft / 2 - self.advance / 2))

        self._append(x_NT[:,idx0:idx1])

        while self._buffer_NT.shape[1] >= self.Npt_per_seq:
            # Copy into an independent variable, so you do not modify it in place with a view.
            # if you're doing overlapped data this will cause weirdness if
            # you don't make it a copy
            frame_NT = self._buffer_NT[:,:self.Npt_per_seq].copy()

            yield TimeBlock(data_NT=frame_NT,sequence=self._frame,array_heading_deg=block.array_heading_deg)

            self._frame += 1

            # drop the data we don't need anyore
            self._buffer_NT = self._buffer_NT[:,self.Npt_per_seq:]

        self._count += 1

    def _append(self,data_NT: npt.ArrayLike) -> None:
        if self._buffer_NT is None:
            # Copy because the source may reuse or modify its array.
            self._buffer_NT = data_NT.copy()
            return

        self._buffer_NT = np.concatenate((self._buffer_NT,data_NT),axis=1)
