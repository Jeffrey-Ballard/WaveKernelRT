from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from wavekernelrt.pipeline.blocks import FFTBlock,BFBlock

class FreqCrop_Stage:
    def __init__(self,band_to_keep_Hz: tuple,freq: npt.ArrayLike,file_type: str) -> None:
        self.file_type = file_type

        self.idx0 = int(np.searchsorted(freq,band_to_keep_Hz[0],side="left"))
        self.idx1 = int(np.searchsorted(freq,band_to_keep_Hz[1],side="right"))

        if self.idx0 >= self.idx1:
            raise ValueError("Requested frequency range contains no FFT bins")

    def process(self,block: FFTBlock | BFBlock) -> Iterator[FFTBlock | BFBlock]:
        # what are we converting, keep track in the pipeline
        match self.file_type:
            case "fft" | ".fft" | "psd" | ".psd":
                yield FFTBlock(data_QN=block.data_QN[self.idx0:self.idx1,:],advance=block.advance,sequence=block.sequence)
            case "bfft" | ".bfft" | "bpsd" | ".bpsd":
                yield BFBlock(data_QJ=block.data_QJ[self.idx0:self.idx1,:],advance=block.advance,
                    sequence=block.sequence,array_heading_deg=block.array_heading_deg)
