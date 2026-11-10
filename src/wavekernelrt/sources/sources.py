from wavekernelrt.pipeline.blocks import TimeBlock, FFTBlock, BFBlock
from wavekernelrt.core.ts_file_utils import ts_read_sequence
from wavekernelrt.core.bts_file_utils import bts_read_sequence
from wavekernelrt.core.fft_file_utils import fft_read_single_sequence
from wavekernelrt.core.bf_file_utils import bf_read_single_sequence

class ExistingFileSourceTS:
    def __init__(self, header, sequence_indices):
        self.header = header
        self.sequence_indices = sequence_indices
        self.sample_rate_hz = header["sample_rate_hz"]

    def __iter__(self):
        sample_start = 0

        for sequence in self.sequence_indices:
            data_NT = ts_read_sequence(self.header,sequence)

            yield TimeBlock(data_NT=data_NT,sequence=sequence)

            sample_start += data_NT.shape[1]

    def close(self) -> None:
        self.header["fid"].close()

class ExistingFileSourceBTS:
    def __init__(self, header, sequence_indices):
        self.header = header
        self.sequence_indices = sequence_indices
        self.sample_rate_hz = header["sample_rate_hz"]

    def __iter__(self):
        sample_start = 0

        for sequence in self.sequence_indices:
            data_JT, headings_T = bts_read_sequence(self.header,sequence)

            # Needs to be init with data_NT, but can be called with data_JT downstream with helper property
            yield TimeBlock(data_NT=data_JT,sequence=sequence,array_heading_deg=float(headings_T[0]))

            sample_start += data_JT.shape[1]

    def close(self) -> None:
        self.header["fid"].close()

class ExistingFileSourceFFT:
    def __init__(self, header, sequence_indices):
        self.header = header
        self.sequence_indices = sequence_indices
        self.advance = header["advance"]
   
    def __iter__(self):
        sample_start = 0

        for sequence in self.sequence_indices:
            data_QN = fft_read_single_sequence(self.header,sequence)

            yield FFTBlock(data_QN=data_QN,advance=self.advance,sequence=sequence)

            sample_start += self.advance

    def close(self) -> None:
        self.header["fid"].close()

class ExistingFileSourceBF:
    def __init__(self, header, sequence_indices):
        self.header = header
        self.sequence_indices = sequence_indices
        self.advance = header["advance"]
   
    def __iter__(self):
        sample_start = 0

        for sequence in self.sequence_indices:
            data_QJ, array_heading_deg = bf_read_single_sequence(self.header,sequence)

            yield BFBlock(data_QJ=data_QJ,advance=self.advance,sequence=sequence,array_heading_deg=array_heading_deg)

            sample_start += self.advance

    def close(self) -> None:
        self.header["fid"].close()
