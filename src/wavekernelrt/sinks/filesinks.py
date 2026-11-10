# import the blocks that will be written out
from wavekernelrt.pipeline.blocks import TimeBlock, FFTBlock, BFBlock, BTRBlock

# import the custom file io to write it out to files
from wavekernelrt.core.ts_file_utils import ts_write_sequence, ts_overwrite_Nseq
from wavekernelrt.core.bts_file_utils import bts_write_sequence, bts_overwrite_Nseq
from wavekernelrt.core.fft_file_utils  import fft_write_sequence, fft_overwrite_Nseq
from wavekernelrt.core.bf_file_utils  import bf_write_sequence, bf_overwrite_Nseq
from wavekernelrt.core.wng_file_utils import wng_overwrite_Nseq,wng_write_sequence
from wavekernelrt.core.bf_weight_file_utils import bf_weight_overwrite_Nseq,bf_weight_write_sequence
from wavekernelrt.core.btr_file_utils  import btr_write_sequence, btr_overwrite_Nseq

class TSFileSink:
    def __init__(self,header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, block: TimeBlock) -> None:
        ts_write_sequence(self.header,block.data_NT, check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            # go back and overwrite the header for the right value
            ts_overwrite_Nseq(self.header,self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed

class BTSFileSink:
    def __init__(self,header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, block: TimeBlock) -> None:
        bts_write_sequence(self.header,block.data_JT,block.array_heading_deg,check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            # go back and overwrite the header for the right value
            bts_overwrite_Nseq(self.header,self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed
        
class FFTFileSink:
    def __init__(self,header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, block: FFTBlock):
        fft_write_sequence(self.header,block.data_QN, check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            # go back and overwrite the header for the right value
            fft_overwrite_Nseq(self.header,self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed

class BFFileSink:
    def __init__(self,header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, block: BFBlock):
        bf_write_sequence(self.header,block.data_QJ,block.array_heading_deg,check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            # go back and overwrite the header for the right value
            bf_overwrite_Nseq(self.header,self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed

class WNGFileSink:
    """Write linear white-noise-gain records to a WNG file."""

    def __init__(self, header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, block: BFBlock):
        wng_write_sequence(self.header,block.data_QJ,block.array_heading_deg,check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            wng_overwrite_Nseq(self.header, self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed

class BFWeightFileSink:
    """Write QJN beamformer-weight arrays to a BFW file."""

    def __init__(self, header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, weights_QJN, array_heading_deg) -> None:
        bf_weight_write_sequence(self.header,weights_QJN,array_heading_deg,check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            bf_weight_overwrite_Nseq(self.header, self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed

class BTRFileSink:
    def __init__(self,header, *, check_finite=False):
        self.header = header
        self.check_finite = check_finite
        self.closed = False
        self.seq_count = 0

    def write(self, block: BTRBlock):
        btr_write_sequence(self.header,block.data_BAE,block.array_heading_deg,check_finite=self.check_finite)
        self.seq_count += 1

    def close(self) -> None:
        if self.closed:
            return

        fid = self.header["fid"]
        try:
            # go back and overwrite the header for the right value
            btr_overwrite_Nseq(self.header,self.seq_count)
        finally:
            try:
                fid.close()
            finally:
                self.closed = fid.closed

