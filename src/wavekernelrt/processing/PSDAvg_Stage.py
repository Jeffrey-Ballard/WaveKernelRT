from wavekernelrt.pipeline.blocks import FFTBlock,BFBlock

from collections.abc import Iterator
from numbers import Integral

from dataclasses import replace
import numpy as np

class PSDAvg_Stage:
    def __init__(self,seq_to_avg: int) -> None:
        if (isinstance(seq_to_avg,bool) or not isinstance(seq_to_avg,Integral) or seq_to_avg <= 0):
            raise ValueError("seq_to_avg must be a positive integer")

        self.seq_to_avg = seq_to_avg
        self.count = 0
        self._sequence = 0

    def process[B: FFTBlock | BFBlock](self,block: B) -> Iterator[B]:
        if self.count==0:
            self.avg_data = block.data / self.seq_to_avg
            self._heading_sin = 0.0
            self._heading_cos = 0.0
        else:
            self.avg_data += block.data / self.seq_to_avg

        if isinstance(block,BFBlock):
            if block.array_heading_deg is None:
                raise ValueError("Beam PSD blocks require array_heading_deg")
            heading_rad = np.deg2rad(block.array_heading_deg)
            self._heading_sin += np.sin(heading_rad)
            self._heading_cos += np.cos(heading_rad)

        self.count += 1

        if self.count==self.seq_to_avg:
            self.count = 0
            output = replace(block.with_data(self.avg_data),advance=block.advance*self.seq_to_avg,sequence=self._sequence)
            if isinstance(block,BFBlock):
                output = replace(output,array_heading_deg=float(np.rad2deg(np.arctan2(self._heading_sin,self._heading_cos,)) % 360.0))
            yield output

            self._sequence += 1
