from wavekernelrt.pipeline.blocks import FFTBlock,BFBlock

from collections.abc import Iterator
import numpy as np

class FFTtoPSD_Stage:
    def __init__(self,integration_period_secs: float) -> None:
        self.integration_period_secs = integration_period_secs

    def process[B: FFTBlock | BFBlock](self,block: B) -> Iterator[B]:
        psd_data = (np.abs(block.data)**2/self.integration_period_secs).astype(np.float32)
        yield block.with_data(psd_data)
