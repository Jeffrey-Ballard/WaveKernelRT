from collections.abc import Iterator

import numpy.typing as npt

from wavekernelrt.pipeline.blocks import FFTBlock,TimeBlock

class ChannelCrop_Stage:
    def __init__(self,idx_channels_to_keep: npt.ArrayLike,file_type: str) -> None:
        self.file_type = file_type
        self.idx_channels_to_keep = idx_channels_to_keep

    def process(self,block: FFTBlock | TimeBlock) -> Iterator[FFTBlock | TimeBlock]:
        # what are we converting, keep track in the pipeline
        match self.file_type:
            case "ts" | ".ts":
                yield TimeBlock(data_NT=block.data_NT[self.idx_channels_to_keep,:],sequence=block.sequence,
                    array_heading_deg=block.array_heading_deg)
            case "fft" | ".fft" | "psd" | ".psd":
                yield FFTBlock(data_QN=block.data_QN[:,self.idx_channels_to_keep],advance=block.advance,sequence=block.sequence)
