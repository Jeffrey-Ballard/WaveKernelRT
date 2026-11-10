import numpy as np
import numpy.typing as npt
from typing import Self
from dataclasses import dataclass, replace
from wavekernelrt.pipeline.timebase import Timebase
from wavekernelrt.core.data_types import RealArray, NumericArray

# define a generic block with common fields that all blocks will inherit
@dataclass(slots=True)
class Block:
    sequence: int

@dataclass
class PipelineDataBlock:
    # a block built and modified in build_pipeline to keep track of parameters that may change from stage to stage (fft crop, etc.)
    # so we can set up stages and outouts easily
    sample_rate_hz: float
    timebase: Timebase
    nfft: int | None = None
    advance: int | None = None
    freq: np.ndarray | None = None
    Nchan: int | None = None
    channels: np.ndarray | None  = None
    window_T: np.ndarray | None = None
    start_time_ns: int | None = None
    file_type: str | None = None
    shade_N: np.ndarray | None = None
    shade_JN: np.ndarray | None = None
    steer_rbe_rad_J3: np.ndarray | None = None
    Npt_per_seq: int | None = None

    # Metadata for bearing-time-record output stages.
    bands_B2: npt.ArrayLike | None = None
    btr_azim_deg_A: np.ndarray | None = None
    btr_elev_deg_E: np.ndarray | None = None
    btr_dtype: np.dtype | type | None = None

    # Original source-channel count and positions retained by this pipeline.
    source_channel_count: int | None = None
    source_channel_indices: np.ndarray | None = None

    def sample_to_unix_ns(self, sample: int | float) -> int:
        return self.timebase.sample_to_unix_ns(sample)

    def unix_ns_to_sample(self, unix_ns: int ) -> int | float:
        return self.timebase.unix_ns_to_sample(unix_ns)
    
@dataclass(slots=True)
class TimeBlock(Block):
    """
    One contiguous block of time-series data.

    Shape:
        data_NT: sensors x time samples
    """

    data_NT: RealArray
    array_heading_deg: float | None = None

    @property
    def nsensors(self) -> int:
        return self.data_NT.shape[0]

    @property
    def nsamples(self) -> int:
        return self.data_NT.shape[1]

    # helper property to use the time block for beamformed timeseries to
    # make it clear in the code what is happening
    @property
    def data_JT(self) -> RealArray:
        return self.data_NT

@dataclass(slots=True)
class FFTBlock(Block):
    """
    One contiguous block of fft data.

    Shape:
        data_QN: freq x sensors
    """

    data_QN: NumericArray
    advance: int
    array_heading_deg: float | None = None

    @property
    def nsensors(self) -> int:
        return self.data_QN.shape[1]

    @property
    def nfreq(self) -> int:
        return self.data_QN.shape[0]
    
    @property
    def data(self) -> NumericArray:
        return self.data_QN

    def with_data(self, data: NumericArray) -> Self:
        return replace(self,data_QN=data)

@dataclass(slots=True)
class BFBlock(Block):
    """
    One contiguous block of beamformed data.

    Shape:
        data_QJ: freq x beams
    """

    data_QJ: NumericArray
    advance: int
    array_heading_deg: float | None = None

    @property
    def nbeams(self) -> int:
        return self.data_QJ.shape[1]

    @property
    def nfreq(self) -> int:
        return self.data_QJ.shape[0]

    @property
    def data(self) -> NumericArray:
        return self.data_QJ

    def with_data(self, data: NumericArray) -> Self:
        return replace(self,data_QJ=data)
    
@dataclass(slots=True)
class BTRBlock(Block):
    """
    One contiguous block of BTR data.

    Shape:
        data_BAE: band x azimuth x elevations
    """

    data_BAE: RealArray
    array_heading_deg: float | None = None

    @property
    def nbands(self) -> int:
        return self.data_BAE.shape[0]

    @property
    def nazimuths(self) -> int:
        return self.data_BAE.shape[1]

    @property
    def nelevations(self) -> int:
        return self.data_BAE.shape[2]
