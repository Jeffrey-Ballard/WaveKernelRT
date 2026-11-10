from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from wavekernelrt.pipeline.blocks import FFTBlock,BFBlock

# import classes needed for beamforming
from wavekernelrt.core.geometry import Geometry,BeamformingGeometry
from wavekernelrt.core.steeringvectors import SteeringVectors
from wavekernelrt.core.validate_utils import validate_array_heading_deg

class FFTtoCBF_Stage:
    def __init__(self,geom: Geometry | BeamformingGeometry,shade: npt.ArrayLike,steer_rbe_rad_J3: npt.ArrayLike,
        freq_process: npt.ArrayLike,heading_provider=None) -> None:

        if geom.xyz_m_N3.shape[0] < 2:
            raise ValueError("Beamforming requires at least two elements")

        # build the CBF weights
        steer = SteeringVectors(geom,steer_rbe_rad_J3,freq_process,precision="float32",flag_relative=True)

        replica_QJN = np.transpose(steer.complex_phasors_QNJ,(0,2,1))

        shade = np.asarray(shade)

        nbeams,nelements = replica_QJN.shape[1:]

        if geom.array_type in ("sphere","cylinder"):
            if shade.shape != (nbeams,nelements):
                raise ValueError("Curved-array shading must have shape (beams, elements)")
            shade_1JN = shade[None,:,:]
        else:
            # Accept a shared vector, including row/column-vector forms.
            if shade.shape not in ((nelements,),(nelements,1),(1,nelements)):
                raise ValueError("Line/panel shading must contain one weight per element")

            shade_1JN = shade.reshape(1,1,nelements)

        if (not np.issubdtype(shade_1JN.dtype,np.number) or np.iscomplexobj(shade_1JN)):
            raise ValueError("Shading weights must be real numbers")

        if (not np.all(np.isfinite(shade_1JN)) or np.any(shade_1JN < 0)):
            raise ValueError("Shading weights must be finite and nonnegative")

        if np.any(np.sum(shade_1JN,axis=2) <= 0):
            raise ValueError("Each shading vector must have positive total weight")

        # Broadcast the per-beam shading across frequencies.
        self.cbf_weight_QJN = shade_1JN * np.conj(replica_QJN)

        norm_QJ1 = np.sum(self.cbf_weight_QJN * replica_QJN,axis=2,keepdims=True)
        self.cbf_weight_QJN /= norm_QJ1
        self.array_heading_deg = float(geom.array_heading_deg) % 360.0
        self.heading_provider = heading_provider

    def process(self,block: FFTBlock) -> Iterator[BFBlock]:
        # beamform the data
        beamformed_QJ = np.sum(block.data_QN[:,np.newaxis,:] * self.cbf_weight_QJN,axis=2)

        if self.heading_provider is not None:
            array_heading_deg = self.heading_provider(block)
        elif block.array_heading_deg is not None:
            array_heading_deg = block.array_heading_deg
        else:
            array_heading_deg = self.array_heading_deg
        array_heading_deg = validate_array_heading_deg(array_heading_deg)

        yield BFBlock(data_QJ=beamformed_QJ,advance=block.advance,sequence=block.sequence,array_heading_deg=array_heading_deg)
