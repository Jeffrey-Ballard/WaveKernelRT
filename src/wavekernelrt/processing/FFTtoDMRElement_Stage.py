"""Element-space Dominant Mode Rejection beamforming stage.

The implementation is a robust DMR variant: it retains the dominant sample
covariance eigenpairs, adds a beam-dependent diagonal load, and normalizes the
result for unity response. This construction most closely follows Cox et al.
(2005), equations (22)--(23). The covariance estimate follows the ordinary
sample-covariance construction in Wage and Buck (2010), equation (5), during
startup, then uses an exponential update. See ``docs/references.md`` for full
citations and a precise implementation-to-literature mapping.
"""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from wavekernelrt.pipeline.blocks import FFTBlock,BFBlock

# import classes needed for beamforming
from wavekernelrt.core.geometry import Geometry,BeamformingGeometry
from wavekernelrt.core.steeringvectors import SteeringVectors
from wavekernelrt.core.dmr_utils import dmr_wng_batch_J
from wavekernelrt.core.validate_utils import validate_array_heading_deg

from numbers import Integral,Real
from math import ceil,isfinite

class FFTtoDMRElement_Stage:
    def __init__(self,geom: Geometry | BeamformingGeometry,shade: npt.ArrayLike,steer_rbe_rad_J3: npt.ArrayLike,
        freq_process: npt.ArrayLike,Dmax: int,Dmin: int,wngc_val_dB: float,wngc_tol_dB: float,eig_thresh: float,
        sinr_loss_dB: float = 1.0,write_weights: bool = False,weight_sink=None,
        weight_freq_range: tuple[int,int] | None = None,write_wng: bool = False,wng_sink=None,
        wng_freq_range: tuple[int,int] | None = None,weight_seq_stride: int = 1,wng_seq_stride: int = 1,
        heading_provider=None) -> None:

        if not isinstance(write_weights,bool):
            raise ValueError("write_weights must be True or False")

        if write_weights and weight_sink is None:
            raise ValueError("weight_sink is required when write_weights is True")

        if not write_weights and weight_sink is not None:
            raise ValueError("weight_sink requires write_weights=True")

        if not isinstance(write_wng,bool):
            raise ValueError("write_wng must be True or False")

        if write_wng and wng_sink is None:
            raise ValueError("wng_sink is required when write_wng is True")

        if not write_wng and wng_sink is not None:
            raise ValueError("wng_sink requires write_wng=True")

        for name,value in (("weight_seq_stride",weight_seq_stride),("wng_seq_stride",wng_seq_stride)):
            if (isinstance(value,bool) or not isinstance(value,Integral) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        if geom.xyz_m_N3.shape[0] < 2:
            raise ValueError("Beamforming requires at least two elements")

        for name,value in (("Dmin",Dmin),("Dmax",Dmax)):
            if (isinstance(value,bool) or not isinstance(value,Integral) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        if Dmin > Dmax:
            raise ValueError("Dmin must not exceed Dmax")

        if (isinstance(eig_thresh,bool) or not isinstance(eig_thresh,Real) or not isfinite(eig_thresh)
            or not 0 < eig_thresh <= 1):
            raise ValueError("eig_thresh must be finite and in (0, 1]")

        for name,value in (("wngc_val_dB",wngc_val_dB),("wngc_tol_dB",wngc_tol_dB)):
            if (isinstance(value,bool) or not isinstance(value,Real) or not isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")

        if wngc_tol_dB < 0:
            raise ValueError("wngc_tol_dB must be nonnegative")

        if (isinstance(sinr_loss_dB,bool) or not isinstance(sinr_loss_dB,Real) or not isfinite(sinr_loss_dB)
            or sinr_loss_dB <= 0):
            raise ValueError("sinr_loss_dB must be a positive finite real number")

        # Build one steering replica for every frequency, element, and beam.
        steer = SteeringVectors(geom,steer_rbe_rad_J3,freq_process,precision="float32",flag_relative=True)
        self.replica_QNJ = steer.complex_phasors_QNJ
        self.array_heading_deg = float(geom.array_heading_deg) % 360.0
        self.heading_provider = heading_provider

        replica_QJN = np.transpose(steer.complex_phasors_QNJ,(0,2,1))

        # Build normalized CBF weights for output during covariance startup.
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

        # Shared shading broadcasts across beams; curved-array shading is per
        # beam. Zero shade values remove elements from that beam's aperture.
        active_JN = np.broadcast_to(shade_1JN[0] > 0,(nbeams,nelements))

        self.active_indices_J = [np.flatnonzero(active_JN[jj]) for jj in range(nbeams)]

        for jj,indices in enumerate(self.active_indices_J):
            if indices.size < 2:
                raise ValueError(
                    f"DMR beam {jj} requires at least two active elements; "
                    f"received {indices.size}")
        # Beams with the same active aperture share a covariance submatrix and
        # therefore one eigendecomposition. This is an algebraic optimization,
        # not a change to the per-beam DMR calculation.
        beams_by_elements = {}

        for jj,indices in enumerate(self.active_indices_J):
            element_key = tuple(indices.tolist())
            beams_by_elements.setdefault(element_key,[]).append(jj)

        self.active_groups = [(np.asarray(element_key,dtype=np.intp),np.asarray(beam_indices,dtype=np.intp))
            for element_key,beam_indices in beams_by_elements.items()]

        self.cbf_weight_QJN = shade_1JN * np.conj(replica_QJN)

        norm_QJ1 = np.sum(self.cbf_weight_QJN * replica_QJN,axis=2,keepdims=True)

        self.cbf_weight_QJN /= norm_QJ1

        # The unshaded CBF gives the maximum WNG for each active aperture and
        # supplies the reference level for the relative WNG constraint.
        cbf_unshaded_weight_QJN = active_JN[None,:,:] * np.conj(replica_QJN)
        norm_unshaded_QJ1 = np.sum(cbf_unshaded_weight_QJN * replica_QJN,axis=2,keepdims=True)

        cbf_unshaded_weight_QJN /= norm_unshaded_QJ1

        self.cbf_wng_QJ = 1.0 / np.sum(np.abs(self.cbf_weight_QJN) ** 2,axis=2)
        self.cbf_wng_dB_QJ = 10 * np.log10(self.cbf_wng_QJ)

        # Hulbert and Wage (2022), Sec. V-C, Eqs. (72)-(73), approximate
        # matched-rank, strong-interferer, signal-free-training SINR loss as
        # L ~= 1 - R/T. Substituting Dmax for R and solving for T gives
        # ceil(Dmax / (1 - 10**(-sinr_loss_dB/10))). The substitution is an
        # engineering policy, not a performance guarantee for signal-
        # containing, correlated, nonstationary, or exponentially weighted
        # data.
        loss_fraction = 1.0 - 10.0**(-sinr_loss_dB / 10.0)
        self.training_snapshots = ceil(Dmax / loss_fraction)

        # For independent stationary snapshots, alpha = 2/(T+1) gives the EMA
        # variance-equivalent sample count (2-alpha)/alpha = T. The finite
        # startup mean and the subsequent EMA therefore share one reference
        # count, although their covariance estimates are not identical.
        self.alpha = 2.0 / (self.training_snapshots + 1)

        self.Dmax = Dmax
        self.Dmin = Dmin
        self.wngc_val_dB = wngc_val_dB
        self.wngc_tol_dB = wngc_tol_dB
        self.eig_thresh = eig_thresh
        self.sinr_loss_dB = sinr_loss_dB

        # Maintain an independent element-space CSM for each frequency bin.
        self.raw_csm_qNN = np.zeros((freq_process.shape[0],self.replica_QNJ.shape[1],self.replica_QNJ.shape[1]),dtype=np.complex64)
        self.count = 0

        self.write_weights = write_weights
        self.weight_sink = weight_sink
        self.weight_seq_stride = int(weight_seq_stride)
        if weight_freq_range is None:
            weight_freq_range = (0,self.replica_QNJ.shape[0])

        if (not isinstance(weight_freq_range,tuple) or len(weight_freq_range) != 2
            or any(isinstance(value,bool) or not isinstance(value,Integral) for value in weight_freq_range)):
            raise ValueError("weight_freq_range must contain two integer indices")

        self.weight_idx0,self.weight_idx1 = map(int,weight_freq_range)
        if not (0 <= self.weight_idx0 < self.weight_idx1 <= self.replica_QNJ.shape[0]):
            raise ValueError("weight_freq_range is outside the frequency grid")

        self.write_wng = write_wng
        self.wng_sink = wng_sink
        self.wng_seq_stride = int(wng_seq_stride)
        if wng_freq_range is None:
            wng_freq_range = (0,self.replica_QNJ.shape[0])

        if (not isinstance(wng_freq_range,tuple) or len(wng_freq_range) != 2
            or any(isinstance(value,bool) or not isinstance(value,Integral) for value in wng_freq_range)):
            raise ValueError("wng_freq_range must contain two integer indices")

        self.wng_idx0,self.wng_idx1 = map(int,wng_freq_range)
        if not (0 <= self.wng_idx0 < self.wng_idx1 <= self.replica_QNJ.shape[0]):
            raise ValueError("wng_freq_range is outside the frequency grid")

    def process(self,block: FFTBlock) -> Iterator[BFBlock]:
        if self.heading_provider is not None:
            array_heading_deg = self.heading_provider(block)
        elif block.array_heading_deg is not None:
            array_heading_deg = block.array_heading_deg
        else:
            array_heading_deg = self.array_heading_deg
        array_heading_deg = validate_array_heading_deg(array_heading_deg)

        weights_QJN = None
        if (self.write_weights and self.count % self.weight_seq_stride == 0):
            weights_QJN = np.zeros((self.weight_idx1 - self.weight_idx0,self.replica_QNJ.shape[2],self.replica_QNJ.shape[1]),
                dtype=self.cbf_weight_QJN.dtype)

        wng_QJ = None
        if self.write_wng and self.count % self.wng_seq_stride == 0:
            wng_QJ = np.zeros((self.wng_idx1 - self.wng_idx0,self.replica_QNJ.shape[2]),dtype=np.float32)

        if self.count<self.training_snapshots:
            # Accumulate an equal-weight sample CSM. Once all startup records
            # have arrived this is (1 / T) sum_t x_t x_t^H, as in Wage and
            # Buck (2010), equation (5). Emit CBF until that estimate is ready.
            self.raw_csm_qNN += np.einsum("qn,qm->qnm",block.data_QN,block.data_QN.conj(),optimize=True)/self.training_snapshots

            beamformed_QJ = np.sum(block.data_QN[:,np.newaxis,:] * self.cbf_weight_QJN,axis=2)
            if weights_QJN is not None:
                weights_QJN[...] = self.cbf_weight_QJN[self.weight_idx0:self.weight_idx1]
            if wng_QJ is not None:
                wng_QJ[...] = self.cbf_wng_QJ[self.wng_idx0:self.wng_idx1]
        else:
            # Update each frequency-bin CSM with the current rank-one snapshot.
            snapshot_csm_qNN = np.einsum("qn,qm->qnm",block.data_QN,block.data_QN.conj(),optimize=True)

            # Exponential update: R <- (1 - alpha) R + alpha x x^H.
            self.raw_csm_qNN *= (1-self.alpha)
            self.raw_csm_qNN += self.alpha*snapshot_csm_qNN

            Q = self.replica_QNJ.shape[0]
            beamformed_QJ = np.empty((Q,self.replica_QNJ.shape[2]),dtype=block.data_QN.dtype)
            for qq in range(Q):
                replica_NJ = self.replica_QNJ[qq]
                raw_csm_NN = self.raw_csm_qNN[qq]

                # The CSM is Hermitian, so eigh is sufficient and returns real
                # eigenvalues. Process each distinct active aperture once.
                for element_indices,beam_indices in self.active_groups:
                    # A: active elements; G: beams sharing those elements.
                    csm_AA = raw_csm_NN[np.ix_(element_indices,element_indices)]
                    if not np.any(csm_AA):
                        # No measured energy for this group; use the startup CBF weights.
                        weights_GA = self.cbf_weight_QJN[qq][np.ix_(beam_indices,element_indices)]

                        beamformed_QJ[qq,beam_indices] = weights_GA @ block.data_QN[qq,element_indices]
                        if (weights_QJN is not None and self.weight_idx0 <= qq < self.weight_idx1):
                            weights_QJN[qq - self.weight_idx0][np.ix_(beam_indices,element_indices)] = weights_GA
                        if (wng_QJ is not None and self.wng_idx0 <= qq < self.wng_idx1):
                            wng_QJ[qq - self.wng_idx0,beam_indices] = self.cbf_wng_QJ[qq,beam_indices]
                        continue

                    replica_AG = replica_NJ[np.ix_(element_indices,beam_indices)]

                    eigvals,U = np.linalg.eigh(csm_AA)

                    S = np.maximum(eigvals[::-1],0.0)
                    U = U[:,::-1]

                    # Local rank heuristic: reach the configured cumulative
                    # energy fraction, then conservatively retain one more
                    # mode. Campos Anchieta and Buck (2022), section III, and
                    # Hulbert and Wage (2025), section V, motivate studying
                    # overestimated rank. The cumulative energy fraction approach
                    # is native to WaveKernelRT to make it more applicable across
                    # multiple data sets.
                    cumulative_eigen_energy = np.cumsum(S) / np.sum(S)

                    D = np.searchsorted(cumulative_eigen_energy,self.eig_thresh) + 2
                    D = int(np.clip(D,self.Dmin,self.Dmax))

                    # Rank cannot exceed the active-aperture dimension.
                    D = min(D,element_indices.size)

                    U_AD = U[:,:D]
                    S_D = S[:D]

                    z_DG = U_AD.conj().T @ replica_AG
                    b_AG = replica_AG - U_AD @ z_DG

                    # Choose one diagonal load per beam to satisfy the relative
                    # WNG target. Direction-dependent loading follows the norm-
                    # constrained construction of Cox et al. (2005), equations
                    # (20)--(23). WaveKernelRT uses epsilon as the total load.
                    epsilon_G,wngc_dmr_G = dmr_wng_batch_J(S_D,z_DG,replica_AG,self.wngc_val_dB,
                        wngc_tol_dB=self.wngc_tol_dB,max_iterations=10,epsilon_tolerance=1e-1)

                    # Apply (U_D diag(S_D) U_D^H + epsilon I)^-1 to each
                    # steering replica in the retained eigenbasis. Multiplying
                    # by epsilon avoids explicit division in the discarded
                    # subspace; that common factor cancels on normalization.
                    scaled_inverse_DG = epsilon_G[None,:] / (S_D[:,None] + epsilon_G[None,:])
                    scaled_Rinv_v_AG = U_AD @ (z_DG * scaled_inverse_DG) + b_AG

                    # Normalize for unity response in the steering direction,
                    # the rank-one specialization of Cox et al. (2005), Eq. 22.
                    norm_G = np.sum(np.conj(replica_AG) * scaled_Rinv_v_AG,axis=0)
                    w_dmr_AG = scaled_Rinv_v_AG / norm_G[None,:]

                    beamformed_QJ[qq,beam_indices] = block.data_QN[qq,element_indices] @ np.conj(w_dmr_AG)

                    if (weights_QJN is not None and self.weight_idx0 <= qq < self.weight_idx1):
                        weights_QJN[qq - self.weight_idx0][np.ix_(beam_indices,element_indices)] = np.conj(w_dmr_AG).T

                    if (wng_QJ is not None and self.wng_idx0 <= qq < self.wng_idx1):
                        wng_QJ[qq - self.wng_idx0,beam_indices] = wngc_dmr_G

        if weights_QJN is not None:
            self.weight_sink.write(weights_QJN,array_heading_deg)

        if wng_QJ is not None:
            self.wng_sink.write(BFBlock(data_QJ=wng_QJ,advance=block.advance,sequence=block.sequence,array_heading_deg=array_heading_deg))

        self.count += 1

        yield BFBlock(data_QJ=beamformed_QJ,advance=block.advance,sequence=block.sequence,array_heading_deg=array_heading_deg)

    def close(self) -> None:
        try:
            if self.weight_sink is not None:
                self.weight_sink.close()
        finally:
            if self.wng_sink is not None:
                self.wng_sink.close()
