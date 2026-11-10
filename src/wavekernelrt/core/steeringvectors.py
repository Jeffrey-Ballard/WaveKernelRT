"""Near- and far-field steering-delay and replica calculations."""

import numpy as np
import warnings
from wavekernelrt.core.steeringvectors_utils import convert_rbe_to_xyz
from wavekernelrt.core.validate_utils import validate_beam_grid
from wavekernelrt.core.config import Config

class SteeringVectors:
    """Calculate propagation delays and complex replicas for an array.

    Parameters
    ----------
    geom : wavekernelrt.Geometry or BeamformingGeometry
        Array element positions, phase center, sound speed, heading, aperture,
        and far-field estimate.
    rbe_m_rad_J3 : array-like, shape (J, 3) or (3,)
        Complete ordered steering grid. Columns contain range in meters,
        bearing in radians, and elevation in radians.
    freq_Q : array-like, shape (Q,) or scalar
        Frequencies in hertz.
    precision : {"float32", "float64"}, default="float32"
        Precision used to store range differences and frequencies. Geometry
        distances are calculated in ``float64`` before conversion.
    flag_relative : bool, default=True
        If ``True``, rotate the geometry so bearings are relative to array
        heading. If ``False``, bearings use the absolute East-North-Up frame.

    Attributes
    ----------
    xyz_m_N3 : numpy.ndarray, shape (N, 3)
        Element positions used by the calculation, in meters.
    phase_center_m_13 : numpy.ndarray, shape (1, 3)
        Phase center in the same coordinate frame as ``xyz_m_N3``.
    soundspeed_mpers : numpy scalar
        Sound speed in meters per second.
    freq_Q : numpy.ndarray, shape (Q,)
        Frequencies in hertz, stored at the requested precision.
    time_delays_NJ : numpy.ndarray, shape (N, J)
        Lazy differential propagation delays in seconds.
    complex_phasors_QNJ : numpy.ndarray, shape (Q, N, J)
        Lazy replicas calculated as ``exp(-1j * 2*pi*f*delay)``, matching
        NumPy's forward-FFT convention for a delayed received signal.

    Raises
    ------
    ValueError
        If the steering grid is invalid, ``precision`` is unsupported, or the
        estimated numerical phase error exceeds the configured limit.

    Notes
    -----
    Ranges beyond 1.5 times the larger of the geometry far-field estimate and
    1000 meters are clipped for the distance calculation. This preserves
    far-field phase differences while avoiding needless loss of precision.
    The caller's steering grid is not modified.

    A ``RuntimeWarning`` is issued when the estimated phase error approaches
    the configured limit. Large apertures, high frequencies, and low sound
    speeds may require ``precision="float64"``.
    """
    def __init__(self,geom,rbe_m_rad_J3,freq_Q,precision="float32",flag_relative=True):
        
        freq_Q = np.atleast_1d(np.asarray(freq_Q))
        
        rbe_m_rad_J3 = np.asarray(rbe_m_rad_J3)
        if rbe_m_rad_J3.ndim==1:
            rbe_m_rad_J3 = rbe_m_rad_J3[None,:]

        # Make sure the grid is what WaveKernelRT will expect downstream
        validate_beam_grid(rbe_m_rad_J3)

        # if we want all the beams to be referenced to relative (which is almost always the case)
        # then the easiest thing to do is align the geometry on the y axis with a heading of north to
        # compute the 
        xyz_m_N3 = geom.xyz_m_N3
        phase_center_m_13 = geom.phase_center_m_13
        if flag_relative is True:
            if np.abs(geom.array_heading_deg)>1:
                print(f"Array heading is {geom.array_heading_deg}, rotating to align on 0 for relative bearings.")
                theta_rad = -geom.array_heading_deg*np.pi/180
                rotation_maxtrix_33 = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                    [np.sin(theta_rad),  np.cos(theta_rad), 0],[0,                  0,                 1]])
                xyz_m_N3 = geom.xyz_m_N3 @ rotation_maxtrix_33
                phase_center_m_13 = geom.phase_center_m_13 @ rotation_maxtrix_33

        # some constants we need
        self.xyz_m_N3 = xyz_m_N3 # save the geometry to update look directions later
        self.geom_far_field_range_m = geom.far_field_range_m # this call actuall generates the max_aperture field
        self.geom_max_aperture_m = geom.max_aperture_m
        self.phase_center_m_13 = phase_center_m_13
        self.soundspeed_mpers = geom.soundspeed_mpers.astype(precision)
        self.precision = precision

        self.compute_delta_range_NJ(rbe_m_rad_J3,freq_Q)

        self._time_delays_NJ = None
        self._complex_phasors_QNJ = None

    @property
    def time_delays_NJ(self):
        """numpy.ndarray: Differential element delays, shape ``(N, J)``."""
        if self._time_delays_NJ is None:
            self._time_delays_NJ = self._compute_time_delays()
        return self._time_delays_NJ

    @property
    def complex_phasors_QNJ(self):
        """numpy.ndarray: Complex steering replicas, shape ``(Q, N, J)``."""
        if self._complex_phasors_QNJ is None:
            self._complex_phasors_QNJ = self._compute_complex_phasors()
        return self._complex_phasors_QNJ
    

    def _compute_time_delays(self):
        self._time_delays_NJ = self._delta_range_m_NJ/self.soundspeed_mpers
        return self._time_delays_NJ
    
    def _compute_complex_phasors(self):
        self._complex_phasors_QNJ = np.exp(-1j * 2 * np.pi * self.freq_Q[:, None, None] * self.time_delays_NJ[None, :, :])
        return self._complex_phasors_QNJ

    def compute_delta_range_NJ(self,rbe_m_rad_J3,freq_Q):
        """Recalculate differential ranges for a steering grid.

        Parameters
        ----------
        rbe_m_rad_J3 : array-like, shape (J, 3)
            Range in meters followed by bearing and elevation in radians.
            The grid must follow WaveKernelRT's complete ordered-grid
            convention.
        freq_Q : numpy.ndarray, shape (Q,)
            Frequencies in hertz, used for the precision-error check and
            stored for subsequent replica calculation.

        Notes
        -----
        This lower-level method updates differential range and frequency but
        does not itself refresh cached delays or phasors. Most callers should
        use
        [`update_look_dir`][wavekernelrt.core.steeringvectors.SteeringVectors.update_look_dir].
        """

        # Clip only the calculation coordinates; preserve the caller's beam grid.
        rbe_m_rad_J3 = np.array(rbe_m_rad_J3, dtype=np.float64, copy=True)

        # figure out the far field so we can reduce any range to increase precision
        # we also need to run the far_field_range_m here so that the max_aperture_m
        # is popuplated later to check the esimated phase error due to precision rounding
        far_field_range_m = max((self.geom_far_field_range_m,1000))*1.5

        rbe_m_rad_J3[rbe_m_rad_J3[:,0]>far_field_range_m,0] = far_field_range_m

        # do a quick check to make sure percision choice isn't going to result in rounding error that we might care about
        # this is if the aperture is very large and very high frequency and or very low sound speed.
        # Most applications should be fine with float32. default to that for performance
        # -----------------------------------------------------------------------------------
        if self.precision=="float32":
            eps_abs = self.geom_max_aperture_m * np.finfo(np.float32).eps
        elif self.precision=="float64":
            eps_abs = self.geom_max_aperture_m * np.finfo(np.float64).eps
        else:
            raise ValueError(f"Precision must be either float32 or float64. Sent in: {self.precision}")

        # this is the estimated phase error that *could* occur due to the precision rounding
        phase_err_deg = 360*np.max(freq_Q)*eps_abs/self.soundspeed_mpers

        if phase_err_deg > Config.PHS_PREC_WARN_DEG and phase_err_deg<Config.PHS_PREC_ERR_DEG:
            msg = [f"\nestimated phase error ≈ {phase_err_deg:.2f} deg, {np.max(freq_Q):.1f} Hz, "
                   f"{self.soundspeed_mpers:.1f} m/s, Max Aperture {self.geom_max_aperture_m:.1f} m."]
            if self.precision=='float32':
                msg.append(f"Consider float64 for better accuracy")
            else:
                msg.append(f"Already at maximum precision")

            warnings.warn("\n".join(msg),RuntimeWarning,stacklevel=2)
            
        elif phase_err_deg>=Config.PHS_PREC_ERR_DEG:
            msg = [f"\nestimated phase error ≈ {phase_err_deg:.2f} deg, {np.max(freq_Q):.1f} Hz, "
                   f"{self.soundspeed_mpers:.1f} m/s, Max Aperture {self.geom_max_aperture_m:.1f} m."]
            if self.precision=='float32':
                msg.append(f"Must go to float64 for better accuracy")
            else:
                msg.append(f"Already at maximum precision")

            raise ValueError("\n".join(msg))
        # -----------------------------------------------------------------------------------

        # Convert range-bearing-elevation directions to Cartesian positions.
        vec_m_J3 = convert_rbe_to_xyz(rbe_m_rad_J3)

        # compute the distance from each element to the steer direction
        diff_NJ3 = self.xyz_m_N3.astype(np.float64)[:,None,:] - vec_m_J3.astype(np.float64)[None,:,:]
        dist_m_NJ  = np.linalg.norm(diff_NJ3,axis=2)

        # compute the distance from the steer direction to the phase center
        diff_phs_ctr_J3 = vec_m_J3.astype(np.float64) - self.phase_center_m_13.astype(np.float64)
        dist_phs_ctr_J1 = np.linalg.norm(diff_phs_ctr_J3,axis=1)

        # compute the range differences and hang onto those in the right precision
        self._delta_range_m_NJ = dist_m_NJ - dist_phs_ctr_J1
        if self.precision=="float32":
            self._delta_range_m_NJ = self._delta_range_m_NJ.astype(np.float32,copy=False)
            self.freq_Q = freq_Q.astype(np.float32)
        else:
            self.freq_Q = freq_Q

    def update_soundspeed_mpers(self,new_soundspeed_mpers):
        """Update sound speed and immediately refresh delays and replicas.

        Parameters
        ----------
        new_soundspeed_mpers : float or numpy scalar
            New sound speed in meters per second.
        """
        self.soundspeed_mpers = new_soundspeed_mpers
        self._time_delays_NJ = self._compute_time_delays()
        self._complex_phasors_QNJ = self._compute_complex_phasors()

    def update_look_dir(self,rbe_m_rad_J3,freq_Q=None):
        """Replace the steering grid and refresh delays and replicas.

        Parameters
        ----------
        rbe_m_rad_J3 : array-like, shape (J, 3) or (3,)
            Complete ordered range-bearing-elevation grid. Range is in meters;
            angles are in radians.
        freq_Q : array-like, shape (Q,) or scalar, optional
            Replacement frequencies in hertz. The existing frequencies are
            retained when omitted.
        """

        if freq_Q is None:
            freq_Q = self.freq_Q

        freq_Q = np.atleast_1d(np.asarray(freq_Q))
        
        rbe_m_rad_J3 = np.asarray(rbe_m_rad_J3)
        if rbe_m_rad_J3.ndim==1:
            rbe_m_rad_J3 = rbe_m_rad_J3[None,:]

        # Make sure the grid is what WaveKernelRT will expect downstream
        validate_beam_grid(rbe_m_rad_J3)

        self.compute_delta_range_NJ(rbe_m_rad_J3,freq_Q)

        # force these to recalculate
        self._time_delays_NJ = self._compute_time_delays()
        self._complex_phasors_QNJ = self._compute_complex_phasors()
