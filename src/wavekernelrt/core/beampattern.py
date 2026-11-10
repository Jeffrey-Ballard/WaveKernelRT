"""Conventional beam-weight construction and beampattern evaluation."""

import numpy as np
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3
from wavekernelrt.core.steeringvectors import SteeringVectors
from wavekernelrt.core.general_utils import find_closest, find_closest_bearing
from wavekernelrt.core.shading import Shading
from wavekernelrt.pipeline.configs import ShadingConfig

import matplotlib.pyplot as plt
from numbers import Integral


class BeamPattern:
    """Construct conventional weights and evaluate array spatial response.

    Parameters
    ----------
    geom : wavekernelrt.Geometry or BeamformingGeometry
        Array geometry used for steering and shading.
    freq : real scalar or array-like, optional
        Nonnegative frequencies in hertz. When omitted, the geometry's
        spacing-derived design frequency is used.
    Nbearings : positive int, optional
        Number of bearings in the automatically constructed steering grid.
        When omitted, a plotting density is estimated from aperture and the
        highest frequency.
    Nelevations : positive int, optional
        Number of elevations. Line arrays default to one; other geometries
        estimate a plotting density when this is omitted.
    focus_range_ft : positive float, default=100000
        Steering and arrival-grid range in feet.
    flag_relative : bool, default=True
        If ``True``, bearings are relative to array heading. If ``False``,
        bearings are geographic, clockwise from north.
    precision : {"float32", "float64"}, default="float32"
        Steering-vector range-difference precision.

    Attributes
    ----------
    freq : numpy.ndarray, shape (Q,)
        Evaluation frequencies in hertz.
    Nbearings, Nelevations : int
        Steering-grid dimensions.
    focus_range_ft : float
        Steering-grid range in feet.
    steer_rbe_rad_J3 : numpy.ndarray, shape (J, 3)
        Lazy steering grid with range in meters and angles in radians.
    steer : wavekernelrt.SteeringVectors
        Steering replicas for ``steer_rbe_rad_J3`` and ``freq``.
    cbf_weight_QNJ : numpy.ndarray, shape (Q, N, J)
        Lazy conventional multiplication weights after shading is configured.

    Raises
    ------
    ValueError
        If frequencies, grid counts, focus range, or precision are invalid, or
        automatic grid sizing is requested for an all-zero frequency array.

    Notes
    -----
    Line and panel grids span bearings from 0 through 180 degrees. Cylinder
    and sphere grids span 0 through 360 degrees without duplicating the
    endpoint. Elevations span -90 through 90 degrees unless only one elevation
    is requested, in which case it is zero.

    Configure shading with
    [`set_shading_config`][wavekernelrt.core.beampattern.BeamPattern.set_shading_config]
    before requesting
    [`cbf_weight_QNJ`][wavekernelrt.core.beampattern.BeamPattern.cbf_weight_QNJ].
    Line and panel arrays may alternatively accept a custom shared vector
    through [`set_shading`][wavekernelrt.core.beampattern.BeamPattern.set_shading];
    curved arrays require a recipe so their visible aperture can be recomputed
    for each beam.
    """
    def __init__(self,geom,freq=None,Nbearings=None,Nelevations=None,focus_range_ft=100000,flag_relative=True,precision="float32"):

        if precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")

        self.precision = precision

        if Nbearings is not None:
            Nbearings = self._validate_grid_count(Nbearings, "Nbearings")

        if Nelevations is not None:
            Nelevations = self._validate_grid_count(Nelevations, "Nelevations")

        focus_range_ft = self._validate_focus_range(focus_range_ft)

        if freq is None:
            if geom.array_type == "line":
                freq = np.round(geom.design_freq)
            else:
                freq = np.round(np.min((geom.design_freq_azimuth,geom.design_freq_elevation)))

        freq = self._validate_freq(freq)

        needs_frequency_for_grid = Nbearings is None or (Nelevations is None and geom.array_type != "line")
        
        if needs_frequency_for_grid and not np.any(freq > 0):
            raise ValueError(
                "Automatic grid sizing requires a positive frequency; "
                "supply explicit grid counts for an all-zero frequency array")

        # set num of bearings or auto select based on the aperture
        if Nbearings is None:
            if geom.array_type == "line":
                # Natural beams (~3 dB scalloping)
                Nbearings = int(10*np.round(2*geom.aperture_m / (geom.soundspeed_mpers/freq.max())))
            else:
                # Natural beams (~3 dB scalloping)
                Nbearings = int(10*np.round(2*geom.width_aperture_m / (geom.soundspeed_mpers/freq.max())))

            # *2 since the natural beams is across 0-180, but this goes 0-360
            if geom.array_type in ("sphere", "cylinder"):
                Nbearings *= 2

            # If even, make it odd to we can pull out main response axis for a likely broadside bean
            if Nbearings%2==0:
                Nbearings+=1

        # set the num of elevations or auto select based on the aperture
        if Nelevations is None:
            if geom.array_type == "line":
                # no elevations for line array for displaying beampattern purposes
                Nelevations = 1
            else:
                # Natural beams (~3 dB scalloping)
                Nelevations = int(10*np.round(2*geom.height_aperture_m / (geom.soundspeed_mpers/freq.max())))

            # If even, make it odd to we can pull out main response axis for a likely at 0 elevation
            if Nelevations%2==0:
                Nelevations+=1
            
        self.geom = geom
        self.freq = np.atleast_1d(freq)
        self.Nbearings = Nbearings
        self.Nelevations = Nelevations
        self.focus_range_ft = focus_range_ft
        self.flag_relative = flag_relative

        self._steer_rbe_rad_J3 = None
        self._shade_N = None
        self._shading_config = None
        self._cbf_weight_QNJ = None

        self.steer =  SteeringVectors(self.geom,self.steer_rbe_rad_J3,freq,precision=self.precision,flag_relative=self.flag_relative)

    @property
    def cbf_weight_QNJ(self):
        """numpy.ndarray: Normalized CBF weights with shape ``(Q, N, J)``."""
        if self._cbf_weight_QNJ is None:
            replica_QNJ = self.steer.complex_phasors_QNJ
            is_curved = self.geom.array_type in ("sphere", "cylinder")

            if self._shading_config is not None:
                config = self._shading_config
                shading = Shading(self.geom)

                match config.window_type:
                    case "uniform":
                        if is_curved:
                            shade = shading.uniform_shading_for_beams(self.steer_rbe_rad_J3,flag_relative=self.flag_relative)
                        else:
                            shade = shading.uniform_shading_N.reshape(-1)

                    case "hamming":
                        if is_curved:
                            shade = shading.hamming_shading_for_beams(self.steer_rbe_rad_J3,flag_relative=self.flag_relative)
                        else:
                            shade = shading.hamming_shading_N.reshape(-1)

                    case "taylor":
                        shading.set_taylor_params(sll_dB=config.sll_dB,nbar=config.nbar)

                        if is_curved:
                            shade = shading.taylor_shading_for_beams(self.steer_rbe_rad_J3,flag_relative=self.flag_relative)
                        else:
                            shade = shading.taylor_shading_N.reshape(-1)

                    case _:
                        raise ValueError("Unsupported shading window type")

            elif self._shade_N is not None and not is_curved:
                shade = self._shade_N

            else:
                raise ValueError("Set a shading recipe before requesting curved-array "
                                "weights; line/panel arrays also accept a shared vector")

            if is_curved:
                # (J, N) -> (1, N, J)
                shade_1NJ = shade.T[None, :, :]
            else:
                # (N,) -> (1, N, 1)
                shade_1NJ = shade[None, :, None]

            weights_QNJ = shade_1NJ * np.conj(replica_QNJ)
            normalization_Q1J = np.sum(weights_QNJ * replica_QNJ,axis=1,keepdims=True)

            self._cbf_weight_QNJ = weights_QNJ / normalization_Q1J

        return self._cbf_weight_QNJ

    @property
    def steer_rbe_rad_J3(self):
        """numpy.ndarray: Ordered steering grid with shape ``(J, 3)``."""
        if self._steer_rbe_rad_J3 is None:
            self._steer_rbe_rad_J3 = self._compute_steer_rbe_rad_J3()
        return self._steer_rbe_rad_J3

    @staticmethod
    def _validate_grid_count(value, name):
        if (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
            raise ValueError(f"{name} must be a positive integer")

        return int(value)

    @staticmethod
    def _validate_freq(freq):
        freq = np.asarray(freq)

        if (freq.ndim > 1 or freq.size == 0 or not np.issubdtype(freq.dtype, np.number) or np.iscomplexobj(freq)):
            raise ValueError("freq must be a real scalar or a nonempty 1D array")

        if not np.all(np.isfinite(freq)) or np.any(freq < 0):
            raise ValueError("Frequencies must be finite and nonnegative")

        return np.atleast_1d(freq).astype(np.float64, copy=True)

    @staticmethod
    def _validate_focus_range(focus_range_ft):
        value = np.asarray(focus_range_ft)

        if (value.ndim != 0 or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)):
            raise ValueError("focus_range_ft must be a real scalar")

        if not np.isfinite(value) or value <= 0:
            raise ValueError("focus_range_ft must be finite and positive")

        return float(value)

    @staticmethod
    def _validate_single_look_direction(look_dir_rad_13):
        look_dir_rad_13 = np.asarray(look_dir_rad_13)

        if look_dir_rad_13.shape not in ((3,), (1, 3)):
            raise ValueError("look_dir_rad_13 must have shape (3,) or (1, 3)")

        if (not np.issubdtype(look_dir_rad_13.dtype, np.number)
            or np.iscomplexobj(look_dir_rad_13)
            or not np.all(np.isfinite(look_dir_rad_13))):
            raise ValueError("Look direction must contain finite real values")

        return look_dir_rad_13.astype(np.float64, copy=True).reshape(3)

    @staticmethod
    def _bearing_plot_limits(bearings_deg):
        lower = float(np.min(bearings_deg))
        upper = float(np.max(bearings_deg))

        if lower == upper:
            return lower - 0.5, upper + 0.5

        return lower, upper

    @staticmethod
    def _bearing_plot_ticks(bearings_deg):
        lower = float(np.min(bearings_deg))
        upper = float(np.max(bearings_deg))

        if lower == upper:
            return np.array([lower])

        start = np.ceil(lower / 30.0) * 30.0
        stop = np.floor(upper / 30.0) * 30.0
        ticks = np.arange(start, stop + 1.0, 30.0)

        # Narrow sectors may contain no multiple of 30 degrees.
        if ticks.size == 0:
            return np.array([lower, upper])

        return ticks

    def set_steer_params(self,freq=None,Nbearings=None,Nelevations=None,focus_range_ft=None):
        """Update steering-grid parameters and rebuild steering replicas.

        Parameters
        ----------
        freq : real scalar or array-like, optional
            Replacement nonnegative frequencies in hertz.
        Nbearings, Nelevations : positive int, optional
            Replacement grid dimensions.
        focus_range_ft : positive float, optional
            Replacement focus range in feet.

        Notes
        -----
        Omitted values retain their current settings. All candidate values are
        validated before the existing state is changed. Previously configured
        shading is retained, while cached beam weights are invalidated.
        """

        # resuse the construction validation to make sure everything passes before updating
        # otherwise if it fails partway through, some settings will be from the new steer params
        # while others will be from the old
        candidate = BeamPattern(self.geom,freq=self.freq if freq is None else freq,
            Nbearings=self.Nbearings if Nbearings is None else Nbearings,
            Nelevations=self.Nelevations if Nelevations is None else Nelevations,
            focus_range_ft=self.focus_range_ft if focus_range_ft is None else focus_range_ft,
            flag_relative=self.flag_relative,precision=self.precision)

        # Construction succeeded; adopt the new steering state.
        self.freq = candidate.freq
        self.Nbearings = candidate.Nbearings
        self.Nelevations = candidate.Nelevations
        self.focus_range_ft = candidate.focus_range_ft
        self.azimuth_deg_A = candidate.azimuth_deg_A
        self.elevation_deg_E = candidate.elevation_deg_E
        self._steer_rbe_rad_J3 = candidate._steer_rbe_rad_J3
        self.steer = candidate.steer

        self._cbf_weight_QNJ = None

    def set_shading_config(self, config: ShadingConfig):
        """Select a generated spatial-shading recipe.

        Parameters
        ----------
        config : wavekernelrt.ShadingConfig
            Uniform, Hamming, or Taylor shading configuration.

        Raises
        ------
        TypeError
            If ``config`` is not a
            [`ShadingConfig`][wavekernelrt.pipeline.configs.ShadingConfig].
        """
        if not isinstance(config, ShadingConfig):
            raise TypeError("config must be a ShadingConfig")

        self._shading_config = config
        self._shade_N = None
        self._cbf_weight_QNJ = None
    
    def set_shading(self, shade_N):
        """Set one custom shading vector for a line or panel array.

        Parameters
        ----------
        shade_N : array-like, shape (N,), (N, 1), or (1, N)
            Real, finite, nonnegative element weights with positive sum. The
            conventional-weight calculation performs its own look-direction
            normalization, so the input need not already sum to one.

        Raises
        ------
        ValueError
            If the geometry is curved or the vector has invalid shape or
            values.
        """

        if self.geom.array_type in ("sphere", "cylinder"):
            raise ValueError("Use set_shading_config() for per-beam curved-array shading")

        shade_N = np.asarray(shade_N)
        nelements = self.geom.xyz_m_N3.shape[0]

        if shade_N.shape not in ((nelements,), (nelements, 1), (1, nelements)):
            raise ValueError("Shading must contain one weight per element")

        if (not np.issubdtype(shade_N.dtype, np.number) or np.iscomplexobj(shade_N)):
            raise ValueError("Shading weights must be real numbers")

        if (not np.all(np.isfinite(shade_N)) or np.any(shade_N < 0) or np.sum(shade_N) <= 0):
            raise ValueError("Shading must be finite, nonnegative, and have positive total weight")

        self._shade_N = shade_N.reshape(-1).copy()
        self._shading_config = None
        self._cbf_weight_QNJ = None

    def _compute_steer_rbe_rad_J3(self):
         #create the rbe grid to steer to
        if self.geom.array_type in ("sphere","cylinder"):
            azimuth_deg_A = np.linspace(0.0, 360.0, self.Nbearings, endpoint=False)
        else:
            azimuth_deg_A = np.arccos(np.linspace(1,-1,self.Nbearings))*180/np.pi

        if self.Nelevations==1:
            elevation_deg_E = np.array([0.0])
        else:
            elevation_deg_E = np.arcsin(np.linspace(-1,1,self.Nelevations))*180/np.pi

        self.azimuth_deg_A = azimuth_deg_A
        self.elevation_deg_E = elevation_deg_E
        self._steer_rbe_rad_J3 = construct_rbe_X3(self.focus_range_ft,azimuth_deg_A,elevation_deg_E)
        
        return self._steer_rbe_rad_J3
    
    def _arrival_visibility_NJ(self, directions_J3):
        """Return the horizontal-aperture mask for each arrival direction."""
        directions_J3 = np.atleast_2d(np.asarray(directions_J3, dtype=np.float64))
        nelements = self.geom.xyz_m_N3.shape[0]

        if self.geom.array_type not in ("sphere", "cylinder"):
            return np.ones((nelements, directions_J3.shape[0]), dtype=bool)

        shading = Shading(self.geom)
        visible_NJ = np.empty((nelements, directions_J3.shape[0]), dtype=bool)

        for jj, look in enumerate(directions_J3):
            visible_NJ[:, jj] = shading.visible_elements(look,flag_relative=self.flag_relative,require_visible_elements=False)

        return visible_NJ

    def compute_cbf_weight_QNJ(self,shade_N):
       """Set shared shading and return conventional beam weights.

       Parameters
       ----------
       shade_N : array-like
           Custom line- or panel-array element shading accepted by
           [`set_shading`][wavekernelrt.core.beampattern.BeamPattern.set_shading].

       Returns
       -------
       numpy.ndarray, shape (Q, N, J)
           Normalized complex multiplication weights.
       """

       self.set_shading(shade_N)
       return self.cbf_weight_QNJ
    
    def compute_beampattern_weight_sweep_QNJ(self,weight_QNJ,look_dir_rad_13):
        """Evaluate a bank of weights for one fixed arrival direction.

        Parameters
        ----------
        weight_QNJ : array-like, shape (Q, N, J)
            Finite multiplication weights for ``J`` beams.
        look_dir_rad_13 : array-like, shape (3,) or (1, 3)
            Fixed arrival direction as range in meters, bearing in radians,
            and elevation in radians.

        Returns
        -------
        numpy.ndarray, shape (Q, J)
            Response magnitude for every frequency and beam.

        Notes
        -----
        A weight sweep moves a bank of receiving beams past a stationary
        source. For the complementary fixed-weight evaluation, use
        [`compute_beampattern_look_sweep_JQ`][wavekernelrt.core.beampattern.BeamPattern.compute_beampattern_look_sweep_JQ].
        """

        look_dir_rad_13 = self._validate_single_look_direction(look_dir_rad_13)

        weight_QNJ = np.asarray(weight_QNJ)

        expected_shape_QN = (self.freq.size,self.geom.xyz_m_N3.shape[0])

        if  (weight_QNJ.ndim != 3 or weight_QNJ.shape[:2] != expected_shape_QN or weight_QNJ.shape[2] == 0):
            raise ValueError(
                "Beam weights must have shape "
                f"({expected_shape_QN[0]}, {expected_shape_QN[1]}, J), "
                "with at least one beam")

        if not np.issubdtype(weight_QNJ.dtype, np.number) or not np.all(np.isfinite(weight_QNJ)):
            raise ValueError("Beam weights must be finite numbers")

        look_dir =  SteeringVectors(self.geom,look_dir_rad_13,self.freq,precision=self.precision,flag_relative=self.flag_relative)

        # if the array is baffled, which is assumed for sphere or cylinder then we need to shut off the arrivals coming in from behind
        if self.geom.array_type in ("sphere","cylinder"):
            visible_N1 = self._arrival_visibility_NJ(look_dir_rad_13)
            arrival_QN1 = (look_dir.complex_phasors_QNJ * visible_N1[None, :, :])

            self.beampattern_QJ = np.abs(np.sum(arrival_QN1 * weight_QNJ, axis=1))
        else:
            self.beampattern_QJ = np.abs(np.sum(look_dir.complex_phasors_QNJ *  weight_QNJ,axis=1))
        
        return self.beampattern_QJ

    def compute_beampattern_look_sweep_JQ(self, weight_QN, look_dir_rad_J3):
        """Evaluate one fixed set of weights over arrival directions.

        Parameters
        ----------
        weight_QN : array-like, shape (Q, N)
            Finite multiplication weights for one beam.
        look_dir_rad_J3 : array-like, shape (J, 3)
            Arrival directions with range in meters and bearing and elevation
            in radians.

        Returns
        -------
        numpy.ndarray, shape (J, Q)
            Response magnitude for every arrival direction and frequency.

        Notes
        -----
        A look sweep holds one receiving beam fixed and moves the source. This
        is the natural evaluation for externally calculated or adaptive
        weights such as one DMR weight vector.
        """
        
        weight_QN = np.asarray(weight_QN)

        expected_shape_QN = (self.freq.size,self.geom.xyz_m_N3.shape[0])
        if weight_QN.shape != expected_shape_QN:
            raise ValueError(f"Fixed beam weights must have shape {expected_shape_QN}")

        if (not np.issubdtype(weight_QN.dtype, np.number) or not np.all(np.isfinite(weight_QN))):
            raise ValueError("Beam weights must be finite numbers")

        look_dir_rad_J3 = np.atleast_2d(np.asarray(look_dir_rad_J3))

        look = SteeringVectors(self.geom,look_dir_rad_J3,self.freq,precision=self.precision,flag_relative=self.flag_relative)

        replica_QNJ = look.complex_phasors_QNJ

        if self.geom.array_type in ("sphere", "cylinder"):
            visible_NJ = self._arrival_visibility_NJ(look_dir_rad_J3)
            replica_QNJ = replica_QNJ * visible_NJ[None, :, :]

        response_QJ = np.sum(weight_QN[:, :, None] * replica_QNJ,axis=1)

        self.beampattern_JQ = np.abs(response_QJ).T
        return self.beampattern_JQ
    
    def plot_beampattern_weight_sweep(self,weight_QNJ,look_dir_rad_13,freq_display=None):
        """Plot a weight sweep on this object's steering grid.

        Parameters
        ----------
        weight_QNJ : array-like, shape (Q, N, J)
            Beam bank to evaluate. ``J`` must equal the number of directions
            in
            [`steer_rbe_rad_J3`][wavekernelrt.core.beampattern.BeamPattern.steer_rbe_rad_J3].
        look_dir_rad_13 : array-like, shape (3,) or (1, 3)
            Fixed arrival direction with range in meters and angles in
            radians. It also selects the displayed slices for a 2-D grid.
        freq_display : float, optional
            Frequency to display in hertz. The nearest configured frequency is
            used; the first configured frequency is used when omitted.

        Returns
        -------
        tuple
            ``(figure, axes)`` for a single elevation, or ``(figure,
            map_axes, horizontal_axes, vertical_axes)`` for a 2-D grid.

        Notes
        -----
        A weight sweep holds one signal-arrival direction fixed and evaluates
        every weight vector in the beam bank against that same arrival. Use it
        to answer "which beams respond to a source at this location?", for
        example when inspecting a complete conventional or adaptive beam bank.
        Each beam on the ``J`` axis is plotted at its corresponding direction
        in this object's steering grid.

        To inspect the spatial response of one fixed weight vector instead,
        use
        [`plot_beampattern_look_sweep`][wavekernelrt.core.beampattern.BeamPattern.plot_beampattern_look_sweep].

        Magnitudes are displayed in decibels with a -60 to 0 dB plot range.
        The numerical response is also stored in ``beampattern_QJ``.
        """

        look_dir_rad_13 = self._validate_single_look_direction(look_dir_rad_13)

        if freq_display is None:
            idx_q = 0
        else:
            if np.asarray(freq_display).ndim != 0:
                raise ValueError("freq_display must be a scalar frequency in Hz")

            freq_display = self._validate_freq(freq_display)[0]
            idx_q = find_closest(self.freq, freq_display)

        weight_QNJ = np.asarray(weight_QNJ)
        expected_beams = self.steer_rbe_rad_J3.shape[0]

        if weight_QNJ.ndim != 3:
            raise ValueError("Beam weights must have shape (frequency, element, beam)")

        if weight_QNJ.shape[2] != expected_beams:
            raise ValueError(
                f"Plotting grid requires {expected_beams} beams; "
                f"received {weight_QNJ.shape[2]}")
    
        # Weight sweep: keep this one source direction stationary while the J
        # weight vectors move the receiving beam across the object's steering
        # grid. The response's J axis therefore maps directly onto that grid.
        self.beampattern_QJ = self.compute_beampattern_weight_sweep_QNJ(weight_QNJ,look_dir_rad_13)
        beampattern_dB_QJ = 20.0 * np.log10(np.maximum(self.beampattern_QJ, 1e-12))

        bearing_label = "Relative Bearing (deg)" if self.flag_relative else "Geographic Bearing (deg)"

        if np.asarray(self.elevation_deg_E).size == 1:
            fig,ax = plt.subplots()
            ax.plot(self.azimuth_deg_A,beampattern_dB_QJ[idx_q,:])
            ax.set_ylim(-60,0)
            ax.set_xlabel(bearing_label)
            ax.set_ylabel("dB")
            ax.set_title(f"Beampattern at {self.freq[idx_q]} Hz")
            ax.set_xticks(self._bearing_plot_ticks(self.azimuth_deg_A))
            ax.set_xlim(*self._bearing_plot_limits(self.azimuth_deg_A))
            ax.grid(True)

            return fig, ax
        else:
            E = np.asarray(self.elevation_deg_E).size
            A = np.asarray(self.azimuth_deg_A).size
            Q = np.asarray(self.freq).size
            
            # The steering grid stores elevation fastest. Reshape the selected
            # frequency into an elevation-by-bearing image for display.
            beampattern_dB_EA = np.transpose(beampattern_dB_QJ[idx_q,:].reshape(A,E),(1, 0))
            
            fig = plt.figure(figsize=(10,8), constrained_layout=True)
            gs = fig.add_gridspec(2,3,width_ratios=[1, 0.05,1])

            ax00 = fig.add_subplot(gs[0,0])
            cax = fig.add_subplot(gs[0,1])
            ax01 = fig.add_subplot(gs[0,2])
            ax10 = fig.add_subplot(gs[1,0])

            pcm = ax00.pcolormesh(self.azimuth_deg_A,self.elevation_deg_E,beampattern_dB_EA,cmap="jet",shading="auto")

            fig.colorbar(pcm,cax=cax)
            ax00.set_ylim(self.elevation_deg_E.min(),self.elevation_deg_E.max())

            ax00.set_xlabel(bearing_label)
            ax00.set_ylabel("Elevation (deg)")
            ax00.set_yticks(np.arange(-90,90+30,30))
            ax00.set_xticks(self._bearing_plot_ticks(self.azimuth_deg_A))
            ax00.set_xlim(*self._bearing_plot_limits(self.azimuth_deg_A))
            pcm.set_clim(-60,0)
            ax00.set_title(f"2D Beampattern at {self.freq[idx_q]} Hz")

            # Show cuts through the grid point nearest the fixed source. These
            # cuts make it easy to see which beams respond around that source
            # bearing and elevation.
            look_elevation_deg = look_dir_rad_13[2]*180/np.pi
            idx_e = find_closest(self.elevation_deg_E, look_elevation_deg)
            look_elevation_deg = self.elevation_deg_E[idx_e]
            ax10.plot(self.azimuth_deg_A,np.squeeze(beampattern_dB_EA[idx_e,:]))
            ax10.set_xlabel(bearing_label)
            ax10.set_ylabel("dB")
            ax10.set_title(f"Horizontal beam slice at Elevation={np.round(look_elevation_deg,1)} deg")
            ax10.set_ylim(-60,0)
            ax10.grid(True)
            ax10.set_xticks(self._bearing_plot_ticks(self.azimuth_deg_A))
            ax10.set_xlim(*self._bearing_plot_limits(self.azimuth_deg_A))

            look_bearing_deg = look_dir_rad_13[1]*180/np.pi
            idx_a = find_closest_bearing(self.azimuth_deg_A, look_bearing_deg)
            look_bearing_deg = self.azimuth_deg_A[idx_a]
            ax01.plot(beampattern_dB_EA[:,idx_a],self.elevation_deg_E)
            ax01.set_ylabel("Elevation (deg)")
            ax01.set_xlabel("dB")
            ax01.set_title(f"Vertical beam slice at Bearing= {np.round(look_bearing_deg,1)} deg")
            ax01.set_xlim(-60,0)
            ax01.set_ylim(self.elevation_deg_E.min(),self.elevation_deg_E.max())
            ax01.grid(True)
            ax01.set_yticks(np.arange(-90,90+30,30))

            return fig, ax00, ax10, ax01

    def plot_beampattern_look_sweep(self,weight_QN,azimuth_deg_A,elevation_deg_E,focus_range_ft=None,freq_display=None,
        look_dir_rad_13=None):
        """Plot fixed weights over a user-supplied arrival grid.

        Parameters
        ----------
        weight_QN : array-like, shape (Q, N) or (N,)
            Fixed multiplication weights. A one-dimensional vector is valid
            when the object has one configured frequency.
        azimuth_deg_A : array-like or scalar
            Nonempty bearing grid in degrees.
        elevation_deg_E : array-like or scalar
            Nonempty elevation grid in degrees.
        focus_range_ft : positive float, optional
            Arrival-grid range in feet. Defaults to this object's focus range.
        freq_display : float, optional
            Frequency to display in hertz. The nearest configured frequency is
            used; the first configured frequency is used when omitted.
        look_dir_rad_13 : array-like, shape (3,) or (1, 3), optional
            Direction used to choose horizontal and vertical slices on a 2-D
            plot. When omitted, the slices pass through the response peak.

        Returns
        -------
        tuple
            ``(figure, axes)`` for one elevation, or ``(figure, map_axes,
            horizontal_axes, vertical_axes)`` for a 2-D grid.

        Notes
        -----
        A look sweep holds ``weight_QN`` fixed and moves the signal-arrival
        direction over the requested bearing/elevation grid. Use it to answer
        "what is the spatial response of this one beam?" This is the natural
        plot for one externally calculated or adaptive weight vector, such as
        a DMR weight vector selected at a particular time, frequency, and
        steering direction.

        ``look_dir_rad_13`` only chooses the displayed horizontal and vertical
        cuts; it does not change the arrival grid being evaluated. A
        conventional fixed-weight look sweep can match the corresponding
        weight sweep when geometry, grid, range, shading, and normalization
        agree. Adaptive or direction-dependent apertures need not match.

        Magnitudes are displayed in decibels with a -60 to 0 dB plot range.
        The numerical response is also stored in ``beampattern_JQ``.
        """
        
        weight_QN = np.asarray(weight_QN)
        if weight_QN.ndim==1:
            weight_QN = weight_QN[np.newaxis,:]

        if look_dir_rad_13 is not None:
            look_dir_rad_13 = self._validate_single_look_direction(look_dir_rad_13)

        azimuth_deg_A = np.atleast_1d(np.asarray(azimuth_deg_A))
        elevation_deg_E = np.atleast_1d(np.asarray(elevation_deg_E))

        for angles in (azimuth_deg_A, elevation_deg_E):
            if (not np.issubdtype(angles.dtype, np.number) or np.iscomplexobj(angles) or not np.all(np.isfinite(angles))):
                raise ValueError("Bearing and elevation grids must contain finite real numbers")

        if (azimuth_deg_A.ndim != 1 or elevation_deg_E.ndim != 1 or azimuth_deg_A.size == 0 or elevation_deg_E.size == 0):
            raise ValueError("Bearing and elevation grids must be nonempty 1D arrays")

        azimuth_deg_A = azimuth_deg_A.astype(np.float64, copy=False)
        elevation_deg_E = elevation_deg_E.astype(np.float64, copy=False)

        if freq_display is None:
            idx_q = 0
        else:
            if np.asarray(freq_display).ndim != 0:
                raise ValueError("freq_display must be a scalar frequency in Hz")

            freq_display = self._validate_freq(freq_display)[0]
            idx_q = find_closest(self.freq, freq_display)

        if focus_range_ft is None:
            focus_range_ft = self.focus_range_ft

        focus_range_ft = self._validate_focus_range(focus_range_ft)

        # Look sweep: construct every requested source direction, then apply
        # the same fixed weight vector to each arrival replica.
        arrival_rbe_rad_J3 = construct_rbe_X3(focus_range_ft, azimuth_deg_A, elevation_deg_E)

        response_JQ = self.compute_beampattern_look_sweep_JQ(weight_QN, arrival_rbe_rad_J3)

        # Floor only the plotted magnitude to avoid log10(0).
        response_dB_J = 20.0 * np.log10(np.maximum(response_JQ[:, idx_q], 1e-12))

        A = azimuth_deg_A.size
        E = elevation_deg_E.size
        beampattern_dB_EA = response_dB_J.reshape(A, E).T

        bearing_label = "Relative Bearing (deg)" if self.flag_relative else "Geographic Bearing (deg)"

        if E==1:

            fig,ax = plt.subplots()
            ax.plot(azimuth_deg_A,beampattern_dB_EA[0, :])
            ax.set_xlabel(bearing_label)
            ax.set_ylabel("dB")
            ax.set_title(f"Beampattern at {self.freq[idx_q]:.1f} Hz")
            ax.set_xticks(self._bearing_plot_ticks(azimuth_deg_A))
            ax.set_xlim(*self._bearing_plot_limits(azimuth_deg_A))
            ax.set_ylim(-60,0)
            ax.grid(True)

            return fig, ax
        else:
            if look_dir_rad_13 is None:
                # With no reference look supplied, put both diagnostic cuts
                # through this fixed beam's strongest displayed response.
                response_EA = response_JQ[:, idx_q].reshape(A, E).T
                idx_e, idx_a = np.unravel_index(np.argmax(response_EA),response_EA.shape)
            else:
                # A supplied look selects plot cuts only; all directions above
                # were still evaluated with the same fixed weights.
                look = look_dir_rad_13

                idx_e = find_closest(elevation_deg_E, np.degrees(look[2]))
                idx_a = find_closest_bearing(azimuth_deg_A, np.degrees(look[1]))

            look_elevation_deg = elevation_deg_E[idx_e]
            look_bearing_deg = azimuth_deg_A[idx_a]

            fig = plt.figure(figsize=(10,8), constrained_layout=True)
            gs = fig.add_gridspec(2,3,width_ratios=[1, 0.05,1])

            ax00 = fig.add_subplot(gs[0,0])
            cax = fig.add_subplot(gs[0,1])
            ax01 = fig.add_subplot(gs[0,2])
            ax10 = fig.add_subplot(gs[1,0])

            pcm = ax00.pcolormesh(azimuth_deg_A,elevation_deg_E,beampattern_dB_EA,cmap="jet",shading="auto")

            fig.colorbar(pcm,cax=cax)
            
            ax00.set_xlabel(bearing_label)
            ax00.set_ylabel("Elevation (deg)")
            ax00.set_yticks(np.arange(-90,90+30,30))
            ax00.set_ylim(elevation_deg_E.min(),elevation_deg_E.max())
            ax00.set_xticks(self._bearing_plot_ticks(azimuth_deg_A))
            ax00.set_xlim(*self._bearing_plot_limits(azimuth_deg_A))
            pcm.set_clim(-60,0)
            ax00.set_title(f"2D Beampattern at {self.freq[idx_q]} Hz")

            ax10.plot(azimuth_deg_A, beampattern_dB_EA[idx_e, :])
            ax10.set_xlabel(bearing_label)
            ax10.set_ylabel("dB")
            ax10.set_title(f"Horizontal beam slice at Elevation={np.round(look_elevation_deg,1)} deg")
            ax10.set_ylim(-60,0)
            ax10.grid(True)
            ax10.set_xticks(self._bearing_plot_ticks(azimuth_deg_A))
            ax10.set_xlim(*self._bearing_plot_limits(azimuth_deg_A))

            ax01.plot(beampattern_dB_EA[:, idx_a], elevation_deg_E)
            ax01.set_ylabel("Elevation (deg)")
            ax01.set_xlabel("dB")
            ax01.set_title(f"Vertical beam slice at Bearing= {np.round(look_bearing_deg,1)} deg")
            ax01.set_xlim(-60,0)
            ax01.grid(True)
            ax01.set_yticks(np.arange(-90,90+30,30))
            ax01.set_ylim(elevation_deg_E.min(),elevation_deg_E.max())

            return fig, ax00, ax10, ax01
