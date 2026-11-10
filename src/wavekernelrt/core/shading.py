"""Spatial shading construction and curved-array aperture selection."""

import numpy as np
import wavekernelrt.core.conversion as conv
from scipy.signal.windows import hamming, taylor
from wavekernelrt.core.geometry_utils import segment_lengths
from wavekernelrt.core.config import Config

class Shading:
    """Construct normalized spatial shading for an array geometry.

    Parameters
    ----------
    geom : wavekernelrt.Geometry or wavekernelrt.core.geometry.BeamformingGeometry
        Complete or selected geometry. Curved geometries must provide element
        normals and regular-grid dimensions.

    Attributes
    ----------
    array_type : str
        Geometry model copied from ``geom``.
    N : int
        Number of retained array elements.
    Nrows, Ncols : int
        Regular-grid dimensions for panels, cylinders, and spheres. These
        attributes are not defined for line arrays.

    Notes
    -----
    Uniform, symmetric Hamming, and symmetric Taylor shading are supported.
    Every generated shading vector is real, nonnegative, and normalized so
    its weights sum to one. This is an amplitude normalization across array
    elements, not FFT-window power normalization.

    Line and panel arrays use one shared shading vector. Cylinders and spheres
    use the outward element normals to select the portion of the horizontal
    aperture facing each beam. Use the ``*_shading_for_beams`` methods when a
    curved array has more than one look direction.

    Panel and curved-array tapers are separable across columns and rows. Rows
    vary fastest in data-channel order, so an array of shape ``(Ncols,
    Nrows)`` is flattened to element order. These geometries are assumed to
    be complete, regularly ordered grids; unequal row or column spacing is not
    compensated. Unevenly spaced line arrays instead interpolate the taper by
    cumulative element position.

    Taylor shading defaults to a positive sidelobe attenuation of 30 dB and
    ``nbar=4``. Call
    [`set_taylor_params`][wavekernelrt.core.shading.Shading.set_taylor_params]
    before requesting Taylor weights to change those values.

    Examples
    --------
    Generate one shared taper for a line or panel array:

    >>> from wavekernelrt import Geometry, Shading
    >>> geom = Geometry("array_geometry.txt", array_type="line")
    >>> shade = Shading(geom).hamming_shading_N
    >>> float(shade.sum())
    1.0

    See Also
    --------
    wavekernelrt.ShadingConfig
        Pipeline configuration for selecting a shading recipe.
    wavekernelrt.BeamPattern
        Construct beam weights and evaluate their spatial response.
    """
    def __init__(self,geom):
        self.array_type = geom.array_type
        
        self.N = geom.xyz_m_N3.shape[0]
        if geom.array_type in ("panel_starboard","panel_port"):
            self.Nrows = geom.Nrows
            self.Ncols = geom.Ncols
        elif geom.array_type in ("sphere","cylinder"):
            # Keep track of all the normals to do per-beam shading as we move around
            # a sphere or cylindar
            self._normals_N3 = np.array(geom.normals_N3, dtype=np.float64, copy=True)

            if self._normals_N3.shape != (self.N, 3):
                raise ValueError("Element normals must have shape (N, 3)")

            normal_lengths = np.linalg.norm(self._normals_N3, axis=1, keepdims=True)

            if (self._normals_N3.shape != (self.N, 3) or not np.all(np.isfinite(self._normals_N3)) or np.any(normal_lengths <= 0)):

                raise ValueError("Element normals must be finite, nonzero 3D vectors")

            self._normals_N3 /= normal_lengths
            self._array_heading_deg = float(geom.array_heading_deg)

            self.Nrows = geom.Nrows
            self.Ncols = geom.Ncols

            # Derive one horizontal direction per azimuth column.
            normals_CR3 = self._normals_N3.reshape(self.Ncols, self.Nrows, 3)
            horizontal_C2 = np.mean(normals_CR3[:, :, :2], axis=1)

            lengths_C1 = np.linalg.norm(horizontal_C2, axis=1, keepdims=True)

            if np.any(lengths_C1 <= 1e-12):
                raise ValueError(
                    "Cannot determine a column's horizontal direction "
                    "from its selected element normals")

            self._normals_c3 = np.zeros((self.Ncols, 3), dtype=np.float64)
            self._normals_c3[:, :2] = horizontal_C2 / lengths_C1
            
            # assume a unit vector coming in from directly ahead [0 1 0]
            # then all we need is the second column to compute the cosine angle between the elements
            self._column_logical_c3 = self._normals_c3[:,1]>Config.CURVATURE_SHUTOFF

            # keep track of the look direction for shading later, particularly if array is cropped
            self._look_azimuth_rad = 0.0

        else:
            # get the element spacing to test if linear or not, if not linear we have to interpolate the shading later
            dr = segment_lengths(geom.xyz_m_N3)
           
            # test line array geometry if evenly spaced, if yes, then don't need to interpoloate
            if np.max(dr)-np.min(dr) < np.median(dr) * 0.001:
                self._line_linear = True
            else:
                self._line_linear = False
                self._dr = dr
            
        self._uniform_shading_N = None
        self._hamming_shading_N = None
        self._taylor_shading_N = None
        self._taylor_sll_dB = 30
        self._taylor_nbar = 4
       
    @property
    def uniform_shading_N(self):
        """numpy.ndarray: Lazy, sum-normalized uniform element weights."""
        if self._uniform_shading_N is None:
            self._uniform_shading_N = self._compute_uniform_shading()
        return self._uniform_shading_N

    @property
    def hamming_shading_N(self):
        """numpy.ndarray: Lazy, sum-normalized symmetric Hamming weights."""
        if self._hamming_shading_N is None:
            self._hamming_shading_N = self._compute_hamming_shading()
        return self._hamming_shading_N
    
    @property
    def taylor_shading_N(self):
        """numpy.ndarray: Lazy, sum-normalized symmetric Taylor weights."""
        if self._taylor_shading_N is None:
            self._taylor_shading_N = self._compute_taylor_shading()
        return self._taylor_shading_N

    def _compute_uniform_shading(self):
        if self.array_type in ("sphere","cylinder"):
            # add in ones to a field of zeros
            active = self._active_columns()

            shade_CR = np.zeros((self.Ncols, self.Nrows), dtype=np.float32)
            shade_CR[active, :] = 1.0
            shade_CR /= np.sum(shade_CR)

            self._uniform_shading_N = shade_CR.reshape(-1, 1)

        elif self.array_type in ("panel_starboard","panel_port","line"):
            # all elements get ones
            shade_N = np.ones((self.N,1), dtype=np.float32)
            shade_N /= np.sum(shade_N)
            self._uniform_shading_N = shade_N
        
        return self._uniform_shading_N

    def _compute_hamming_shading(self):
        if self.array_type in ("sphere", "cylinder"):
            active = self._active_columns()
            active_indices = np.flatnonzero(active)

            column_azimuth_rad = np.arctan2(self._normals_c3[:, 0],self._normals_c3[:, 1])
            offset_rad = (column_azimuth_rad - self._look_azimuth_rad + np.pi) % (2 * np.pi) - np.pi
            ordered_indices = active_indices[np.argsort(offset_rad[active_indices])]

            shade_C = hamming(M=ordered_indices.size, sym=True)
            shade_R = hamming(M=self.Nrows, sym=True)
            shade_CR = np.zeros((self.Ncols, self.Nrows), dtype=np.float64)
            shade_CR[ordered_indices, :] = shade_C[:, None] * shade_R[None, :]
            shade_N = shade_CR.reshape(-1, 1)

        elif self.array_type in ("panel_starboard", "panel_port"):
            shade_C = hamming(M=self.Ncols, sym=True)
            shade_R = hamming(M=self.Nrows, sym=True)
            shade_N = (shade_C[:, None] * shade_R[None, :]).reshape(-1, 1)

        elif self.array_type == "line":
            if self._line_linear:
                shade_N = hamming(M=self.N, sym=True)
            else:
                shade_M = hamming(M=301, sym=True)
                element_position = np.concatenate(([0], np.cumsum(self._dr)))
                shade_N = np.interp(element_position / element_position[-1],np.linspace(0, 1, shade_M.size),shade_M)

        shade_N /= np.sum(shade_N)
        self._hamming_shading_N = shade_N.astype(np.float32)
        return self._hamming_shading_N
    
    def set_taylor_params(self,sll_dB=30,nbar=4):
        """Set the Taylor-window parameters and invalidate cached weights.

        Parameters
        ----------
        sll_dB : float, default=30
            Desired sidelobe attenuation in decibels. The absolute value is
            stored, so either ``30`` or ``-30`` selects 30 dB attenuation.
        nbar : int, default=4
            Number of nearly constant-level sidelobes passed to SciPy's
            Taylor-window calculation.

        Notes
        -----
        This method does not calculate weights immediately. The next access
        to
        [`taylor_shading_N`][wavekernelrt.core.shading.Shading.taylor_shading_N]
        or call to
        [`taylor_shading_for_beams`][wavekernelrt.core.shading.Shading.taylor_shading_for_beams]
        uses the new values.
        """
        self._taylor_sll_dB = np.abs(sll_dB)
        self._taylor_nbar = nbar
        self._taylor_shading_N = None

    def _compute_taylor_shading(self):

        if self.array_type in ("sphere","cylinder"):
            active = self._active_columns()
            active_indices = np.flatnonzero(active)

            # Bearing of each column's outward normal:
            # atan2(east, north), consistent with clockwise-from-north.
            column_azimuth_rad = np.arctan2(self._normals_c3[:, 0],self._normals_c3[:, 1])

            # Express bearings relative to the look direction in [-pi, pi).
            offset_rad = (column_azimuth_rad - self._look_azimuth_rad + np.pi) % (2 * np.pi) - np.pi

            # Order visible columns across the aperture.
            ordered_indices = active_indices[np.argsort(offset_rad[active_indices])]

            shade_C = taylor(M=ordered_indices.size,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)
            shade_R = taylor(M=self.Nrows,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)

            shade_CR = np.zeros((self.Ncols, self.Nrows), dtype=np.float32)

            # Write weights back into their original data-column positions.
            shade_CR[ordered_indices, :] = (shade_C[:, None] * shade_R[None, :])

            shade_N = shade_CR.reshape(-1, 1)

        elif self.array_type in ("panel_starboard","panel_port"):
            shade_C = taylor(M=self.Ncols,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)
            shade_R = taylor(M=self.Nrows,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)
            
            shade_CR = shade_C[:,None] * shade_R[None,:]
            shade_N = shade_CR.reshape(-1,1)
        elif self.array_type=="line":
            if self._line_linear:
                shade_N = taylor(M=self.N,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)
            else:
                min_dr = np.min(self._dr)
                M = len(np.arange(0,np.sum(self._dr),min_dr))
                shade_M = taylor(M=301,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)

                shade_N = np.interp(np.concatenate(([0], np.cumsum(self._dr))) / np.sum(self._dr),np.linspace(0,1,301),shade_M)
                

        shade_N /= np.sum(shade_N)
        self._taylor_shading_N = shade_N.astype(np.float32)        
        return self._taylor_shading_N
    
    def update_look_dir(self,look_dir_rbe_rad):
        """Update the aperture direction used by curved-array properties.

        Parameters
        ----------
        look_dir_rbe_rad : array-like, shape (3,)
            Look direction as ``[range, bearing, elevation]``. Angles are in
            radians; only bearing affects the horizontal aperture selection.

        Notes
        -----
        For cylinders and spheres, changing the look direction invalidates
        all cached shared shading properties. For line and panel geometries,
        this method has no effect. For multiple curved-array beams, prefer the
        ``*_shading_for_beams`` methods rather than repeatedly updating this
        state.
        """
        # only care about the xy plane here
        if self.array_type in ("sphere","cylinder"):

            # udpate the look direction
            self._look_azimuth_rad = float(look_dir_rbe_rad[1])

            look_vec_13 = np.array([np.sin(look_dir_rbe_rad[1]), np.cos(look_dir_rbe_rad[1]), 0])
            cos_c1 = self._normals_c3 @ look_vec_13[:,None]
            self._column_logical_c3 = cos_c1>Config.CURVATURE_SHUTOFF

            # clear the weights to recalculate the next time called
            self._uniform_shading_N = None
            self._hamming_shading_N = None
            self._taylor_shading_N = None

    def plot_shading(self,shade_type,geom=None,look_dir_rbe_rad=None):
        """Plot element shading by index, position, or three-dimensional location.

        Parameters
        ----------
        shade_type : {"uniform", "hamming", "taylor"}
            Shading recipe to plot.
        geom : wavekernelrt.Geometry or BeamformingGeometry, optional
            Geometry used for plot coordinates. For a line array, omitting it
            plots element number; supplying it plots aperture position in
            feet. It is required for panel, cylinder, and sphere plots.
        look_dir_rbe_rad : array-like, shape (3,), optional
            Curved-array look direction as ``[range, bearing, elevation]``.
            If supplied, the visible aperture is updated before plotting.

        Returns
        -------
        None
            The Matplotlib figure is created as a plotting side effect.

        Raises
        ------
        ValueError
            If ``shade_type`` is unsupported or no curved-array columns face
            the requested look direction.
        """
        import matplotlib.pyplot as plt

        if self.array_type in ("sphere","cylinder") and look_dir_rbe_rad is not None:
            self.update_look_dir(look_dir_rbe_rad)

        if shade_type=="uniform":
            shade_N = self.uniform_shading_N
        elif shade_type=="hamming":
            shade_N = self.hamming_shading_N
        elif shade_type=="taylor":
            shade_N = self.taylor_shading_N
        else:
            raise ValueError("shade_type must be 'uniform', 'hamming', or 'taylor'")
        
        fig = plt.figure(constrained_layout=True)
        if self.array_type=="line":
            ax = fig.add_subplot(111)
            if geom is not None:
                dr_ft = segment_lengths(geom.xyz_m_N3)* conv.M_TO_FT
                aperture_ft = np.concatenate(([0],np.cumsum(dr_ft)))
                aperture_ft -= np.mean(aperture_ft)
                ax.plot(aperture_ft,shade_N,'-o',linewidth=2)
                ax.set_xlabel('Position (ft)')
                ax.set_xlim((np.min(aperture_ft),np.max(aperture_ft)))
            else:
                ax.plot(range(1,len(shade_N)+1),shade_N,linewidth=2)
                ax.set_xlabel('Element #')
                ax.set_xlim((1,len(shade_N)))

            ax.set_ylabel('Shading Amplitudes')
            
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(plt.NullLocator()) # turn off y minor ticks
            ax.grid(True,which='both')
        else:
            ax = fig.add_subplot(111, projection='3d')
            xyz_ft_N3 = geom.xyz_m_N3 * conv.M_TO_FT
            sc = ax.scatter(xyz_ft_N3[:,0],xyz_ft_N3[:,1],xyz_ft_N3[:,2],c=shade_N,cmap='jet',s=40)
            plt.colorbar(sc, ax=ax, label='Shading Value')
            ax.set_aspect('equal')
            ax.set_xlabel('x (ft)')
            ax.set_ylabel('y (ft)')
            ax.set_zlabel('z (ft)')

    def _active_columns(self):
        """Return the columns facing the current shading look direction.
        
        This is used in case channels are cropped and now no part of the array
        faces the look direction. This is part of that error checking.
        """
        active = np.asarray(self._column_logical_c3, dtype=bool).reshape(-1)

        if not np.any(active):
            raise ValueError(
                "No selected array columns face the shading look direction; "
                "check the selected segment and look direction")

        return active

    def visible_elements(self, look_dir_rbe_rad, *, flag_relative=True,require_visible_elements=True):
        """Identify the elements facing one look direction.

        Parameters
        ----------
        look_dir_rbe_rad : array-like, shape (3,)
            Look direction as ``[range, bearing, elevation]``. Angles are in
            radians. Range and elevation do not affect visibility.
        flag_relative : bool, default=True
            If ``True``, interpret bearing relative to array heading. If
            ``False``, interpret it in the absolute East-North-Up frame used
            by the element normals.
        require_visible_elements : bool, default=True
            If ``True``, raise an error when no curved-array column faces the
            look direction. If ``False``, return an all-false mask instead.

        Returns
        -------
        numpy.ndarray, shape (N,), dtype bool
            Visibility mask in data-channel order. All elements are visible
            for line and panel arrays. For cylinders and spheres, every row
            in a visible horizontal column is included.

        Raises
        ------
        ValueError
            If the direction is not a finite three-component direction, or
            if visibility is required and no curved-array column is active.

        Notes
        -----
        Curved-array columns are active when their horizontal normal has a
        dot product greater than ``Config.CURVATURE_SHUTOFF`` with the
        horizontal look vector.
        """
        look = np.asarray(look_dir_rbe_rad, dtype=np.float64)

        if look.shape != (3,) or not np.all(np.isfinite(look[1:])):
            raise ValueError("Expected [range, bearing, elevation] with finite angles")

        if self.array_type not in ("sphere", "cylinder"):
            return np.ones(self.N, dtype=bool)

        bearing = look[1]

        # Horizontal aperture only; elevation does not select rows.
        look_vector = np.array([np.sin(bearing),np.cos(bearing),0.0])

        normals_C3 = self._normals_c3

        if flag_relative and abs(self._array_heading_deg) > 1:
            theta = -np.deg2rad(self._array_heading_deg)
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta),  np.cos(theta), 0],[0,              0,             1]])
            normals_C3 = normals_C3 @ rotation

        visible_C = (normals_C3 @ look_vector) > Config.CURVATURE_SHUTOFF

        if require_visible_elements and not np.any(visible_C):
            raise ValueError("No selected columns face this beam's bearing")

        # Rows vary fastest within each column.
        return np.repeat(visible_C, self.Nrows)

    def uniform_shading_for_beams(self, steer_rbe_rad_J3, *, flag_relative=True):
        """Construct sum-normalized uniform shading for one or more beams.

        Parameters
        ----------
        steer_rbe_rad_J3 : array-like, shape (J, 3)
            Nonempty steering directions as ``[range, bearing, elevation]``.
            Angles are in radians.
        flag_relative : bool, default=True
            If ``True``, interpret bearings relative to array heading. If
            ``False``, use the absolute element-normal frame.

        Returns
        -------
        numpy.ndarray, shape (J, N), dtype float32
            One shading vector per beam. Visible elements receive equal
            weights summing to one; invisible elements receive zero.

        Raises
        ------
        ValueError
            If the steering array does not have nonempty shape ``(J, 3)`` or
            a beam has no visible curved-array elements. Beam-specific errors
            identify the zero-based beam index.
        """
        beams = np.asarray(steer_rbe_rad_J3, dtype=np.float64)

        if (beams.ndim != 2 or beams.shape[0] == 0 or beams.shape[1] != 3):
            raise ValueError("Steering directions must have nonempty shape (J, 3)")

        shade_JN = np.zeros((beams.shape[0], self.N), dtype=np.float32)

        for jj, look in enumerate(beams):
            try:
                active_N = self.visible_elements(look, flag_relative=flag_relative)
            except ValueError as exc:
                raise ValueError(f"Beam {jj}: {exc}") from exc

            nactive = np.count_nonzero(active_N)
            shade_JN[jj, active_N] = 1.0 / nactive

        return shade_JN

    def hamming_shading_for_beams(self, steer_rbe_rad_J3, *, flag_relative=True):
        """Construct sum-normalized Hamming shading for one or more beams.

        Parameters
        ----------
        steer_rbe_rad_J3 : array-like, shape (J, 3)
            Nonempty steering directions as ``[range, bearing, elevation]``.
            Angles are in radians.
        flag_relative : bool, default=True
            If ``True``, interpret bearings relative to array heading. If
            ``False``, use the absolute element-normal frame.

        Returns
        -------
        numpy.ndarray, shape (J, N), dtype float32
            One sum-normalized shading vector per beam. Line and panel arrays
            repeat their shared taper. Curved arrays apply a separable
            symmetric Hamming taper to each beam's visible columns and rows;
            invisible elements receive zero.

        Raises
        ------
        ValueError
            If the steering array does not have nonempty shape ``(J, 3)``,
            contains a nonfinite bearing or elevation, or a curved-array beam
            has no valid visible-aperture weights.
        """
        beams = np.asarray(steer_rbe_rad_J3, dtype=np.float64)

        if (beams.ndim != 2 or beams.shape[0] == 0 or beams.shape[1] != 3):
            raise ValueError("Steering directions must have nonempty shape (J, 3)")

        if not np.all(np.isfinite(beams[:, 1:])):
            raise ValueError("Beam bearings and elevations must be finite")

        if self.array_type not in ("sphere", "cylinder"):
            shade_N = np.asarray(self.hamming_shading_N).reshape(-1)
            return np.broadcast_to(shade_N, (beams.shape[0], self.N)).copy()

        normals = self._normals_N3

        if flag_relative and abs(self._array_heading_deg) > 1:
            theta = -np.deg2rad(self._array_heading_deg)
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta),  np.cos(theta), 0],[0,              0,             1]])
            normals = normals @ rotation

        normals_CR3 = normals.reshape(self.Ncols, self.Nrows, 3)
        column_normals_C3 = np.mean(normals_CR3, axis=1)
        column_azimuth_C = np.arctan2(column_normals_C3[:, 0],column_normals_C3[:, 1])

        shade_JN = np.zeros((beams.shape[0], self.N), dtype=np.float32)

        for jj, look in enumerate(beams):
            try:
                active_N = self.visible_elements(look,flag_relative=flag_relative)
            except ValueError as exc:
                raise ValueError(f"Beam {jj}: {exc}") from exc

            active_CR = active_N.reshape(self.Ncols, self.Nrows)
            active_columns = np.flatnonzero(np.any(active_CR, axis=1))
            active_rows = np.flatnonzero(np.any(active_CR, axis=0))

            offset_C = (column_azimuth_C - look[1] + np.pi) % (2 * np.pi) - np.pi
            ordered_columns = active_columns[np.argsort(offset_C[active_columns])]

            taper_C = hamming(M=ordered_columns.size, sym=True)
            taper_R = hamming(M=active_rows.size, sym=True)
            weights_CR = np.zeros((self.Ncols, self.Nrows), dtype=np.float64)
            weights_CR[np.ix_(ordered_columns, active_rows)] = (taper_C[:, None] * taper_R[None, :])
            weights_CR[~active_CR] = 0.0

            total = np.sum(weights_CR)
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"Beam {jj}: Hamming weights cannot be normalized")

            shade_JN[jj, :] = (weights_CR / total).ravel()

        return shade_JN

    def taylor_shading_for_beams(self, steer_rbe_rad_J3, *, flag_relative=True):
        """Construct sum-normalized Taylor shading for one or more beams.

        Parameters
        ----------
        steer_rbe_rad_J3 : array-like, shape (J, 3)
            Nonempty steering directions as ``[range, bearing, elevation]``.
            Angles are in radians.
        flag_relative : bool, default=True
            If ``True``, interpret bearings relative to array heading. If
            ``False``, use the absolute element-normal frame.

        Returns
        -------
        numpy.ndarray, shape (J, N), dtype float32
            One sum-normalized shading vector per beam. Line and panel arrays
            repeat their shared taper. Curved arrays apply separable Taylor
            tapers to each visible row/column aperture and set invisible
            elements to zero.

        Raises
        ------
        ValueError
            If the steering array does not have nonempty shape ``(J, 3)``,
            contains a nonfinite bearing or elevation, or a curved-array beam
            has no valid visible-aperture weights.

        Notes
        -----
        The taper uses the most recent parameters supplied to
        [`set_taylor_params`][wavekernelrt.core.shading.Shading.set_taylor_params].
        Unequal curved-array row or column spacing is not compensated.
        """

        beams = np.asarray(steer_rbe_rad_J3, dtype=np.float64)

        if (beams.ndim != 2 or beams.shape[0] == 0 or beams.shape[1] != 3):
            raise ValueError("Steering directions must have nonempty shape (J, 3)")

        if not np.all(np.isfinite(beams[:, 1:])):
            raise ValueError("Beam bearings and elevations must be finite")

        if self.array_type not in ("sphere", "cylinder"):
            shade_N = np.asarray(self.taylor_shading_N).reshape(-1)
            return np.broadcast_to(shade_N, (beams.shape[0], self.N)).copy()

        # Express normals in the same frame as the steering directions.
        normals = self._normals_N3

        if flag_relative and abs(self._array_heading_deg) > 1:
            theta = -np.deg2rad(self._array_heading_deg)
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta),  np.cos(theta), 0],[0,              0,             1]])
            normals = normals @ rotation

        # Average across rows to obtain each column's azimuth.
        normals_CR3 = normals.reshape(self.Ncols, self.Nrows, 3)
        column_normals_C3 = np.mean(normals_CR3, axis=1)
        column_azimuth_C = np.arctan2(column_normals_C3[:, 0],column_normals_C3[:, 1])

        shade_JN = np.zeros((beams.shape[0], self.N), dtype=np.float32)

        for jj, look in enumerate(beams):
            try:
                active_N = self.visible_elements(look,flag_relative=flag_relative)
            except ValueError as exc:
                raise ValueError(f"Beam {jj}: {exc}") from exc

            active_CR = active_N.reshape(self.Ncols, self.Nrows)

            active_columns = np.flatnonzero(np.any(active_CR, axis=1))
            active_rows = np.flatnonzero(np.any(active_CR, axis=0))

            # Order columns across the beam, including across 0/360.
            offset_C = (column_azimuth_C - look[1] + np.pi) % (2 * np.pi) - np.pi

            ordered_columns = active_columns[np.argsort(offset_C[active_columns])]

            taper_C = taylor(M=ordered_columns.size,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)
            taper_R = taylor(M=active_rows.size,nbar=self._taylor_nbar,sll=self._taylor_sll_dB,norm=False,sym=True)

            weights_CR = np.zeros((self.Ncols, self.Nrows), dtype=np.float64)
            weights_CR[np.ix_(ordered_columns, active_rows)] = (taper_C[:, None] * taper_R[None, :])

            weights_CR[~active_CR] = 0.0

            total = np.sum(weights_CR)
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"Beam {jj}: Taylor weights cannot be normalized")

            shade_JN[jj, :] = (weights_CR / total).ravel()

        return shade_JN
