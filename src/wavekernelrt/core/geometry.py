"""Array-geometry loading, derived properties, plotting, and selection."""

import numpy as np
import os
from wavekernelrt.core.geometry_utils import segment_lengths
import wavekernelrt.core.conversion as conv

from dataclasses import dataclass
from copy import deepcopy

@dataclass
class BeamformingGeometry:
    """Geometry subset used by the beamforming stages.

    Instances are returned by
    [`Geometry.select_elements`][wavekernelrt.core.geometry.Geometry.select_elements];
    users normally do not construct them directly.

    Attributes
    ----------
    array_type : str
        Parent geometry type.
    xyz_m_N3 : numpy.ndarray, shape (N, 3)
        Selected element positions in the centered coordinate system of the
        complete parent array, in meters.
    phase_center_m_13 : numpy.ndarray, shape (1, 3)
        Arithmetic mean of the selected positions, in meters.
    normals_N3 : numpy.ndarray or None, shape (N, 3)
        Selected element-normal vectors, or ``None`` when the parent geometry
        has no normals.
    soundspeed_mpers : numpy.ndarray
        Sound speed in meters per second.
    array_heading_deg : float
        Parent array heading in degrees clockwise from north.
    far_field_range_m : float
        Parent array's far-field distance estimate, in meters.
    max_aperture_m : float
        Aperture used for the parent far-field estimate, in meters.
    Nrows, Ncols : int or None
        Selected regular-grid dimensions. Both are ``None`` for line arrays.

    Notes
    -----
    Positions are not recentered after selection. Retaining the parent
    coordinate system lets separately selected subarrays use a common spatial
    reference, while ``phase_center_m_13`` records the selected subset's phase
    center.
    """
    array_type: str
    xyz_m_N3: np.ndarray
    phase_center_m_13: np.ndarray
    normals_N3: np.ndarray | None
    soundspeed_mpers: np.ndarray
    array_heading_deg: float
    far_field_range_m: float
    max_aperture_m: float
    Nrows: int | None = None
    Ncols: int | None = None

class Geometry:
    """Load an acoustic-array geometry and calculate derived properties.

    Parameters
    ----------
    file : str or path-like
        Text file containing one element per row. The first three columns are
        ``x``, ``y``, and ``z`` positions in inches. Optional columns four
        through six are dimensionless element-normal components ``nx``,
        ``ny``, and ``nz``.
    array_type : {"line", "panel_starboard", "panel_port", "sphere", "cylinder"}
        Geometry model used for spacing, aperture, normal, and shading
        calculations. The two panel values select opposite broadside-normal
        directions when normals are not supplied.
    soundspeed_ftpers : float, default=5000
        Sound speed in feet per second. It is converted to meters per second
        for internal calculations.

    Attributes
    ----------
    file : str or path-like
        Geometry-file location supplied to the constructor.
    array_type : str
        Selected geometry model.
    xyz_m_N3 : numpy.ndarray, shape (N, 3)
        Element positions in meters after subtracting the array-wide mean.
        Coordinates use East-North-Up axes: ``x`` east, ``y`` north, and ``z``
        up.
    phase_center_m_13 : numpy.ndarray, shape (1, 3)
        Arithmetic mean of the centered element positions, in meters. This is
        approximately zero, subject to ``float32`` rounding.
    soundspeed_mpers : numpy.ndarray
        Sound speed in meters per second, stored with ``float32`` precision.
    Nrows, Ncols : int
        Regular-grid dimensions for panels, spheres, and cylinders. These
        attributes are not defined for line arrays.

    Raises
    ------
    ValueError
        If ``file`` does not exist, ``array_type`` is unsupported, or the
        loaded file contains fewer than three columns.
    OSError
        If NumPy cannot read the geometry file.

    Notes
    -----
    Bearings are measured clockwise from north. The implemented array heading
    points toward the first element in data-channel order.

    Panels, spheres, and cylinders must be complete regular grids ordered with
    vertical rows varying fastest within each horizontal column. Their spacing,
    aperture, normals, and shading calculations rely on this organization.

    Derived quantities are calculated lazily and cached when their properties
    are first accessed. Property availability depends on ``array_type``:

    - Line: ``dr_m``, ``design_freq``, and ``aperture_m``.
    - Panel: ``dcol_m``, ``drow_m``, ``design_freq_azimuth``,
      ``design_freq_elevation``, ``width_aperture_m``, and
      ``height_aperture_m``.
    - Sphere or cylinder: ``dring_m``, ``drow_m``,
      ``design_freq_azimuth``, ``design_freq_elevation``,
      ``width_aperture_m``, and ``height_aperture_m``.

    See Also
    --------
    wavekernelrt.Shading
        Construct spatial shading for a geometry.
    wavekernelrt.SteeringVectors
        Construct steering delays and replicas.
    wavekernelrt.BeamPattern
        Evaluate beam weights and spatial response.

    Examples
    --------
    >>> from wavekernelrt import Geometry
    >>> geom = Geometry(
    ...     "array_geometry.txt",
    ...     array_type="line",
    ...     soundspeed_ftpers=5000.0,
    ... )
    >>> geom.xyz_m_N3.shape[1]
    3
    """

    def __init__(self,file,array_type,soundspeed_ftpers=5000):
        # class initialization, runs when an instance is called for the first time

        # Do some lite error checking
        if not os.path.exists(file):
            raise ValueError(f"File does not exist: {file}")

        if array_type not in ("line","panel_starboard","panel_port","sphere","cylinder"):
            raise ValueError(f"Array type is: {array_type}\nIt must be one of: line, panel_starboard, panel_port, sphere, cylinder")
        
        self.file = file

        # load in the geometry file
        xyz_in_N3 = np.loadtxt(file, dtype=np.float32, ndmin=2)
        if xyz_in_N3.shape[1]>3:
            # seperate out the first 3 columns for xyz and the last 3 columns for element normals
            self.xyz_m_N3 = xyz_in_N3[:,0:3] * conv.IN_TO_M # all internal calculations done in meters
            self._normals_N3 = xyz_in_N3[:,3:6]
        elif xyz_in_N3.shape[1]<3:
            raise ValueError("Geometry file must have at least 3 columns")
        else:
            self.xyz_m_N3 = xyz_in_N3 * conv.IN_TO_M # all internal calculations done in meters
            self._normals_N3 = None # default to None to calculate with lazy property when needed

        # store minimal amount of things
        self.array_type = array_type
        self.xyz_m_N3 -= np.mean(self.xyz_m_N3, axis=0, keepdims=True, dtype=np.float32) # center the array around 0
        # define the phase center of the array as the physical center
        self.phase_center_m_13 = np.atleast_2d(np.mean(self.xyz_m_N3,axis=0,dtype=np.float32))
        self.soundspeed_mpers = np.array(soundspeed_ftpers, dtype=np.float32)/conv.M_TO_FT

        if array_type in ("panel_starboard","panel_port","sphere","cylinder"):
            # For 2D/3D arrays define how many rows and columns there are
            # Baked into these values are an assumption about symmetry
            # The downstream processing doesn't need symmetry to correctly apply its algorithms, but these values
            # would be wrong and care would have to be taken to calculate the proper shading
            self.Nrows = len(np.unique(self.xyz_m_N3[:,2]))
            self.Ncols = int(np.round(self.xyz_m_N3.shape[0]/self.Nrows))

            # Default the lazy properities
            self._drow_m = None
            if array_type in ("panel_starboard","panel_port"):
                self._dcol_m = None
            else:
                self._dring_m = None

            self._design_freq_azimuth = None
            self._design_freq_elevation = None
            self._height_aperture_m = None
            self._width_aperture_m = None
        else:
            # Default the lazy properities
            self._dr_m = None
            self._design_freq = None
            self._aperture_m = None

        # initialize values for properties later if called
        self._array_heading_deg = None   
        self._far_field_range_m = None

    def __str__(self):
        """Return a human-readable geometry summary.

        Lazily derived values are included only if they have already been
        calculated.
        """
        # print out relevant information to the command window or file to be able to spot check things
        N=20 # how many spaces to use so the text lines up when it is printed out
        text = [f"Geometry"]
        text.append(f"  {'file':{N}}: {self.file}")
        text.append(f"  {'array_type':{N}}: {self.array_type}")
   
        text.append(f"  {'xyz_m_N3':{N}}: [{self.xyz_m_N3.shape[0]}x{self.xyz_m_N3.shape[1]}]")
        text.append(f"  {'phase_center_m_13':{N}}: [{self.phase_center_m_13.shape[0]}x{self.phase_center_m_13.shape[1]}]")

        if self.array_type in ("panel_starboard","panel_port","sphere","cylinder"):
            text.append(f"  {'Nrows':{N}}: {self.Nrows}")
            text.append(f"  {'Ncols':{N}}: {self.Ncols}")
            if self.array_type in ("sphere","cylinder") and self._dring_m is not None:
                text.append(f"  {'dring_m':{N}}: {self._dring_m:.1f} inches")
            
            if self.array_type in ("panel_starboard","panel_port") and self._dcol_m is not None:
                text.append(f"  {'dcol_m':{N}}: {self._dcol_m:.1f} inches")

            if self._drow_m is not None:
                text.append(f"  {'drow_m':{N}}: {self._drow_m:.1f} inches")

            if self._design_freq_azimuth is not None:
                text.append(f"  {'design_freq_azimuth':{N}}: {self._design_freq_azimuth:.1f} Hz")
            
            if self._design_freq_elevation is not None:
                text.append(f"  {'design_freq_elevation':{N}}: {self._design_freq_elevation:.1f} Hz")
        else:
            if self._design_freq is not None:
                text.append(f"  {'design_freq':{N}}: {self._design_freq:.1f} Hz")

        if self._array_heading_deg is not None:
            text.append(f"  {'array_heading_deg':{N}}: {self._array_heading_deg:.1f}")
          
        text.append(f"  {'soundspeed_mpers':{N}}: {self.soundspeed_mpers:.1f} m/s")

        return "\n".join(text)
    
    # Lazy properties that will compute the first time called and cache the result for future calls
    @property
    def array_heading_deg(self):
        """float: Heading toward the first element, clockwise from north."""
        if self._array_heading_deg is None:
            self._array_heading_deg = self._compute_array_heading()
        return self._array_heading_deg
    
    @property
    def dr_m(self):
        """float: Median adjacent-element spacing for a line array, in meters."""
        if self._dr_m is None:
            self._compute_array_spacing()
        return self._dr_m
    
    @property
    def drow_m(self):
        """float: Median vertical row spacing, in meters."""
        if self._drow_m is None and self.array_type in ("panel_starboard","panel_port","sphere","cylinder"):
            self._compute_array_spacing()
        return self._drow_m
    
    @property
    def dcol_m(self):
        """float: Median horizontal column spacing for a panel, in meters."""
        if self._dcol_m is None and self.array_type in ("panel_starboard","panel_port"):
            self._compute_array_spacing()
        return self._dcol_m
    
    @property
    def dring_m(self):
        """float: Median middle-ring spacing for a sphere or cylinder, in meters."""
        if self._dring_m is None and self.array_type in ("sphere","cylinder"):
            self._compute_array_spacing()
        return self._dring_m

    @property
    def design_freq(self):
        """float: Line-array half-wavelength design frequency, in hertz."""
        if self._design_freq is None:
            self._compute_design_freq()
        return self._design_freq
    
    @property
    def design_freq_azimuth(self):
        """float: Horizontal half-wavelength design frequency, in hertz."""
        if self._design_freq_azimuth is None:
            self._compute_design_freq()
        return self._design_freq_azimuth
    
    @property
    def design_freq_elevation(self):
        """float: Vertical half-wavelength design frequency, in hertz."""
        if self._design_freq_elevation is None:
            self._compute_design_freq()
        return self._design_freq_elevation
    
    @property
    def normals_N3(self):
        """numpy.ndarray or None: Element normals with shape ``(N, 3)``."""
        if self._normals_N3 is None:
            self._compute_xyz_normals()
        return self._normals_N3

    @property
    def aperture_m(self):
        """float: First-to-last line-array aperture, in meters."""
        if self._aperture_m is None:
            self._compute_aperture()
        return self._aperture_m
    
    @property
    def height_aperture_m(self):
        """float: Vertical aperture of a multidimensional array, in meters."""
        if self._height_aperture_m is None:
            self._compute_aperture()
        return self._height_aperture_m
    
    @property
    def width_aperture_m(self):
        """float: Horizontal aperture of a multidimensional array, in meters."""
        if self._width_aperture_m is None:
            self._compute_aperture()
        return self._width_aperture_m
    
    @property
    def far_field_range_m(self):
        """float: Fraunhofer-style far-field distance estimate, in meters.

        The implementation uses ``2 * aperture**2 / wavelength``. The
        wavelength is evaluated at twice the relevant spacing-derived design
        frequency, and multidimensional arrays use their diagonal aperture and
        larger design frequency.
        """
        if self._far_field_range_m is None:
            self._compute_far_field_range()
        return self._far_field_range_m
    
    
    # private functions
    # compute the heading of the array
    def _compute_array_heading(self):
        if self.array_type in ("panel_starboard","panel_port"):
            # By convetion, z is the fast dimension, we want one horiztonal stack at all the same z
            # so start at the first element and step every Nrows to get the next element in
            # at the same z
            xyz_n3 = self.xyz_m_N3[::self.Nrows,:]
        elif self.array_type in ("sphere","cylinder"):
            avg_heading_deg = np.array(0, dtype=np.float32)
            return avg_heading_deg
        else:
            xyz_n3 = self.xyz_m_N3
        
        # take the diff along the elements
        array_diff = np.diff(xyz_n3,n=1,axis=0)

        # compute the heading to the last element (way the data is ordered)
        headings_rad = np.arctan2(array_diff[:,0],array_diff[:,1])
    
        # compute the distances between the elements to use as weights in the circular mean
        weights = segment_lengths(xyz_n3)

        # take the circular mean to handle the [0 360] border
        # average the x and y components of the units vectors together weighted by the element distances
        # and take the angle of the result. Do not need to normalize the weights since the same normalization
        # will cancel in the arctan2
        xcomp = np.sum(np.sin(headings_rad) * weights)
        ycomp = np.sum(np.cos(headings_rad) * weights)

        heading_avg_rad = np.arctan2(xcomp,ycomp)

        # flip it so it's the heading to the first element and follows 0-360 convention
        avg_heading_deg = np.degrees(np.mod(heading_avg_rad+np.pi,2*np.pi))

        return avg_heading_deg
    
    # compute the spacing between the elements
    # this helper function returns one value for a line array or two values (distance along rows and columns) for 2D/3D arrays
    # which assumes elements are uniformly spaced
    def _compute_array_spacing(self):
        if self.array_type in ("panel_starboard","panel_port"):
            # By convetion, z is the fast dimension, we want one horiztonal stack at all the same z
            # so start at the first element and step every Nrows to get the next element in
            # at the same z
            xyz_n3 = self.xyz_m_N3[::self.Nrows,:]
            dcol = segment_lengths(xyz_n3)
            
            self._dcol_m = np.median(dcol)

            # z is the fast dimension so pull out the first vertical stack at the first
            # column element
            xyz_n3 = self.xyz_m_N3[:self.Nrows:,:]
            drow = segment_lengths(xyz_n3)
            
            self._drow_m = np.median(drow)
        elif self.array_type in ("sphere","cylinder"):
            # the horizontal spacing for a sphere can vary with elevation
            # so pull out one row near the middle for the maximum distance (lowest design frequency)
            start_row_index = int(np.floor(self.Nrows/2)-1)

            # By convetion, z is the fast dimension, we want one horiztonal stack at all the same z
            # so start at the first element and step every Nrows to get the next element in
            # at the same z
            xyz_n3 = self.xyz_m_N3[start_row_index::self.Nrows,:]
            dring = segment_lengths(xyz_n3)
           
            self._dring_m = np.median(dring)

            # z is the fast dimension so pull out the first vertical stack at the first
            # column element
            xyz_n3 = self.xyz_m_N3[:self.Nrows:,:]
            drow = segment_lengths(xyz_n3)
            
            self._drow_m = np.median(drow)
        else:
            self._dr_m = np.median(segment_lengths(self.xyz_m_N3))   

    # compute the design frequency of the array based on the element spacing and the soundspeed
    # returns a horizontal (azimuth) and vertical (elevation) design frequency for 2D/3D arrays
    def _compute_design_freq(self):
        if self.array_type in ("panel_starboard","panel_port"):
            self._design_freq_azimuth = self.soundspeed_mpers/(2*self.dcol_m)
            self._design_freq_elevation = self.soundspeed_mpers/(2*self.drow_m)

        elif self.array_type in ("sphere","cylinder"):
             self._design_freq_azimuth = self.soundspeed_mpers/(2*self.dring_m)
             self._design_freq_elevation = self.soundspeed_mpers/(2*self.drow_m)
        else:
            self._design_freq = self.soundspeed_mpers/(2*self.dr_m)

    # compute the normals for each element to be able to take the dot product with an incoming wave later
    # This assumes the normals for
    #   - panel arrays are perpendicular to the flat surface. The user needs to define which direction the normal
    #   faces with "panel_starboard" or "panel_port" when the geometry is read in
    #   - sphere array: normals point out from 0,0,0
    #   - cylindrical array: normals point out from 0,0,z
    def _compute_xyz_normals(self):
        if self.array_type in ("panel_starboard","panel_port"):
            if self.array_type=="panel_starboard":
                broadside_rad = np.mod(self.array_heading_deg + 90,360)*np.pi/180
            else:
                broadside_rad = np.mod(self.array_heading_deg + 270,360)*np.pi/180
             
            normal_13 = np.array([np.sin(broadside_rad), np.cos(broadside_rad), 0])
            self._normals_N3 = np.tile(normal_13, (self.xyz_m_N3.shape[0],1))

        elif self.array_type=="sphere":
            self._normals_N3 = self.xyz_m_N3/np.linalg.norm(self.xyz_m_N3, axis=1, keepdims=True)
        elif self.array_type=="cylinder":
            radius_inches = np.linalg.norm(self.xyz_m_N3[0,0:2])
            self._normals_N3 = np.zeros((self.xyz_m_N3.shape[0],3),dtype=np.float32)
            self._normals_N3[:,0:2] = self.xyz_m_N3[:,0:2]/radius_inches

    # Compue the array aperture to be used in calculations elsewhere (calculating beam widths at specific frequencies)
    def _compute_aperture(self):
        if self.array_type=="line":
            self._aperture_m = np.linalg.norm(self.xyz_m_N3[-1,:]-self.xyz_m_N3[0,:])
        elif self.array_type in ("panel_starboard","panel_port"):
            x = self.xyz_m_N3[::self.Nrows,0]
            y = self.xyz_m_N3[::self.Nrows,1]
            z = self.xyz_m_N3[:self.Nrows:,2]

            self._height_aperture_m = z[-1] - z[0]
            self._width_aperture_m = np.sqrt((x[-1]-x[0]) ** 2 + (y[-1] - y[0]) ** 2)
        elif self.array_type in ("cylinder","sphere"):
            z = self.xyz_m_N3[:self.Nrows:,2]
            radius_m = np.linalg.norm(self.xyz_m_N3[0,:])
            self._height_aperture_m = z[-1] - z[0]
            self._width_aperture_m = 2*radius_m
    
    # Compute the Fraunhofer (far field) distance metric 2*aperture^2/wavelength
    def _compute_far_field_range(self):
        if self.array_type=="line":
            
            aperture_m = self.aperture_m
            wavelength = self.soundspeed_mpers/(self.design_freq*2)       
            self._far_field_range_m = 2*(aperture_m ** 2)/wavelength

        elif self.array_type in ("panel_starboard","panel_port"):

            #  use the diagnonal distance across the aperture as a safe max to compute the far field
            aperture_m = np.sqrt(self.height_aperture_m ** 2 + self.width_aperture_m ** 2)
            max_design_freq = max((self.design_freq_azimuth, self.design_freq_elevation))
            wavelength = self.soundspeed_mpers/(max_design_freq*2)
            self._far_field_range_m = 2*(aperture_m ** 2)/wavelength

        elif self.array_type in ("cylinder","sphere"):
            
            #  use the diagnonal distance across the aperture as a safe max to compute the far field
            aperture_m = np.sqrt(self.height_aperture_m ** 2 + self.width_aperture_m ** 2)
            max_design_freq = max((self.design_freq_azimuth, self.design_freq_elevation))
            wavelength = self.soundspeed_mpers/(max_design_freq*2)
            self._far_field_range_m = 2*(aperture_m ** 2)/wavelength
        
        self.max_aperture_m = aperture_m; # we want to hang onto this to check for possible float32 rounding errors later
    
    # optional plotting        
    def plot_geometry(self):
        """Plot element positions, first-element marker, and element normals.

        Positions are displayed in feet. Normals are displayed for every
        geometry type except ``"line"``. The title reports heading, spacing,
        and spacing-derived design frequencies.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        ax : matplotlib.axes.Axes
            Three-dimensional geometry axes.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')

        xyz_ft_N3 = self.xyz_m_N3 * conv.M_TO_FT

        ax.plot(xyz_ft_N3[:,0],xyz_ft_N3[:,1],xyz_ft_N3[:,2],'k.')
        ax.plot(xyz_ft_N3[0,0],xyz_ft_N3[0,1],xyz_ft_N3[0,2],'ro',markerfacecolor='none')
        if self.array_type!="line":
            ax.quiver(xyz_ft_N3[:,0],xyz_ft_N3[:,1],xyz_ft_N3[:,2],\
                      self.normals_N3[:,0],self.normals_N3[:,1],self.normals_N3[:,2],\
                      normalize=False,color='b')

        ax.set_aspect('equal')
        ax.set_xlabel('X (ft)')
        ax.set_ylabel('Y (ft)')
        ax.set_zlabel('Z (ft)')
        ax.legend(['Array', 'First Element'])

        title_text1 = [f"Array type: {self.array_type}"]
        title_text1.append(f"Array Heading: {self.array_heading_deg:.0f} deg")
        first_line = " , ".join(title_text1[:2])

        if self.array_type == "line":
            title_text2 = [f"dr= {conv.M_TO_IN * self.dr_m:.0f} in"]
            title_text2.append(f"Design Frequency: {self.design_freq:.0f} Hz")
            second_line = " , ".join(title_text2[:2])
           
            final_text = first_line + "\n" + second_line
        elif self.array_type in ("panel_starboard","panel_port","sphere","cylinder"):
            
            if self.array_type in ("panel_starboard","panel_port"):
                title_text2 = [f"dcol= {conv.M_TO_IN * self.dcol_m:.0f} in"]
            elif self.array_type in ("sphere","cylinder"):
                title_text2 = [f"dring= {conv.M_TO_IN * self.dring_m:.0f} in"]
            
            title_text2.append(f"drow= {conv.M_TO_IN * self.drow_m:.0f} in")

            second_line = " , ".join(title_text2[:2])

            title_text3 = [rf"$f_{{design}}(\theta) = {self.design_freq_azimuth:.0f} \, \mathrm{{Hz}}$"]
            title_text3.append(rf"$f_{{design}}(\phi) = {self.design_freq_elevation:.0f} \, \mathrm{{Hz}}$")
            third_line =  " , ".join(title_text3[:2])
            final_text = first_line + "\n" + second_line + "\n" + third_line

        else:
            final_text = first_line
               
        ax.set_title(final_text)

        return fig, ax

    def select_elements(self, indices) -> BeamformingGeometry:
        """Select a beamforming subset in data-channel order.

        Parameters
        ----------
        indices : array-like of int, shape (K,)
            Zero-based element positions to retain. For panels, spheres, and
            cylinders, the selection must contain every combination of its
            selected rows and columns in original order, with rows varying
            fastest.

        Returns
        -------
        BeamformingGeometry
            Independent geometry data for the selected elements. Positions
            remain in the complete parent array's centered coordinate system;
            the returned phase center is the mean of the selected positions.

        Raises
        ------
        ValueError
            If ``indices`` is empty, not one-dimensional, not integer-valued,
            outside the element range, or not a complete ordered rectangular
            selection for a multidimensional array.

        Notes
        -----
        The returned object preserves normals computed from the complete
        parent array. It also retains the parent's heading and far-field
        distance so multiple subsets use a common steering reference and range
        approximation.
        """
        indices = np.asarray(indices)

        if (indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("indices must be a nonempty 1D integer array")

        if np.any(indices < 0) or np.any(indices >= self.xyz_m_N3.shape[0]):
            raise ValueError("Element index is outside the geometry")

        selected_nrows = None
        selected_ncols = None

        if self.array_type in ("panel_starboard", "panel_port", "sphere", "cylinder"):

            parent_nrows = self.Nrows
            parent_ncols = self.Ncols

            if (parent_nrows < 1 or parent_ncols < 1 or parent_nrows * parent_ncols != self.xyz_m_N3.shape[0]):
                raise ValueError("Parent geometry has inconsistent grid dimensions")

            # Rows vary fastest within each column.
            column_indices = indices // parent_nrows
            row_indices = indices % parent_nrows

            selected_columns = np.unique(column_indices)
            selected_rows = np.unique(row_indices)

            # Every selected column must contain the same selected rows,
            # in the original column-major order.
            expected_indices = (selected_columns[:, None] * parent_nrows + selected_rows[None, :]).ravel()

            if not np.array_equal(indices, expected_indices):
                raise ValueError(
                    "Selected geometry must contain every combination of "
                    "the selected rows and columns, in the original order "
                    "with rows varying fastest")

            selected_nrows = selected_rows.size
            selected_ncols = selected_columns.size

        # Resolve lazy properties privately without modifying the caller.
        parent = deepcopy(self)

        xyz_m_N3 = parent.xyz_m_N3[indices].copy()

        # Preserve the physical normals calculated from the complete array.
        normals_N3 = parent.normals_N3
        if normals_N3 is not None:
            normals_N3 = normals_N3[indices].copy()

        # This property also initializes parent.max_aperture_m.
        far_field_range_m = parent.far_field_range_m

        return BeamformingGeometry(array_type=parent.array_type,xyz_m_N3=xyz_m_N3,
            phase_center_m_13=np.mean(xyz_m_N3,axis=0,keepdims=True,dtype=np.float64),normals_N3=normals_N3,
            soundspeed_mpers=np.array(parent.soundspeed_mpers, copy=True),array_heading_deg=parent.array_heading_deg,
            far_field_range_m=far_field_range_m,max_aperture_m=parent.max_aperture_m,Nrows=selected_nrows,
            Ncols=selected_ncols)
