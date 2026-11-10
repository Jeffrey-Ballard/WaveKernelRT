"""Construction and coordinate conversion helpers for steering grids."""

import numpy as np
from wavekernelrt.core.general_utils import find_instance
from wavekernelrt.core.validate_utils import validate_beam_grid

# helper function to take in range, bearing, and elevation and multiplex them together
# range is sent in as feet, but converted to meters for internal calculations
# bearing and elevation are sent in degrees, but converted to radians for internal calculations
def construct_rbe_X3(range_ft_R,bearing_deg_B,elevation_deg_E):
    """Construct an ordered range-bearing-elevation steering grid.

    Parameters
    ----------
    range_ft_R : array-like or scalar
        Ascending steering ranges in feet. Infinite values are represented by
        10,000,000 feet before conversion.
    bearing_deg_B : array-like or scalar
        Ascending navigation bearings in degrees clockwise from north. Sectors
        crossing north must be unwrapped, for example 358 through 362 degrees.
    elevation_deg_E : array-like or scalar
        Ascending elevations in degrees, positive upward.

    Returns
    -------
    numpy.ndarray, shape (R*B*E, 3), dtype float32
        Range in meters and angles in radians. Range varies slowest, bearing
        next, and elevation fastest.

    Raises
    ------
    ValueError
        If the inputs do not produce a finite, complete, ascending grid.

    Notes
    -----
    Inputs are converted with ``numpy.asarray``. A writable floating-point
    range array supplied by the caller is modified in place when it contains
    infinity.
    """

    # rbe_X3 is always multiplexed in WaveKernelRT as
    # range (r) is the slowest-varying index
    # bearing (b) is the middle index
    # elevation (e) is the fastest-varying index
    # r1 b1 e1
    # r1 b1 e2
    # r1 b1 e3
    # r1 b2 e1
    # r1 b2 e2
    # r1 b2 e3
    # ...

    range_ft_R = np.asarray(range_ft_R)
    bearing_deg_B = np.asarray(bearing_deg_B)
    elevation_deg_E = np.asarray(elevation_deg_E)

    R = range_ft_R.size
    B = bearing_deg_B.size
    E = elevation_deg_E.size

    X = R * B * E

    rbe_rad_K3 = np.empty((X,3), dtype=np.float32)

    # replace any inf with a large value 
    range_ft_R[np.isinf(range_ft_R)] = 10000000
    range_m_R = range_ft_R / 3.2808

    rbe_rad_K3[:,0] = np.repeat(range_m_R,B*E)
    rbe_rad_K3[:,1] = np.tile(np.repeat(bearing_deg_B*np.pi/180,E),R)
    rbe_rad_K3[:,2] = np.tile(elevation_deg_E*np.pi/180,R*B)

    # Make sure the grid is what WaveKernelRT will expect downstream
    validate_beam_grid(rbe_rad_K3)

    return rbe_rad_K3

# helper function that takes a range/bearing/elevation multiplexed matrix Kx3 and turns it into an xyz coordinate
# bearing is defined as 0 degrees north (+y), 90 degrees east (+x)
# range units doesn't matter here but is generally assumed to be meters for internal calculations
def convert_rbe_to_xyz(rbe_rad_K3):
    rbe_rad_K3 = np.asarray(rbe_rad_K3)
    if rbe_rad_K3.ndim==1:
        rbe_rad_K3 = rbe_rad_K3[None,:]

    vec_K3 = np.empty((rbe_rad_K3.shape[0],3), dtype=np.float32)

    vec_K3[:,0] = rbe_rad_K3[:,0] * np.sin(rbe_rad_K3[:,1]) * np.cos(rbe_rad_K3[:,2])
    vec_K3[:,1] = rbe_rad_K3[:,0] * np.cos(rbe_rad_K3[:,1]) * np.cos(rbe_rad_K3[:,2])
    vec_K3[:,2] = rbe_rad_K3[:,0] * np.sin(rbe_rad_K3[:,2])

    return vec_K3

# helper function that finds index of a mask necessary to circshift a template to then line up
# This was written to find how to shift a template of the shading for a spherical/cylinder array along
# the columns to line up with the desired look dir
# template = [1, 1, 1, 1, 1, 0, 0, 0]
# mask_C1  = [1, 1, 1, 0, 0, 0, 1, 1]
# It is assumed that mask_C1 is a logical and that 1's and 0's make up two contiguous blocks but can wrap around the boundary
#
# shift = find_circular_aperture_shift(mask_C1)
# shift would be 6 (0-based)
# shifted = np.roll(template,shift,axis=0)
# shifted = [1, 1, 1, 0, 0, 0, 1, 1]
def find_circular_aperture_shift(mask_C1):
    mask_C1 = np.asarray(mask_C1).reshape(-1,1)
    shift = find_instance(mask_C1,"first")
    if shift == 0:
        shift = find_instance(mask_C1==0,"last")+1

    return shift
