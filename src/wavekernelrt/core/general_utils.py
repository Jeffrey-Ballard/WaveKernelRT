import numpy as np
from wavekernelrt.core.validate_utils import validate_beam_grid
from matplotlib.ticker import AutoMinorLocator, NullLocator
import time
from contextlib import contextmanager
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt

# function that finds the "first" or "last" index of any nonzero elements
# used mostly for logical arrays to find first and last 1s.
def find_instance(arr,ftype):
    arr = np.asarray(arr)
    if ftype=="first":
        idxs = np.flatnonzero(arr) # returns 1D array of indices for non-zero elemetns
        idx = idxs[0] if idxs.size > 0 else None
    elif ftype=="last":
        idxs = np.flatnonzero(arr) # returns 1D array of indices for non-zero elemetns
        idx = idxs[-1] if idxs.size > 0 else None
    else:
        # throw error
        raise ValueError(f"ftype is {ftype}. Needs to be either 'first' or 'last'")

    return idx

# function that returns the index of the closest value val in array arr
# if arr is [1xN] and val is [1xM] the returned index with be [1xM]
def find_closest(arr,val):
    # force them to be np arrays
    arr = np.asarray(arr)
    val = np.asarray(val)

    # force them to be row and column vectors to broadcast against each other
    if arr.ndim==1:
        arr = arr[:,None]
    if val.ndim==1:
        val = val[None,:]

    idx = np.argmin(np.abs(arr - val),axis=0)
    
    return int(idx[0])

def find_closest_bearing(bearings_deg, target_deg):
    """Return the index of the bearing nearest to a scalar target, in degrees.

    ``bearings_deg`` must be a nonempty 1D array of finite real values;
    ``target_deg`` must be a finite real scalar. Angles may use any wrapping
    convention, including 0..360 and -180..180, and need not be sorted.
    Distance is measured along the shortest arc. Ties (including equivalent
    bearings such as 0 and 360) select the first index in the input array.
    Only supplied bearings are considered, so partial sectors are supported.
    """
    bearings = np.asarray(bearings_deg)
    target = np.asarray(target_deg)
    if bearings.ndim != 1 or bearings.size == 0:
        raise ValueError("bearings_deg must be a nonempty 1D array")
    if target.ndim != 0:
        raise ValueError("target_deg must be a scalar")
    for value in (bearings, target):
        if (not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or not np.all(np.isfinite(value))):
            raise ValueError("Bearings and target must be finite real numbers")

    # Wrap each operand first to avoid overflow when subtracting large angles.
    bearings = bearings.astype(np.float64) % 360.0
    target = float(target) % 360.0
    offsets = (bearings - target + 180.0) % 360.0 - 180.0
    return int(np.argmin(np.abs(offsets)))


def old_man_eyes(ax,minor_ticks="none"):

    if minor_ticks not in ["x", "y", "both", "none"]:
        raise ValueError("minor_axis must be 'x', 'y', 'both', or 'none'")

    ax.xaxis.label.set_fontsize(12)
    ax.xaxis.label.set_fontweight('bold')
    
    ax.yaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontweight('bold')

    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')

    ax.tick_params(axis='both',labelsize=12)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    if minor_ticks in ["x", "both"]:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True,which="minor",alpha=0.3)
    else:
        ax.xaxis.set_minor_locator(NullLocator())

    if minor_ticks in ["y", "both"]:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True,which="minor",alpha=0.3)
    else:
        ax.yaxis.set_minor_locator(NullLocator())


def set_bearing_ticks_for_width(ax,tick_intervals_deg=(15, 30, 45, 90),minimum_tick_spacing_px=45):
    """Choose bearing ticks from allowed intervals using the axes width.

    The finest interval that leaves at least ``minimum_tick_spacing_px``
    rendered pixels between adjacent ticks is selected. If none fits, the
    coarsest supplied interval is used. Existing axis limits are preserved.

    Returns
    -------
    float
        Selected tick interval in degrees.
    """

    intervals = np.asarray(tick_intervals_deg)
    if (intervals.ndim != 1
            or intervals.size == 0
            or not np.issubdtype(intervals.dtype, np.number)
            or np.iscomplexobj(intervals)
            or not np.all(np.isfinite(intervals))
            or np.any(intervals <= 0)
            or np.any(np.diff(intervals) <= 0)):
        raise ValueError(
            "tick_intervals_deg must contain strictly increasing positive "
            "finite values")

    spacing = np.asarray(minimum_tick_spacing_px)
    if (spacing.ndim != 0
            or not np.issubdtype(spacing.dtype, np.number)
            or np.iscomplexobj(spacing)
            or not np.isfinite(spacing)
            or spacing <= 0):
        raise ValueError("minimum_tick_spacing_px must be a finite positive number")

    low_deg, high_deg = sorted(ax.get_xlim())
    bearing_span_deg = high_deg - low_deg
    # Matplotlib reports the axes extent in device pixels on high-DPI GUI
    # backends. Normalize it to logical desktop pixels so moving a figure
    # between displays with different device-pixel ratios does not change the
    # selected tick interval when its logical size is unchanged.
    axes_width_px = (ax.get_window_extent().width / ax.figure.canvas.device_pixel_ratio)

    tick_interval_deg = float(intervals[-1])
    if bearing_span_deg > 0 and axes_width_px > 0:
        for candidate_deg in intervals:
            candidate_spacing_px = (axes_width_px * candidate_deg / bearing_span_deg)
            if candidate_spacing_px >= spacing:
                tick_interval_deg = float(candidate_deg)
                break

    first_tick_deg = np.ceil(low_deg / tick_interval_deg) * tick_interval_deg
    last_tick_deg = np.floor(high_deg / tick_interval_deg) * tick_interval_deg
    ax.set_xticks(np.arange(first_tick_deg,last_tick_deg + tick_interval_deg / 2,tick_interval_deg))

    return tick_interval_deg

@contextmanager
def timer_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] Elapsed: {end - start:.6f}s")

def clip_and_replace_dB(data_dB_QJ,car_filters):
    """Apply sequential guarded-mean, upper-peak replacement filters.

    The decision-filter structure is Hampel-like, but the reference is a
    guarded sideband mean and the threshold is a fixed one-sided dB offset;
    this is not a median/MAD Hampel filter.
    """
    x_dB_QJ = np.asarray(data_dB_QJ)
    
    for window,origin,threshold_dB in car_filters:
       
        bg_dB_QJ = convolve1d(x_dB_QJ,window,axis=0,mode="reflect",origin=origin)

        mask_QJ = x_dB_QJ > bg_dB_QJ + threshold_dB
        x_dB_QJ[mask_QJ] = bg_dB_QJ[mask_QJ]

    return x_dB_QJ

def make_clip_and_replace_filters(car_parms_WP,dtype=np.float32):
    """Build normalized guarded-sideband kernels for clip-and-replace."""
    filters = []

    for side, guard, threshold_dB in car_parms_WP:
        window = np.concatenate((np.ones(side),np.zeros(guard),np.ones(side))) / (2*side)
        origin = -1 if window.size % 2 == 0 else 0

        filters.append((window, origin, float(threshold_dB)))
        
    return filters

def unpack_beams_J3(beams_J3):
    """Return unique values of range, azimuth, and elevations
    
    Must pass a validation check for compliance with WaveKernelRT conventions
    
    Columns are range in meters, azimuth in radians, elevation in radians.
    Axes must be ascending, with range slowest and elevation fastest.
    Azimuths crossing north must be unwrapped, e.g. 358 through 362 degrees.
    """
    return validate_beam_grid(beams_J3)


def fig_pixel_size(axes_width_px,axes_height_px):

    left_px = 80
    right_px = 80
    bottom_px = 50
    top_px = 50

    dpi = 100

    figure_width_px = left_px + axes_width_px + right_px
    figure_height_px = bottom_px + axes_height_px + top_px

    fig = plt.figure(figsize=(figure_width_px / dpi, figure_height_px / dpi),dpi=dpi)

    ax = fig.add_axes([left_px / figure_width_px,bottom_px / figure_height_px,axes_width_px / figure_width_px,
        axes_height_px / figure_height_px])

    return fig, ax

def compute_Neffective_samples(K,N,advance,window,exponent=2):

    N = int(N)
    H = int(advance)
    m = (N-1)//H # of snapshops that overlap

    # compute the correlation value of the window at the desired overlap
    win_denom = np.sum(window**2)
    win_corr = np.empty(m)
    K_sum=0
    for mm in range(m):
        idx=N-1-(mm+1)*H+1 # +1 since zero based
        win_numer = np.sum(window[:idx] * window[((mm+1)*H):])
        win_corr[mm] = win_numer/win_denom
        K_sum+=(1-((mm+1)/K))*(win_corr[mm]**exponent)

    return K/(1+2*K_sum)

def generate_time_array_from_header(header):

    # handle timeseries file a little differently
    seq = np.arange(header["Nseq"])
    if "Npt_per_seq" in header:
        time_T = header["start_unix_seconds"] + seq*header["secs_per_record"]
    else:
        time_T = header["start_unix_seconds"] + seq*header["advance"]/header["sample_rate_hz"]

    return time_T
