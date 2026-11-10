"""Plots for spectra, PSD, beam products, and white-noise gain (WNG).

Data and axis conventions
--------------------------
Q = frequency
N = channel
J = beams
T = time

Channel inputs use (T, Q, N) [dimensions of output from fft_read_multiple_sequence]
Beamformed inputs use (Q, J, T) [dimensions of output from bf_read_multiple_sequence]

Supplied coordinates are assumed to have one coordinate per bin (freq will have a frequency defined at every Q bin)
Omitted coordinates use zero-based positions

Spectrum, PSD, and WNG inputs are assumed to be in linear units (not dB).
BTR and generic scalar images display their supplied values directly.

Data below -400 dB is thresholded to -400 dB when converting to dB for display or dB averaging.
PSD data containing NaN, infinity, or negative values raises an error.

Spectrum is assumed to be one-sided μPa/Hz, normalized to preserve the total energy of two-sided spectrum
PSD is assumed to be one-sided μPa²/Hz, normalized to preserve the total energy of two-sided spectrum

Spectra data is complex. User may specify how to display the data with plot_type:
- "complex" displays the amplitude and phase
- "amplitude" displays only the amplitude, abs(data)
- "phase" displays only the phase, angle(data), displays in radians

User may further specify how spectra data is displayed with:
- amp_disp - "dB" or "linear" [default "dB"; 20*log10(data)]
- phs_disp - "wrapped" or "unwrapped"  [default "wrapped"]

and for PSD data
- mag_disp - "dB" or "linear" [default "dB"; 10*log10(data)]


Function naming convention
--------------------------
Each image function follows a specific naming pattern:
[data_type]_image_[porientation]

data_type:
fft_spectrum  - channel spectra from fft_read_multiple_sequence
psd           - channel PSD from fft_read_multiple_sequence
bfft_spectrum - beam spectra from bf_read_multiple_sequence
bpsd          - beam PSD from bf_read_multiple_sequence
wng           - beam white-noise gain from wng_read_multiple_sequence

plot_orientation:
Any two-letter combination of either (T, Q, N) from channel data or (Q, J, T) from beamformed data
The first letter is displayed as the row (y-axis).
The second letter is displayed as the column (x-axis).

Example: bfft_spectrum_image_QJ
Displays frequency vertically and beams horizontally (a frequency-azimuth image), while selecting
or reducing along the time T dimension.


Selection and reductions
------------------------
All data inputs are expected to be 3 dimensions, even if one of the dimensions is 1.

For a single beamformed record, use bf_read_multiple_sequence(header, [sequence])
to retain shape (Q, J, 1). The single-sequence reader returns (Q, J); add a
trailing axis with data_QJ[:, :, None] before passing its output to these functions.

The user must decide how to collapse the omitted dimension. Default behavior, if left unspecified,
is to take the first slice of the omitted dimension.

The user may specify how to operate on the omitted dimension with
idxT/idxN/idxJ/idxQ - zero-based indexes (not physical coordinates or channel IDs).
integer, range, or 1-D integer sequence
collapse_func - how to operate on the omitted dimension

Currently supported collapse_func options:
- None - first selected index is displayed
- "mean" - mean in linear for amplitudes/magnitudes
- "median"
- "min"
- "max"
- "mean_dB" - mean in dB for amplitudes/magnitudes

Reduction happens before display conversion. Therefore, amp_disp="dB",
collapse_func "mean" would mean the data in linear then display the result in dB.
Flooring to -400 dB is done before averaging with "mean_dB", while "mean" with
amp_disp/mag_disp is floored after averaging.

Special note for spectrum phase images:
- "mean", "mean_dB" and "median" map internally to "circular_mean" and "circular_median"
This is done so -179 and 179 degrees do not average to 0 degrees.

Default behavior:
- idxT/idxN/idxJ/idxQ unspecified/specified & collapse_func unspecified: take 1st slice to collapse dimension over
- idxT/idxN/idxJ/idxQ unspecified & collapse_func specified: operates over all of the collapse dimension
- idxT/idxN/idxJ/idxQ specified & collapse_func specified: operates on indices supplied in the collapse dimension

Titles describe how the collapse dimension was handled.


Coordinates, styling, and returned objects
-----------------------------------------
- times, numeric times are used as supplied (no conversion seconds/minutes/epoch times);
- time_label, describes the chosen units.
- channels, Channel IDs label equally spaced bins.

Horizontal channel axes show at most 10 labels.

Beam coordinates are angles in degrees and retain physical spacing;
set_beam_ticks places ticks at multiples of 30 within the current view.

Each image wrapper creates a figure and returns (fig, ax, pcm). For complex spectrum
plots, ax contains amplitude/phase axes in that order and pcm is a two-item
list of meshes. Single-component, PSD, WNG image, BTR, and generic scalar
plots return one Axes and one mesh. wng_line_T returns (fig, ax, line).

The caller can customize those objects and decides when to call plt.show().

Spectrum color limits are independent for amplitude and phase (radians).
- amp_vmin, amp_vmax, phs_vmin, phs_vmax for spectrum images
- vmin/vmax for PSD and ordinary WNG images

Unwrapped phase autoscales by default; wrapped phase defaults to [0, 2*pi].
Unwrapping follows frequency after reduction; a plot without frequency warns
and displays wrapped phase when unwrapping is requested.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import warnings
from numbers import Real
from wavekernelrt.core.general_utils import old_man_eyes,set_bearing_ticks_for_width
from wavekernelrt.core.collapse_functions import collapse_data, validate_collapse_func

__all__ = [
    "fft_spectrum_image_QN","fft_spectrum_image_NQ","fft_spectrum_image_QT","fft_spectrum_image_TQ",
    "fft_spectrum_image_NT","fft_spectrum_image_TN","bfft_spectrum_image_QJ","bfft_spectrum_image_JQ",
    "bfft_spectrum_image_QT","bfft_spectrum_image_TQ","bfft_spectrum_image_JT","bfft_spectrum_image_TJ",
    "psd_image_QN","psd_image_NQ","psd_image_QT","psd_image_TQ","psd_image_NT","psd_image_TN",
    "bpsd_image_QJ","bpsd_image_JQ","bpsd_image_QT","bpsd_image_TQ","bpsd_image_JT","bpsd_image_TJ",
    "wng_image_QJ","wng_line_T","btr_image","image_data_RC","create_spectrum_image","create_psd_image",
    "set_beam_ticks",
]

"""
Implementation layers
---------------------
Named wrappers assign coordinates/labels and orient the 2-D plane.
- create_spectrum_image convert display units and set component titles;
- create_psd_image convert display units and set component titles;
- image_data_RC draws the mesh/colorbar. Axis decoration and selection titles
are applied last. Keep domain-specific decisions out of image_data_RC.

_get_spectrum_data selects/reduces values

"""

# FFT Spectrum Image Functions
def fft_spectrum_image_QN(spectrum_TQN,*,plot_type="complex",freq=None,channels=None,idxT=None,collapse_func=None,
                    title=None, cmap="jet", amp_vmin=None, amp_vmax=None, phs_vmin=None, phs_vmax=None,amp_disp="dB",phs_disp="wrapped"):

    """Images Channel spectrum data. Frequency vertically x Channels horizontally.

    - spectrum_TQN: contains complex FFT values in (T, Q, N) order [output of fft_read_sequence]
    spectrum_TQN must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - freq: Frequency with one entry per Q bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimensions
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")

    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)

    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if channels is not None:
        col_label="Channels"
    else:
        col_label="Channels seq"

    channel_positions, channel_labels = _channel_plot_coordinates(channels, spectrum_TQN.shape[2])

    amp_QN, phs_QN = _get_spectrum_data(np.transpose(spectrum_TQN,(1,2,0)),plot_type=plot_type,idxP=idxT,pdim=2,collapse_func=collapse_func)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_QN,phs_QN,plot_type=plot_type,row_data=freq,col_data=channel_positions,
        row_label=row_label,col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,
        phs_vmax=phs_vmax,title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=0)

    # Keep every channel bin, but show at most 10 horizontal labels.
    step = max(1, int(np.ceil(len(channel_positions) / 10)))
    for axis in np.atleast_1d(ax).ravel():
        axis.set_xticks(channel_positions[::step], labels=channel_labels[::step])

    _append_collapse_title(ax, "T", idxT, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def fft_spectrum_image_NQ(spectrum_TQN,*,plot_type="complex",freq=None,channels=None,idxT=None,collapse_func=None,
                    title=None, cmap="jet", amp_vmin=None, amp_vmax=None, phs_vmin=None, phs_vmax=None,amp_disp="dB",phs_disp="wrapped"):

    """Images Channel spectrum data. Channels vertically x Frequency horizontally.

    - spectrum_TQN: contains complex FFT values in (T, Q, N) order [output of fft_read_sequence]
    spectrum_TQN must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (horizontal axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
     
    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if channels is not None:
        row_label="Channels"
    else:
        row_label="Channels seq"

    channel_positions, channel_labels = _channel_plot_coordinates(channels, spectrum_TQN.shape[2])

    amp_NQ, phs_NQ = _get_spectrum_data(np.transpose(spectrum_TQN,(1,2,0)),plot_type=plot_type,idxP=idxT,pdim=2,
        collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_NQ,phs_NQ,plot_type=plot_type,row_data=channel_positions,col_data=freq,
        row_label=row_label,col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,
        phs_vmax=phs_vmax,title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=1)

    for axis in np.atleast_1d(ax).ravel():
        axis.set_yticks(channel_positions, labels=channel_labels)

    _append_collapse_title(ax, "T", idxT, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def fft_spectrum_image_QT(spectrum_TQN,*,plot_type="complex",freq=None,times=None,idxN=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Channel spectrum data. Frequency vertically x Time horizontally.

    - spectrum_TQN: contains complex FFT values in (T, Q, N) order [output of fft_read_sequence]
    spectrum_TQN must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxN: selects the omitted channel dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the channel dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (vertical axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label

    amp_QT, phs_QT = _get_spectrum_data(np.transpose(spectrum_TQN,(1,2,0)),plot_type=plot_type,idxP=idxN,pdim=1,collapse_func=collapse_func)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_QT,phs_QT,plot_type=plot_type,row_data=freq,col_data=times,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=0)

    _append_collapse_title(ax, "N", idxN, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def fft_spectrum_image_TQ(spectrum_TQN,*,plot_type="complex",freq=None,times=None,idxN=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Channel spectrum data. Time vertically x Frequency horizontally.

    - spectrum_TQN: contains complex FFT values in (T, Q, N) order [output of fft_read_sequence]
    spectrum_TQN must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxN: selects the omitted channel dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the channel dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (horizontal axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    amp_TQ, phs_TQ = _get_spectrum_data(np.transpose(spectrum_TQN,(1,2,0)),plot_type=plot_type,idxP=idxN,pdim=1,
        collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_TQ,phs_TQ,plot_type=plot_type,col_data=freq,row_data=times,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=1)

    _append_collapse_title(ax, "N", idxN, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def fft_spectrum_image_NT(spectrum_TQN,*,plot_type="complex",channels=None,times=None,idxQ=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Channel spectrum data. Channels vertically x Time horizontally.

    - spectrum_TQN: contains complex FFT values in (T, Q, N) order [output of fft_read_sequence]
    spectrum_TQN must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is not applicable because the frequency dimension has been removed.
    Requesting "unwrapped" for phase issues a warning and displays wrapped phase.
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if channels is not None:
        row_label="Channels"
    else:
        row_label="Channels seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label

    channel_positions, channel_labels = _channel_plot_coordinates(channels, spectrum_TQN.shape[2])

    amp_NT, phs_NT = _get_spectrum_data(np.transpose(spectrum_TQN,(1,2,0)),plot_type=plot_type,idxP=idxQ,pdim=0,collapse_func=collapse_func)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_NT,phs_NT,plot_type=plot_type,col_data=times,row_data=channel_positions,
        row_label=row_label,col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,
        phs_vmax=phs_vmax,title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp)

    for axis in np.atleast_1d(ax).ravel():
        axis.set_yticks(channel_positions, labels=channel_labels)

    _append_collapse_title(ax, "Q", idxQ, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def fft_spectrum_image_TN(spectrum_TQN,*,plot_type="complex",channels=None,times=None,idxQ=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Channel spectrum data. Time vertically x Channels horizontally.

    - spectrum_TQN: contains complex FFT values in (T, Q, N) order [output of fft_read_sequence]
    spectrum_TQN must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is not applicable because the frequency dimension has been removed.
    Requesting "unwrapped" for phase issues a warning and displays wrapped phase.
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if channels is not None:
        col_label="Channels"
    else:
        col_label="Channels seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    channel_positions, channel_labels = _channel_plot_coordinates(channels, spectrum_TQN.shape[2])

    amp_TN, phs_TN = _get_spectrum_data(np.transpose(spectrum_TQN,(1,2,0)),plot_type=plot_type,idxP=idxQ,pdim=0,
        collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_TN,phs_TN,plot_type=plot_type,row_data=times,col_data=channel_positions,
        row_label=row_label,col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,
        phs_vmax=phs_vmax,title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp)

    # Keep every channel bin, but show at most 10 horizontal labels.
    step = max(1, int(np.ceil(len(channel_positions) / 10)))
    for axis in np.atleast_1d(ax).ravel():
        axis.set_xticks(channel_positions[::step], labels=channel_labels[::step])

    _append_collapse_title(ax, "Q", idxQ, collapse_func, plot_type=plot_type)

    return fig, ax, pcm


# BFFT Spectrum Image Functions
def bfft_spectrum_image_QJ(spectrum_QJT,*,plot_type="complex",freq=None,beams=None,idxT=None,collapse_func=None,
                    title=None, cmap="jet", amp_vmin=None, amp_vmax=None, phs_vmin=None, phs_vmax=None,amp_disp="dB",phs_disp="wrapped"):
    
    """Images Beamformed spectrum data. Frequency vertically x Beams horizontally.

    - spectrum_QJT: contains complex FFT values in (Q, J, T) order [output of bf_read_multiple_sequence]
    spectrum_QJT must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (vertical axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if beams is not None:
        col_label="Beams"
    else:
        col_label="Beams seq"

    amp_QJ, phs_QJ = _get_spectrum_data(spectrum_QJT,plot_type=plot_type,idxP=idxT,pdim=2,collapse_func=collapse_func)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_QJ,phs_QJ,plot_type=plot_type,row_data=freq,col_data=beams,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=0)

    if beams is not None:
        set_beam_ticks(ax, "x")

    _append_collapse_title(ax, "T", idxT, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def bfft_spectrum_image_JQ(spectrum_QJT,*,plot_type="complex",freq=None,beams=None,idxT=None,collapse_func=None,
                    title=None, cmap="jet", amp_vmin=None, amp_vmax=None, phs_vmin=None, phs_vmax=None,amp_disp="dB",phs_disp="wrapped"):
    
    """Images Beamformed spectrum data. Beams vertically x Frequency horizontally.

    - spectrum_QJT: contains complex FFT values in (Q, J, T) order [output of bf_read_multiple_sequence]
    spectrum_QJT must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (horizontal axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if beams is not None:
        row_label="Beams"
    else:
        row_label="Beams seq"

    amp_JQ, phs_JQ = _get_spectrum_data(spectrum_QJT,plot_type=plot_type,idxP=idxT,pdim=2,collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_JQ,phs_JQ,plot_type=plot_type,col_data=freq,row_data=beams,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=1)

    if beams is not None:
        set_beam_ticks(ax, "y")

    _append_collapse_title(ax, "T", idxT, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def bfft_spectrum_image_QT(spectrum_QJT,*,plot_type="complex",freq=None,times=None,idxJ=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Beamformed spectrum data. Frequency vertically x Time horizontally.

    - spectrum_QJT: contains complex FFT values in (Q, J, T) order [output of bf_read_multiple_sequence]
    spectrum_QJT must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxJ: selects the omitted beam dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the beam dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (vertical axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label
        

    amp_QT, phs_QT = _get_spectrum_data(spectrum_QJT,plot_type=plot_type,idxP=idxJ,pdim=1,collapse_func=collapse_func)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_QT,phs_QT,plot_type=plot_type,row_data=freq,col_data=times,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=0)

    _append_collapse_title(ax, "J", idxJ, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def bfft_spectrum_image_TQ(spectrum_QJT,*,plot_type="complex",freq=None,times=None,idxJ=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Beamformed spectrum data. Time vertically x Frequency horizontally.

    - spectrum_QJT: contains complex FFT values in (Q, J, T) order [output of bf_read_multiple_sequence]
    spectrum_QJT must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxJ: selects the omitted beam dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the beam dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is along the frequency dimension (horizontal axis), after reduction
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    amp_TQ, phs_TQ = _get_spectrum_data(spectrum_QJT,plot_type=plot_type,idxP=idxJ,pdim=1,collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_TQ,phs_TQ,plot_type=plot_type,col_data=freq,row_data=times,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp,qdim=1)

    _append_collapse_title(ax, "J", idxJ, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def bfft_spectrum_image_JT(spectrum_QJT,*,plot_type="complex",beams=None,times=None,idxQ=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Beamformed spectrum data. Beams vertically x Time horizontally.

    - spectrum_QJT: contains complex FFT values in (Q, J, T) order [output of bf_read_multiple_sequence]
    spectrum_QJT must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is not applicable because the frequency dimension has been removed.
    Requesting "unwrapped" for phase issues a warning and displays wrapped phase.
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if beams is not None:
        row_label="Beams"
    else:
        row_label="Beams Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label
        

    amp_JT, phs_JT = _get_spectrum_data(spectrum_QJT,plot_type=plot_type,idxP=idxQ,pdim=0,collapse_func=collapse_func)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_JT,phs_JT,plot_type=plot_type,row_data=beams,col_data=times,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp)

    if beams is not None:
        set_beam_ticks(ax, "y")

    _append_collapse_title(ax, "Q", idxQ, collapse_func, plot_type=plot_type)

    return fig, ax, pcm

def bfft_spectrum_image_TJ(spectrum_QJT,*,plot_type="complex",beams=None,times=None,idxQ=None,collapse_func=None,
    title=None,cmap="jet",amp_vmin=None,amp_vmax=None,phs_vmin=None,phs_vmax=None,amp_disp="dB",phs_disp="wrapped",
    time_label=None):

    """Images Beamformed spectrum data. Time vertically x Beams horizontally.

    - spectrum_QJT: contains complex FFT values in (Q, J, T) order [output of bf_read_multiple_sequence]
    spectrum_QJT must be 3 dimensional, including singleton dimensions
    - plot_type: "amplitude", "phase", or "complex" (both).
    Amplitude is abs(data), not PSD.
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for amplitudes
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for amplitudes

    Reduction happens before display conversion. Therefore, amp_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    Phase uses circular mean/median for mean/median requests
    amplitude uses the requested reducer unchanged.

    - amp_disp: selects "dB" (20*log10) or "linear".
    - phs_disp: selects "wrapped" or "unwrapped"
    Unwrapped is not applicable because the frequency dimension has been removed.
    Requesting "unwrapped" for phase issues a warning and displays wrapped phase.
    
    - amp_vmin/amp_vmax: color limits for amplitude plot
    - phs_vmin/phs_vmax: color limits for phase plot [radians; wrapped defaults to 0 to 2*pi if unspecified]

    Returns (fig, ax, pcm). Complex mode returns two axes and a mesh list,
    ordered amplitude then phase; other modes return a single axis and mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "amplitude")
    _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func)
    
    if beams is not None:
        col_label="Beams"
    else:
        col_label="Beams Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    amp_TJ, phs_TJ = _get_spectrum_data(spectrum_QJT,plot_type=plot_type,idxP=idxQ,pdim=0,collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_spectrum_plot(plot_type)

    pcm = create_spectrum_image(amp_TJ,phs_TJ,plot_type=plot_type,row_data=times,col_data=beams,row_label=row_label,
        col_label=col_label,cmap=cmap,amp_vmin=amp_vmin,amp_vmax=amp_vmax,phs_vmin=phs_vmin,phs_vmax=phs_vmax,
        title=title,ax=ax,amp_disp=amp_disp,phs_disp=phs_disp)

    if beams is not None:
        set_beam_ticks(ax, "x")

    _append_collapse_title(ax, "Q", idxQ, collapse_func, plot_type=plot_type)

    return fig, ax, pcm


# Channel PSD Images
def psd_image_QN(psd_TQN,*,freq=None,channels=None,idxT=None,collapse_func=None,title=None,cmap="jet",vmin=None,vmax=None,mag_disp="dB"):

    """Images Channel power spectral density (PSD) data. Frequency vertically x Channels horizontally.

    - psd_TQN: contains nonnegative linear PSD values in (T, Q, N) order [output of fft_read_sequence]
    psd_TQN must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - freq: Frequency with one entry per Q bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if channels is not None:
        col_label="Channels"
    else:
        col_label="Channels seq"

    channel_positions, channel_labels = _channel_plot_coordinates(channels, psd_TQN.shape[2])

    mag_QN, _ = _get_spectrum_data(np.transpose(psd_TQN,(1,2,0)),plot_type="magnitude",idxP=idxT,pdim=2,collapse_func=collapse_func)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_QN,row_data=freq,col_data=channel_positions,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    # Keep every channel bin, but show at most 10 horizontal labels.
    step = max(1, int(np.ceil(len(channel_positions) / 10)))
    for axis in np.atleast_1d(ax).ravel():
        axis.set_xticks(channel_positions[::step], labels=channel_labels[::step])

    _append_collapse_title(ax, "T", idxT, collapse_func)

    return fig, ax, pcm

def psd_image_NQ(psd_TQN,*,freq=None,channels=None,idxT=None,collapse_func=None,title=None,cmap="jet",vmin=None,vmax=None,mag_disp="dB"):

    """Images Channel power spectral density (PSD) data. Channels vertically x Frequency horizontally.

    - psd_TQN: contains nonnegative linear PSD values in (T, Q, N) order [output of fft_read_sequence]
    psd_TQN must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if channels is not None:
        row_label="Channels"
    else:
        row_label="Channels seq"

    channel_positions, channel_labels = _channel_plot_coordinates(channels, psd_TQN.shape[2])

    mag_NQ, _ = _get_spectrum_data(np.transpose(psd_TQN,(1,2,0)),plot_type="magnitude",idxP=idxT,pdim=2,
        collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_psd_plot()
    
    pcm = create_psd_image(mag_NQ,col_data=freq,row_data=channel_positions,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    for axis in np.atleast_1d(ax).ravel():
        axis.set_yticks(channel_positions, labels=channel_labels)

    _append_collapse_title(ax, "T", idxT, collapse_func)

    return fig, ax, pcm

def psd_image_QT(psd_TQN,*,freq=None,times=None,idxN=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):

    """Images Channel power spectral density (PSD) data. Frequency vertically x Time horizontally.

    - psd_TQN: contains nonnegative linear PSD values in (T, Q, N) order [output of fft_read_sequence]
    psd_TQN must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxN: selects the omitted channel dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the channel dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label

    mag_QT, _ = _get_spectrum_data(np.transpose(psd_TQN,(1,2,0)),plot_type="magnitude",idxP=idxN,pdim=1,
        collapse_func=collapse_func,flag_transpose=False)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_QT,row_data=freq,col_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)
    
    _append_collapse_title(ax, "N", idxN, collapse_func)

    return fig, ax, pcm

def psd_image_TQ(psd_TQN,*,freq=None,times=None,idxN=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):

    """Images Channel power spectral density (PSD) data. Time vertically x Frequency horizontally.

    - psd_TQN: contains nonnegative linear PSD values in (T, Q, N) order [output of fft_read_sequence]
    psd_TQN must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxN: selects the omitted channel dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the channel dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    mag_TQ, _ = _get_spectrum_data(np.transpose(psd_TQN,(1,2,0)),plot_type="magnitude",idxP=idxN,pdim=1,
        collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_TQ,col_data=freq,row_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)
    
    _append_collapse_title(ax, "N", idxN, collapse_func)

    return fig, ax, pcm

def psd_image_NT(psd_TQN,*,channels=None,times=None,idxQ=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):

    """Images Channel power spectral density (PSD) data. Channels vertically x Time horizontally.

    - psd_TQN: contains nonnegative linear PSD values in (T, Q, N) order [output of fft_read_sequence]
    psd_TQN must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if channels is not None:
        row_label="Channels"
    else:
        row_label="Channels Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label
        

    channel_positions, channel_labels = _channel_plot_coordinates(channels, psd_TQN.shape[2])

    mag_NT, _ = _get_spectrum_data(np.transpose(psd_TQN,(1,2,0)),plot_type="magnitude",idxP=idxQ,pdim=0,
        collapse_func=collapse_func,flag_transpose=False)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_NT,row_data=channel_positions,col_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)
    
    for axis in np.atleast_1d(ax).ravel():
        axis.set_yticks(channel_positions, labels=channel_labels)

    _append_collapse_title(ax, "Q", idxQ, collapse_func)

    return fig, ax, pcm

def psd_image_TN(psd_TQN,*,channels=None,times=None,idxQ=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):

    """Images Channel power spectral density (PSD) data. Time vertically x Channels horizontally.

    - psd_TQN: contains nonnegative linear PSD values in (T, Q, N) order [output of fft_read_sequence]
    psd_TQN must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - channels: supplies IDs for equally spaced bins, not physical coordinates, one entry per N bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if channels is not None:
        col_label="Channels"
    else:
        col_label="Channels Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        
    channel_positions, channel_labels = _channel_plot_coordinates(channels, psd_TQN.shape[2])

    mag_TN, _ = _get_spectrum_data(np.transpose(psd_TQN,(1,2,0)),plot_type="magnitude",idxP=idxQ,pdim=0,
        collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_TN,col_data=channel_positions,row_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)
    
    # Keep every channel bin, but show at most 10 horizontal labels.
    step = max(1, int(np.ceil(len(channel_positions) / 10)))
    for axis in np.atleast_1d(ax).ravel():
        axis.set_xticks(channel_positions[::step], labels=channel_labels[::step])

    _append_collapse_title(ax, "Q", idxQ, collapse_func)

    return fig, ax, pcm


# Beamformed PSD Images
def bpsd_image_QJ(bpsd_QJT,*,freq=None,beams=None,idxT=None,collapse_func=None,title=None,cmap="jet",vmin=None,vmax=None,mag_disp="dB"):
    
    """Images Beamformed power spectral density (PSD) data. Frequency vertically x Beams horizontally.

    - bpsd_QJT: contains nonnegative linear PSD values in (Q, J, T) order [output of bf_read_multiple_sequence]
    bpsd_QJT must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if beams is not None:
        col_label="Beams"
    else:
        col_label="Beams seq"

    mag_QJ, _ = _get_spectrum_data(bpsd_QJT,plot_type="magnitude",idxP=idxT,pdim=2,collapse_func=collapse_func)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_QJ,row_data=freq,col_data=beams,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    if beams is not None:
        set_beam_ticks(ax, "x")

    _append_collapse_title(ax, "T", idxT, collapse_func)

    return fig, ax, pcm

def bpsd_image_JQ(bpsd_QJT,*,freq=None,beams=None,idxT=None,collapse_func=None,title=None,cmap="jet",vmin=None,vmax=None,mag_disp="dB"):
    
    """Images Beamformed power spectral density (PSD) data. Beams vertically x Frequency horizontally.

    - bpsd_QJT: contains nonnegative linear PSD values in (Q, J, T) order [output of bf_read_multiple_sequence]
    bpsd_QJT must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - idxT: selects the omitted time dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the time dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if beams is not None:
        row_label="Beams"
    else:
        row_label="Beams seq"

    mag_JQ, _ = _get_spectrum_data(bpsd_QJT,plot_type="magnitude",idxP=idxT,pdim=2,collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_JQ,col_data=freq,row_data=beams,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    if beams is not None:
        set_beam_ticks(ax, "y")

    _append_collapse_title(ax, "T", idxT, collapse_func)

    return fig, ax, pcm

def bpsd_image_QT(bpsd_QJT,*,freq=None,times=None,idxJ=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):
    
    """Images Beamformed power spectral density (PSD) data. Frequency vertically x Time horizontally.

    - bpsd_QJT: contains nonnegative linear PSD values in (Q, J, T) order [output of bf_read_multiple_sequence]
    bpsd_QJT must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxJ: selects the omitted beam dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the beam dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        row_label="Frequency"
    else:
        row_label="Frequency Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label

    mag_QT, _ = _get_spectrum_data(bpsd_QJT,plot_type="magnitude",idxP=idxJ,pdim=1,collapse_func=collapse_func,flag_transpose=False)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_QT,row_data=freq,col_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    _append_collapse_title(ax, "J", idxJ, collapse_func)

    return fig, ax, pcm

def bpsd_image_TQ(bpsd_QJT,*,freq=None,times=None,idxJ=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):
    
    """Images Beamformed power spectral density (PSD) data. Time vertically x Frequency horizontally.

    - bpsd_QJT: contains nonnegative linear PSD values in (Q, J, T) order [output of bf_read_multiple_sequence]
    bpsd_QJT must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - freq: Frequency with one entry per Q bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxJ: selects the omitted beam dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the beam dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if freq is not None:
        col_label="Frequency"
    else:
        col_label="Frequency Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    mag_TQ, _ = _get_spectrum_data(bpsd_QJT,plot_type="magnitude",idxP=idxJ,pdim=1,collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_TQ,col_data=freq,row_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    _append_collapse_title(ax, "J", idxJ, collapse_func)

    return fig, ax, pcm

def bpsd_image_JT(bpsd_QJT,*,beams=None,times=None,idxQ=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):
    
    """Images Beamformed power spectral density (PSD) data. Beams vertically x Time horizontally.

    - bpsd_QJT: contains nonnegative linear PSD values in (Q, J, T) order [output of bf_read_multiple_sequence]
    bpsd_QJT must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if beams is not None:
        row_label="Beams"
    else:
        row_label="Beams Seq"

    if times is not None:
        col_label="Time"
    else:
        col_label="Time Seq"

    if time_label is not None:
        col_label=time_label
        
    mag_JT, _ = _get_spectrum_data(bpsd_QJT,plot_type="magnitude",idxP=idxQ,pdim=0,collapse_func=collapse_func,flag_transpose=False)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_JT,row_data=beams,col_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    if beams is not None:
        set_beam_ticks(ax, "y")

    _append_collapse_title(ax, "Q", idxQ, collapse_func)

    return fig, ax, pcm

def bpsd_image_TJ(bpsd_QJT,*,beams=None,times=None,idxQ=None,collapse_func=None,
                    title=None, cmap="jet", vmin=None, vmax=None,mag_disp="dB",time_label=None):
    
    """Images Beamformed power spectral density (PSD) data. Time vertically x Beams horizontally.

    - bpsd_QJT: contains nonnegative linear PSD values in (Q, J, T) order [output of bf_read_multiple_sequence]
    bpsd_QJT must be 3 dimensional, including singleton dimensions
    Input is already computed PSD, not complex FFT values or dB values.
    The title assumes PSD calibrated in μPa²/Hz.
    - beams: supplies angular coordinates in degrees with physical spacing, one entry per J bin
    - times: Time coordinates with one entry per T bin, used without unit conversion
    - time_label: overrides the time-axis label to describe the supplied units
    - idxQ: selects the omitted frequency dimension by zero-based position
    supply an integer, range, or 1-D integer sequence. None selects
    all positions for a reduction.

    - collapse_func: how to reduce the frequency dimension. None displays only the 
    first selected position  
    
    Currently supported collapse_func options:
    - None - first selected index is displayed
    - "mean" - mean in linear for power
    - "median"
    - "min"
    - "max"
    - "mean_dB" - mean in dB for power

    Reduction happens before display conversion. Therefore, mag_disp="dB",
    collapse_func "mean" would mean the data in linear then display the result in dB.

    Coordinates describe the displayed dimensions and must match the data.
    Missing coordinates use zero-based positions; inputs are not modified.
    
    - mag_disp: selects "dB" (10*log10) or "linear".
    
    - vmin/vmax: color limits for the PSD plot in the units selected by mag_disp
    Limits are automatically selected when unspecified.

    Returns (fig, ax, pcm): one figure, one axis, and its color mesh.
    title adds a prefix; component units and selection details are appended.
    cmap selects the colormap.
    
    Displaying/saving the figure is up to the caller. (plt.show() or (plt.savefig()))
    """
    collapse_func = _select_collapse_func(collapse_func, "power")
    _validate_psd_image_inputs(mag_disp,collapse_func)

    if beams is not None:
        col_label="Beams"
    else:
        col_label="Beams Seq"

    if times is not None:
        row_label="Time"
    else:
        row_label="Time Seq"

    if time_label is not None:
        row_label=time_label
        

    mag_TJ, _ = _get_spectrum_data(bpsd_QJT,plot_type="magnitude",idxP=idxQ,pdim=0,collapse_func=collapse_func,flag_transpose=True)

    fig, ax = _get_psd_plot()

    pcm = create_psd_image(mag_TJ,col_data=beams,row_data=times,row_label=row_label,col_label=col_label,
                            cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax,mag_disp=mag_disp)

    if beams is not None:
        set_beam_ticks(ax, "x")

    _append_collapse_title(ax, "Q", idxQ, collapse_func)

    return fig, ax, pcm


# White-noise-gain diagnostics
def wng_image_QJ(wng_QJT,*,freq=None,beams=None,idxT=None,constraint_wng=None,title=None,cmap="viridis",
    constraint_cmap="Greys_r",failure_cmap="jet",vmin=None,vmax=None,wng_disp="dB"):
    """Image one WNG snapshot with frequency rows and beam columns.

    ``wng_QJT`` contains positive linear WNG with shape ``(Q, J, T)``.
    ``idxT`` selects one stored record and defaults to zero. ``freq`` and
    ``beams`` optionally provide physical frequency and relative-bearing
    coordinates.

    With no ``constraint_wng``, the image shows absolute WNG using ``cmap``.
    When a constraint is supplied, the image instead shows
    ``constraint_dB - achieved_WNG_dB``, clipped to ``[-10, 10]`` dB. Passing
    values from -10 to 0 dB use ``constraint_cmap`` (black to white), while
    failing values from 0 to 10 dB use ``failure_cmap``. Zero is exactly on
    the WNG constraint. The constraint may be scalar, one value per beam with
    shape ``(J,)``, or one value per frequency and beam with shape ``(Q, J)``.
    Constraint values are linear WNG, matching the input data.

    ``wng_disp`` selects ``"dB"`` (``10*log10``) or ``"linear"`` display.
    ``vmin`` and ``vmax`` are interpreted in those display units for the
    absolute image. The constraint-relative image always uses -10 to 10 dB.

    Returns ``(fig, ax, mesh)``. Absolute and constraint-relative images use
    the same fixed data-axis and colorbar regions. The reserved title area is
    large enough for the longer constraint diagnostic, so a shorter title does
    not cause the plotted data to move or resize between images.
    """

    wng_QJT = _validate_wng_data(wng_QJT)
    time_index = _validate_single_index(idxT, wng_QJT.shape[2], "idxT")
    display_QJ, unit = _convert_wng_display(wng_QJT[:, :, time_index],wng_disp)

    if freq is None:
        row_data = np.arange(display_QJ.shape[0])
    else:
        _coordinate_at(freq, 0, display_QJ.shape[0], "freq")
        row_data = np.asarray(freq)
    if beams is None:
        col_data = np.arange(display_QJ.shape[1])
    else:
        _coordinate_at(beams, 0, display_QJ.shape[1], "beams")
        col_data = np.asarray(beams)
    row_label = "Frequency" if freq is not None else "Frequency Seq"
    col_label = "Relative bearing (deg)" if beams is not None else "Beams Seq"

    # Keep comparison exports registered pixel-for-pixel. Constrained layout
    # otherwise shrinks the data axis when the constraint diagnostic adds title
    # lines or uses a longer colorbar label.
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_axes([0.09, 0.082, 0.757, 0.728])
    colorbar_ax = fig.add_axes([0.885, 0.082, 0.031, 0.728])

    if constraint_wng is None:
        plot_vmin, plot_vmax = _resolve_wng_limits(display_QJ, vmin, vmax)
        _, _, mesh = image_data_RC(display_QJ,row_data=row_data,col_data=col_data,row_label=row_label,
            col_label=col_label,cmap=cmap,vmin=plot_vmin,vmax=plot_vmax,ax=ax,colorbar=False)
        description = f"Absolute white-noise gain [{unit}]"
        colorbar_label = f"WNG [{unit}]"
    else:
        constraint_QJ = _broadcast_wng_constraint(constraint_wng,display_QJ.shape)
        achieved_dB_QJ = 10.0 * np.log10(wng_QJT[:, :, time_index])
        constraint_dB_QJ = 10.0 * np.log10(constraint_QJ)
        difference_dB_QJ = np.clip(constraint_dB_QJ - achieved_dB_QJ,-10.0,10.0)
        threshold_cmap = _wng_constraint_colormap(passing_cmap=constraint_cmap,failing_cmap=failure_cmap)
        _, _, mesh = image_data_RC(difference_dB_QJ,row_data=row_data,col_data=col_data,row_label=row_label,
            col_label=col_label,cmap=threshold_cmap,vmin=-10.0,vmax=10.0,ax=ax,colorbar=False)
        constraint_description = _describe_wng_constraint(constraint_dB_QJ)
        description = (
            "WNG constraint − achieved WNG [dB]\n"
            f"Effective WNGC: {constraint_description}; "
            "negative passes, positive fails")
        colorbar_label = "WNGC − achieved WNG [dB]"

    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    if constraint_wng is not None:
        colorbar.set_ticks([-10.0, -5.0, 0.0, 5.0, 10.0])
    colorbar.set_label(colorbar_label)

    plot_title = description if title is None else f"{title}\n{description}"
    ax.set_title(plot_title, pad=12)
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")
    if beams is not None:
        set_beam_ticks(ax, "x")
    _append_collapse_title(ax, "T", time_index)

    return fig, ax, mesh


def wng_line_T(wng_QJT,*,idxQ,idxJ,times=None,freq=None,beams=None,constraint_wng=None,title=None,wng_disp="dB",
    time_label=None,color="tab:blue"):
    """Plot WNG through time for one frequency and beam.

    ``idxQ`` and ``idxJ`` are zero-based positions in linear ``wng_QJT``.
    ``times`` supplies one coordinate per record and defaults to record index.
    If supplied, ``constraint_wng`` accepts the same scalar, ``(J,)``, or
    ``(Q, J)`` forms as :func:`wng_image_QJ` and is drawn as a reference line.

    Returns ``(fig, ax, line)`` for the WNG trace. A constraint reference is
    available as the second entry in ``ax.lines`` when requested.
    """

    wng_QJT = _validate_wng_data(wng_QJT)
    frequency_index = _validate_single_index(idxQ, wng_QJT.shape[0], "idxQ")
    beam_index = _validate_single_index(idxJ, wng_QJT.shape[1], "idxJ")

    if times is None:
        times_T = np.arange(wng_QJT.shape[2])
        x_label = "Time Seq"
    else:
        times_T = np.asarray(times)
        if times_T.ndim != 1 or times_T.size != wng_QJT.shape[2]:
            raise ValueError(f"times must contain {wng_QJT.shape[2]} values, one per record")
        if times_T.dtype.kind not in "iuf" or not np.all(np.isfinite(times_T)):
            raise ValueError("times must contain finite real values")
        x_label = "Time"
    if time_label is not None:
        x_label = time_label

    trace_T, unit = _convert_wng_display(wng_QJT[frequency_index, beam_index, :],wng_disp)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    line, = ax.plot(times_T, trace_T, color=color, label="Achieved WNG")

    if constraint_wng is not None:
        constraint_QJ = _broadcast_wng_constraint(constraint_wng,wng_QJT.shape[:2])
        constraint_value, _ = _convert_wng_display(constraint_QJ[frequency_index, beam_index],wng_disp)
        ax.axhline(float(constraint_value),color="tab:red",linestyle="--",linewidth=1.5,label="Effective WNG constraint")

    selection = [f"Q index {frequency_index}", f"J index {beam_index}"]
    if freq is not None:
        selection[0] = f"{_coordinate_at(freq, frequency_index, wng_QJT.shape[0], 'freq'):g} Hz"
    if beams is not None:
        selection[1] = (f"{_coordinate_at(beams, beam_index, wng_QJT.shape[1], 'beams'):g} deg")
    description = f"White-noise gain at {', '.join(selection)}"
    ax.set_title(description if title is None else f"{title}\n{description}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"WNG [{unit}]")
    ax.grid(True)
    old_man_eyes(ax)
    ax.legend()

    return fig, ax, line


# general helper functions
def _validate_wng_data(wng_QJT):
    """Return positive, finite real WNG data with shape (Q, J, T)."""

    data = np.asarray(wng_QJT)
    if data.ndim != 3:
        raise ValueError("wng_QJT must have shape (Q, J, T)")
    if any(size == 0 for size in data.shape):
        raise ValueError("wng_QJT dimensions must be nonempty")
    if data.dtype.kind not in "iuf" or not np.all(np.isfinite(data)):
        raise ValueError("WNG data must contain finite real values")
    if np.any(data <= 0):
        raise ValueError("WNG data must be strictly positive")
    return data


def _validate_single_index(index, size, name):
    """Validate one zero-based array position."""

    if index is None:
        index = 0
    values = np.asarray(index)
    if (values.ndim != 0 or values.dtype.kind not in "iu" or values.dtype.kind == "b"):
        raise ValueError(f"{name} must be one integer index")
    result = int(values)
    if not 0 <= result < size:
        raise IndexError(f"{name}={result} is outside [0, {size})")
    return result


def _convert_wng_display(values, wng_disp):
    """Convert positive linear WNG values to requested display units."""

    if not isinstance(wng_disp, str) or wng_disp.lower() not in ("db", "linear"):
        raise ValueError(f"wng_disp must be 'dB' or 'linear'; got {wng_disp!r}")
    values = np.asarray(values)
    if wng_disp.lower() == "db":
        return 10.0 * np.log10(values), "dB"
    return values, "linear"


def _broadcast_wng_constraint(constraint_wng, shape_QJ):
    """Validate and broadcast a linear WNG constraint to (Q, J)."""

    constraint = np.asarray(constraint_wng)
    if constraint.dtype.kind not in "iuf" or not np.all(np.isfinite(constraint)):
        raise ValueError("constraint_wng must contain finite real values")
    if np.any(constraint <= 0):
        raise ValueError("constraint_wng must be strictly positive")
    try:
        return np.broadcast_to(constraint, shape_QJ)
    except ValueError as exc:
        raise ValueError("constraint_wng must be scalar, shape (J,), or shape (Q, J)") from exc


def _wng_constraint_colormap(passing_cmap, failing_cmap):
    """Build the fixed -10-to-10 dB WNG constraint diagnostic colormap."""

    passing_colors = plt.get_cmap(passing_cmap)(np.linspace(0.0, 1.0, 128))
    failing_base = plt.get_cmap(failing_cmap)(np.linspace(0.0, 1.0, 128))

    # End the passing side at white, then jump directly to the first failure
    # color. The deliberate discontinuity makes even a small positive deficit
    # visually distinct from a value that still satisfies the constraint.
    failing_colors = failing_base.copy()
    passing_colors[-1, :3] = 1.0

    return ListedColormap(np.vstack((passing_colors, failing_colors)),name="wng_constraint_difference")


def _describe_wng_constraint(constraint_dB_QJ):
    """Format a scalar or varying effective absolute WNG constraint."""

    minimum = float(np.min(constraint_dB_QJ))
    maximum = float(np.max(constraint_dB_QJ))
    if np.isclose(minimum, maximum):
        return f"{minimum:.2f} dB absolute"
    return f"{minimum:.2f} to {maximum:.2f} dB absolute across beams"


def _resolve_wng_limits(display_QJ, vmin, vmax):
    """Resolve one shared normalization for colored and grayscale WNG cells."""

    plot_vmin = float(np.min(display_QJ)) if vmin is None else vmin
    plot_vmax = float(np.max(display_QJ)) if vmax is None else vmax
    if not isinstance(plot_vmin, Real) or not np.isfinite(plot_vmin):
        raise ValueError("vmin must be finite")
    if not isinstance(plot_vmax, Real) or not np.isfinite(plot_vmax):
        raise ValueError("vmax must be finite")
    if plot_vmin > plot_vmax:
        raise ValueError("vmin must not exceed vmax")
    if plot_vmin == plot_vmax:
        padding = max(0.5, abs(float(plot_vmin)) * 0.01)
        plot_vmin -= padding
        plot_vmax += padding
    return plot_vmin, plot_vmax


def _coordinate_at(coordinates, index, size, name):
    """Return one validated numeric coordinate from a plotting axis."""

    values = np.asarray(coordinates)
    if values.ndim != 1 or values.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite real values")
    return float(values[index])


def _channel_plot_coordinates(channels, count):
    """Use uniform channel positions, with channel IDs only as tick labels."""
    positions = np.arange(count)
    labels = positions if channels is None else np.asarray(channels)
    if labels.ndim != 1 or labels.size != count:
        raise ValueError(f"channels must contain {count} labels, one per data channel")
    return positions, labels

def set_beam_ticks(ax, coordinate):
    """Set 30-degree ticks on 'x' or 'y' without expanding the plot limits.

    Call after plotting angular beam coordinates. Accepts one Axes or an
    array of subplots, including inverted axes. Bounds come from the current
    view (including pcolormesh cell edges), so no beam array is required.
    If no multiple of 30 falls in the view, leave existing ticks in place.
    """
    if coordinate not in ("x", "y"):
        raise ValueError(f"coordinate must be 'x' or 'y'; got {coordinate!r}")

    for axis in np.atleast_1d(ax).ravel():
        limits = axis.get_xlim() if coordinate == "x" else axis.get_ylim()
        low, high = sorted(limits)
        first = np.ceil(low / 30) * 30
        last = np.floor(high / 30) * 30
        ticks = np.arange(first, last + 15, 30)
        if ticks.size:
            if coordinate == "x":
                axis.set_xticks(ticks)
            else:
                axis.set_yticks(ticks)

def _phase_collapse_func(collapse_func):
    """Resolve the phase reducer for both computation and title text."""
    # Exact aliases keep unrelated future registry names from matching accidentally.
    return {"mean": "circular_mean","mean_amp_dB": "circular_mean","mean_pwr_dB": "circular_mean",
        "median": "circular_median"}.get(collapse_func, collapse_func)

def _append_collapse_title(ax, dim, indices=None, collapse_func=None, *, plot_type=None):
    """Append the slice/reduction description to one axis or a subplot array.

    Call after creating the plot and setting its main title, for example:

        _append_collapse_title(ax, "J", idxJ, collapse_func)

    dim is the original dimension name (Q, N, J, or T), not an axis number.
    Pass the original indices so ranges retain their start, stop, and step.
    With no collapse function, the first supplied index is used (default 0),
    matching _get_spectrum_data. Otherwise, omitted indices mean all indices
    of that dimension in the supplied data. Indices are zero-based.

    Spectrum wrappers pass plot_type so phase means/medians are labeled
    "circular mean"/"circular median". Complex plots use amplitude first and phase second;
    amplitude and PSD retain the requested reducer name.
    """
    if collapse_func is None:
        index = 0 if indices is None else int(np.atleast_1d(indices)[0])
        line = f"{dim} Dim collapsed by slice at idx{dim}={index}"
    elif indices is None:
        line = f"{dim} Dim collapsed by {collapse_func} over all idx{dim}"
    else:
        if isinstance(indices, range):
            if indices.step == 1:
                selection = f"range({indices.start},{indices.stop})"
            else:
                selection = f"range({indices.start},{indices.stop},{indices.step})"
        else:
            selected = np.atleast_1d(indices).astype(int)
            if selected.size == 1:
                selection = str(selected[0])
            else:
                selection = np.array2string(selected, separator=",", max_line_width=1000000)
        line = f"{dim} Dim collapsed by {collapse_func} over idx{dim}={selection}"

    # The same phase resolver drives both calculations and their title wording.
    for i, axis in enumerate(np.atleast_1d(ax).ravel()):
        axis_line = line
        is_phase = plot_type is not None and (plot_type.lower() == "phase" or (plot_type.lower() == "complex" and i == 1))
        if is_phase and collapse_func is not None:
            phase_reducer = _phase_collapse_func(collapse_func).replace("_", " ")
            axis_line = line.replace(f"collapsed by {collapse_func}",f"collapsed by {phase_reducer}",1)
        title = axis.get_title()
        axis.set_title(f"{title}\n{axis_line}" if title else axis_line)
        axis.title.set_fontsize(14)
        axis.title.set_fontweight('bold')

def _validate_spectrum_image_inputs(plot_type,amp_disp,phs_disp,collapse_func):

    """Check spectrum option names and registry membership before plotting.

    Display/plot choices are case-insensitive; reducer names are exact registry
    key defined in collapse_functions.py. This helper does not normalize or return options, inspect array
    shapes, or check coordinate lengths. Circular reducer compatibility with
    the requested component is checked separately in _get_spectrum_data.
    """
    if not isinstance(plot_type, str) or plot_type.lower() not in ("amplitude", "phase","complex"):
        raise ValueError(f"plot_type must be 'amplitude', 'phase', or 'complex'; got {plot_type!r}")

    if not isinstance(amp_disp, str) or amp_disp.lower() not in ("db", "linear"):
        raise ValueError(f"amp_disp must be 'dB' or 'linear'; got {amp_disp!r}")

    if not isinstance(phs_disp, str) or phs_disp.lower() not in ("wrapped", "unwrapped"):
        raise ValueError(f"phs_disp must be 'wrapped' or 'unwrapped'; got {phs_disp!r}")

    if collapse_func is not None:
        validate_collapse_func(collapse_func)

    return

def _validate_psd_image_inputs(mag_disp,collapse_func):

    """Check the PSD display mode and reducer registry membership.

    mag_disp accepts case-insensitive dB/linear. This checks option names,
    not calibration, nonnegativity, array shape, or coordinate lengths.
    """
    if not isinstance(mag_disp, str) or mag_disp.lower() not in ("db", "linear"):
        raise ValueError(f"mag_disp must be 'dB' or 'linear'; got {mag_disp!r}")

    if collapse_func is not None:
        validate_collapse_func(collapse_func)

    return

def _get_spectrum_plot(plot_type):

    """Create the axes expected by create_spectrum_image.

    Complex mode returns a two-element Axes array sharing x/y coordinates;
    amplitude is left and phase is right. Other modes return a single Axes.
    The caller validates plot_type before calling this helper.
    """
    match plot_type.lower():
        case "complex":
            fig,ax = plt.subplots(1,2,figsize=(14,5),sharex=True,sharey=True)
        case _:
            fig,ax = plt.subplots(1,1,figsize=(7,5))

    return fig, ax

def _get_psd_plot():

    """Create and return (fig, ax) for a single PSD image; do not show it.
    """
    fig,ax = plt.subplots(1,1,figsize=(7,5))

    return fig, ax

def _get_spectrum_data(spectrum_XYZ,plot_type,idxP=None,pdim=None,collapse_func=None,flag_transpose=False):
    """Select/reduce a 3-D array and return (amplitude_or_PSD, phase).

    spectrum_XYZ - already in the wrapper's working axis order.
    pdim - the numeric axis to remove; it must be provided.
    idxP - selects integer positions along that axis (None means the full dimension). Without a
    reducer, only the first selected position is used. Empty selections fail.

    plot_type
    - "complex" computes abs(data) and angle(data)
    - "amplitude" computes only abs(data)
    - "phase" computes only angle(data).
    
    The internal "magnitude" mode also uses abs(data) and is used with linear PSD inputs.
    Unrequested outputs are None. This helper does not convert to dB,
    unwrapped phase, normalize FFTs to PSD, or apply physical calibration.

    Phase mean/median use registered circular_mean/circular_median reducers.
    mean_dB also map to circular_mean for phase only
    
    Amplitude/PSD get the requested reducer without unit conversion.
    
    Explicit circular reducers require phase-only mode. Min/max on phase
    use numerical extrema of wrapped radians, not circular order statistics.

    Reduction removes only pdim. Remaining axes keep their input order.
    
    flag_transpose=True swaps dimensions for the reversed display orientation.
    
    Existing output arrays are transposed independently, preserving None.
    Inputs are not modified.
    """
    if collapse_func in ("circular_mean", "circular_median") and plot_type.lower() != "phase":
        raise ValueError("Circular reducers require plot_type='phase'; use 'mean' or 'median' for combined plots")

    if not np.all(np.isfinite(spectrum_XYZ)):
        raise ValueError(
            "Cannot image data containing NaN or +/-Inf. "
            "Check the upstream processing.")

    # Preserve the omitted axis during selection so reducers see the same pdim.
    if idxP is not None:
        idxP = np.atleast_1d(idxP).astype(int)
    else:
        idxP = np.arange(spectrum_XYZ.shape[pdim])

    if idxP.size == 0:
        raise ValueError("At least one index must be selected along the collapsed dimension")

    # A fixed pair of outputs keeps callers predictable in single-component mode.
    amp_AB = None
    phs_AB = None

    # Reduce amplitude and phase independently; averaging complex FFTs directly
    # would be a different (coherent) operation.
    match plot_type.lower():
        case "complex":
            amp_XYZ = np.abs(spectrum_XYZ)
            phs_XYZ = np.angle(spectrum_XYZ)
            if collapse_func is not None:
                
                amp_AB = collapse_data(np.take(amp_XYZ,idxP,pdim),pdim,collapse_func)

                phs_AB = collapse_data(np.take(phs_XYZ, idxP, axis=pdim),pdim,_phase_collapse_func(collapse_func))
            else:
                amp_AB = np.take(amp_XYZ,idxP[0],pdim)
                phs_AB = np.take(phs_XYZ,idxP[0],pdim)

        case "amplitude" | "magnitude":
            if plot_type.lower() == "magnitude":
                if np.iscomplexobj(spectrum_XYZ):
                    raise ValueError("PSD data must be real")

                if np.any(spectrum_XYZ < 0):
                    raise ValueError("PSD data must be nonnegative")
    
            amp_XYZ = np.abs(spectrum_XYZ)
            if collapse_func is not None:
                amp_AB = collapse_data(np.take(amp_XYZ,idxP,pdim),pdim,collapse_func)
            else:
                amp_AB = np.take(amp_XYZ,idxP[0],pdim)

        case "phase":
            phs_XYZ = np.angle(spectrum_XYZ)
            if collapse_func is not None:
                phs_AB = collapse_data(np.take(phs_XYZ, idxP, axis=pdim),pdim,_phase_collapse_func(collapse_func))
            else:
                phs_AB = np.take(phs_XYZ,idxP[0],pdim)

    # Check computed outputs, including results produced without a reducer.
    for name, data in (("Amplitude/PSD", amp_AB), ("Phase", phs_AB)):
        if data is not None and not np.all(np.isfinite(data)):
            raise ValueError(
                f"{name} calculation produced NaN or +/-Inf "
                "from finite input. Check for numerical overflow.")
        
    # Reverse only the displayed dimensions; an unrequested output stays None.
    if flag_transpose:
        if amp_AB is not None:
            amp_BA = amp_AB.T
        else:
            amp_BA = None

        if phs_AB is not None:
            phs_BA = phs_AB.T
        else:
            phs_BA = None

        return amp_BA, phs_BA
    else:
        return amp_AB, phs_AB

def create_spectrum_image(amp_RC,phs_RC,plot_type,row_data,col_data,row_label,col_label,cmap,amp_vmin,amp_vmax,
    phs_vmin,phs_vmax,ax,title,amp_disp,phs_disp,qdim=2):
    """Draw prepared 2-D spectrum components on existing axes.

    amp_RC - linear amplitude
    phs_RC -  phase in radians
    Both can be None when not requested by plot_type.
    
    Rows are vertical, columns horizontal.
    row_data/col_data and their labels follow that same order.
    
    Selection/reduction and orientation have already been performed.

    amp_disp chooses linear values or 20*log10(amplitude).
    Amplitude limits use those display units.
    
    phase limits are independent and use radians.
    Wrapped phase is mapped to [0, 2*pi), defaulting unspecified limits to 0 and 2*pi.

    None autoscales amplitude and unwrapped phase. 

    qdim identifies the frequency axis in the prepared plane: 0 for rows,
    1 for columns, or 2 when the image has no frequency dimension.
    
    Unwrapping uses qdim; requesting it without frequency warns and falls back to wrapped.

    ax is a single Axes or the amplitude/phase pair from _get_spectrum_plot.

    Each component gets a colorbar and title prefixed by title if supplied.
    
    Returns one QuadMesh or [amplitude_mesh, phase_mesh]; modifies axes in
    place. The wrapper adds the reduction description afterward.
    """


    if amp_RC is not None:
        if amp_disp.lower() == "db":
            amp_RC = 20*np.log10(np.maximum(amp_RC,1e-20))
            amp_unit = "dB"
        else:
            amp_unit = "linear"

    if phs_RC is not None and phs_disp.lower() == "unwrapped" and qdim == 2:
        warnings.warn("Frequency-axis unwrapping is unavailable because this plot "
            "has no frequency axis. Displaying wrapped phase instead.",UserWarning,stacklevel=3)

    if phs_RC is not None:
        if phs_disp.lower() == "unwrapped" and qdim<2:
            # Frequency can be rows or columns after the wrapper transposes.
            phs_RC = np.unwrap(phs_RC,axis=qdim)
            phs_modifer = "Unwrapped"
        else:
            phs_RC = phs_RC%(2*np.pi)
            phs_modifer = "Wrapped"
            if phs_vmin is None:
                phs_vmin=0
            if phs_vmax is None:
                phs_vmax=2*np.pi

    match plot_type.lower():
        case "complex":
            pcm=[None]*len(ax)
            _, _, pcm[0] = image_data_RC(amp_RC,row_data=row_data,col_data=col_data,row_label=row_label,col_label=col_label,
                                        cmap=cmap, vmin=amp_vmin, vmax=amp_vmax, ax=ax[0])

            if title is not None:
                title_str1 = title + f"\nAmplitude [{amp_unit}]"
            else:
                title_str1 = f"Amplitude [{amp_unit}]"

            ax[0].set_title(title_str1)
            ax[0].title.set_fontsize(14)
            ax[0].title.set_fontweight('bold')

            _, _, pcm[1] = image_data_RC(phs_RC,row_data=row_data,col_data=col_data,row_label=row_label,col_label=col_label,
                                                    cmap=cmap, vmin=phs_vmin, vmax=phs_vmax, ax=ax[1])

            if title is not None:
                title_str2 = title + f"\nPhase {phs_modifer} [rad]"
            else:
                title_str2 = f"Phase {phs_modifer} [rad]"

            ax[1].set_title(title_str2)
            ax[1].title.set_fontsize(14)
            ax[1].title.set_fontweight('bold')

        case "amplitude":

            _, _, pcm = image_data_RC(amp_RC,row_data=row_data,col_data=col_data,row_label=row_label,col_label=col_label,
                                                    cmap=cmap, vmin=amp_vmin, vmax=amp_vmax, ax=ax)

            if title is not None:
                title_str1 = title + f"\nAmplitude [{amp_unit}]"
            else:
                title_str1 = f"Amplitude [{amp_unit}]"

            ax.set_title(title_str1)
            ax.title.set_fontsize(14)
            ax.title.set_fontweight('bold')

        case "phase":

            _, _, pcm = image_data_RC(phs_RC,row_data=row_data,col_data=col_data,row_label=row_label,col_label=col_label,
                                                                cmap=cmap, vmin=phs_vmin, vmax=phs_vmax, ax=ax)

            if title is not None:
                title_str2 = title + f"\nPhase {phs_modifer} [rad]"
            else:
                title_str2 = f"Phase {phs_modifer} [rad]"

            ax.set_title(title_str2)
            ax.title.set_fontsize(14)
            ax.title.set_fontweight('bold')
            
    return pcm

def create_psd_image(mag_RC,row_data,col_data,row_label,col_label,cmap,vmin,vmax,ax,title,mag_disp):
   
    """Draw prepared 2-D linear PSD values on an existing axis.

    mag_RC has rows vertical and columns horizontal.
    
    Coordinates and labels follow row_data/col_data.
    Selection and reduction are already complete.
    
    mag_disp="dB" applies 10*log10, otherwise values remain linear
    vmin/vmax use display units.
    
    This performs no physical calibration; the title
    assumes microPascal squared per Hz. Zero values become -400 in dB.

    Adds the mesh, colorbar, and component title to ax and returns the mesh.
    The wrapper appends the slice/reduction description afterward.
    """
    if mag_disp.lower() == "db":
        mag_RC = 10*np.log10(np.maximum(mag_RC,1e-40))
        mag_unit = "dB re 1 μPa²/Hz"
    else:
        mag_unit = "linear μPa²/Hz"

    _, _, pcm = image_data_RC(mag_RC,row_data=row_data,col_data=col_data,row_label=row_label,col_label=col_label,
                                            cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    description = f"Power Spectral Density [{mag_unit}]"
    if title is not None:
        title_str1 = f"{title}\n{description}"
    else:
        title_str1 = description

    ax.set_title(title_str1)
    ax.title.set_fontsize(14)
    ax.title.set_fontweight('bold')
            
    return pcm

def btr_image(data_RC,row_data=None,col_data=None,row_label=None,col_label=None,cmap="jet",vmin=None,vmax=None,ax=None,
    *,pixels_per_sample=1.0,colorbar=False,bearing_tick_intervals_deg=(15, 30, 45, 90),
    minimum_bearing_tick_spacing_px=45):
    """Draw a bearing-time image; return ``(fig, ax, pcm)``.

    data_RC is a nonempty 2-D array with time in rows and bearing in columns.
    Values are displayed directly (no dB conversion). Coordinates may be
    cell centers or edges; omitted coordinates use zero-based centers.
    A singleton center receives a unit-width cell; supply two edges when
    its physical width matters. Labels describe the supplied coordinate units.

    pixels_per_sample is a finite positive number of logical (desktop) pixels
    per row/column, default 1.0. Use 2.0 to enlarge or 0.5 to shrink the image.
    Additional label/title margins are also in logical pixels. The backend's
    device pixel ratio handles Retina/high-DPI rendering: at ratio 2, one
    logical pixel occupies two physical pixels along each dimension. Uniformly
    spaced coordinates therefore give pixels_per_sample per sample. Nonuniform
    coordinates retain their physical spacing and have unequal cell widths.
    This specifies the initial size, not a monitor-fitting policy. Window
    resizing, IDE preview scaling, tight_layout, or saving at another DPI
    changes the pixel mapping. Fractional sizes may round to device pixels.
    A supplied ax's figure is resized and its
    layout engine disabled; use a dedicated figure for this image.

    cmap, vmin, and vmax control the stored values' color mapping. Set
    colorbar=True to add a vertical colorbar in space reserved beside the
    image; the data area's pixel-per-sample dimensions remain unchanged. That
    space is also reserved when the colorbar is disabled so rank and float32
    exports have identical axes and canvas dimensions. The default is False
    for backward compatibility with rank-order displays. The caller can set a
    title and call plt.show() or fig.savefig().

    Bearing ticks automatically select the finest interval in
    ``bearing_tick_intervals_deg`` that leaves at least
    ``minimum_bearing_tick_spacing_px`` between labels. The selection updates
    when the figure window is resized.
    """
    data_RC = np.asarray(data_RC)
    if data_RC.ndim != 2 or 0 in data_RC.shape:
        raise ValueError("BTR image data must be a nonempty 2-D array")

    if (isinstance(pixels_per_sample, (bool, np.bool_))
            or not isinstance(pixels_per_sample, Real)
            or not np.isfinite(pixels_per_sample) or pixels_per_sample <= 0):
        raise ValueError("pixels_per_sample must be a finite positive number")
    if not isinstance(colorbar, (bool, np.bool_)):
        raise ValueError("colorbar must be True or False")

    n_times, n_bearings = data_RC.shape

    def cell_edges(coordinates, count):
        if coordinates is None:
            return np.arange(count + 1) - 0.5
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 1 or coordinates.size not in (count, count + 1):
            raise ValueError("Coordinates must supply one center per cell or one more edge")
        if coordinates.size == count + 1:
            return coordinates
        if count == 1:
            return np.array([coordinates[0] - 0.5, coordinates[0] + 0.5])
        return np.concatenate(([coordinates[0] - (coordinates[1] - coordinates[0]) / 2],
            coordinates[:-1] + np.diff(coordinates) / 2,[coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2]))

    # Extra symmetric title margin makes room for descriptive multi-line BTR
    # titles without changing the requested data-area pixel dimensions.
    title_side_margin_px = 50
    left_px = 140 + title_side_margin_px
    colorbar_gap_px = 20
    colorbar_width_px = 20
    colorbar_right_px = 80
    # Always reserve the colorbar region. Rank images leave it blank, keeping
    # their canvas, data axis, labels, and title aligned with float32 images.
    right_px = (colorbar_gap_px + colorbar_width_px + colorbar_right_px + title_side_margin_px)
    bottom_px, top_px = 100, 100
    data_width_px = n_bearings * pixels_per_sample
    data_height_px = n_times * pixels_per_sample
    width_px = left_px + data_width_px + right_px
    height_px = bottom_px + data_height_px + top_px
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        fig = ax.figure

    fig.set_layout_engine(None)

    # GUI backends may already have multiplied figure DPI for Retina.
    # Divide out that factor to size the figure in logical desktop pixels.
    logical_dpi = fig.dpi / fig.canvas.device_pixel_ratio
    fig.set_size_inches(width_px / logical_dpi, height_px / logical_dpi, forward=True)
    ax.set_aspect("auto")
    ax.set_position([left_px / width_px,bottom_px / height_px,data_width_px / width_px,data_height_px / height_px])
    
    fig, ax, pcm = image_data_RC(data_RC,row_data=cell_edges(row_data, n_times),
        col_data=cell_edges(col_data, n_bearings),row_label=row_label,col_label=col_label,cmap=cmap,vmin=vmin,
        vmax=vmax,ax=ax,colorbar=False)
    pcm.set_snap(True)
    pcm.set_antialiased(False)

    if colorbar:
        colorbar_left_px = left_px + data_width_px + colorbar_gap_px
        colorbar_ax = fig.add_axes([colorbar_left_px / width_px,bottom_px / height_px,colorbar_width_px / width_px,
            data_height_px / height_px])
        cbar = fig.colorbar(pcm, cax=colorbar_ax)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")

    if col_data is not None:
        set_bearing_ticks_for_width(ax,bearing_tick_intervals_deg,minimum_bearing_tick_spacing_px)

        def update_bearing_ticks(event):
            set_bearing_ticks_for_width(ax,bearing_tick_intervals_deg,minimum_bearing_tick_spacing_px)
            event.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", update_bearing_ticks)

    return fig, ax, pcm


def image_data_RC(data_RC,row_data=None,col_data=None,row_label=None,col_label=None,cmap="jet",vmin=None,vmax=None,ax=None,*,colorbar=True):

    """Draw a 2-D scalar array and return (fig, ax, pcm).

    data_RC rows map to vertical row_data and columns to horizontal col_data.
    
    Missing coordinates default independently to zero-based positions.
    
    Coordinates are forwarded to pcolormesh(shading="auto"); use compatible
    cell centers or edges.
    
    No selection, transposition, unit conversion,
    phase processing, or domain-specific tick placement happens here.

    cmap/vmin/vmax control the supplied values' color mapping.
    
    An existing ax is modified in place; otherwise create a figure and one axis.
    
    Adds a colorbar unless colorbar=False, and the shared old_man_eyes style.
    pcm is the QuadMesh handle
    for later customization. Does not show the figure or modify data_RC.
    """
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(7,5))
    else:
        fig = ax.figure

    if row_data is None:
        row_data = np.arange(data_RC.shape[0])

    if col_data is None:
        col_data = np.arange(data_RC.shape[1])

    # pcolormesh takes horizontal coordinates first, then vertical coordinates.
    pcm = ax.pcolormesh(col_data,row_data,data_RC,cmap=cmap,shading="auto",vmin=vmin,vmax=vmax)

    if col_label is not None:
        ax.set_xlabel(col_label)

    if row_label is not None:
        ax.set_ylabel(row_label)

    if colorbar:
        cbar = fig.colorbar(pcm)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")

    old_man_eyes(ax)

    return fig, ax, pcm

def _select_collapse_func(collapse_func,image_type):
    """Resolve mean_dB before validation: amplitude uses 20log10, power 10log10.

    Both reducers accept linear values, average in dB, and return linear
    values for display conversion. Other reducer names pass through unchanged.
    Phase subsequently maps either resolved mean to circular_mean.
    """

    if collapse_func == "mean_dB":
        match image_type.lower():
            case "amplitude":
                collapse_func = "mean_amp_dB"
            case "power":
                collapse_func = "mean_pwr_dB"

    return collapse_func
