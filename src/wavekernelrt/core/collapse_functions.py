"""Named reductions along one array axis for analysis and plotting.

collapse_data(data, dim, name)
1. validates the request against a list of known functions in _COLLAPSE_FUNCTIONS
2. Calls the requested function and operates on the requested dimension.

Removes requested dimension, but preserves singleton dimensions elsewhere.
These functions do not select indices, transpose output, or modify input data;
callers handle those steps.

Currently supported functions
- None - selects the first slice along the requested axis.
- "mean" - mean in units sent in
- "median"
- "min"
- "max"
- "mean_amp_dB" - mean in dB for amplitudes.
means 20*log10(data) along requested dimension and returns it back in linear values
- "mean_pwr_dB" - mean in dB for power.
means 10*log10(data) along requested dimension and returns it back in linear values
- "circular_mean" - accept real angles in radians and return
radians. For phase to handle the -180/180 or 0/360 boundary, not complex.
Every angle is equally weighted. Equal opposing directions have no unique circular mean. 
The result is due to roundoff. 

- "circular_median" - same as circular_mean but a median instead of a mean.

This module does not infer whether an array represents amplitude, power, or phase.

Note: "mean_dB" is a plotting option in analysis_images that is then resolved to
mean_amp_dB or mean_pwr_dB. It is not accepted directly into collapse_data

To add a collapse function:
1. Define a function accepting the axis keyword before _COLLAPSE_FUNCTIONS
2. Add function name to _COLLAPSE_FUNCTIONS:
"new_collapse_func": new_collapse_func,
Validation & error message update automatically
"""

import numpy as np

def first_slice(data,axis):
    """Take index 0 along axis, removing that axis only; preserve input units.

    This is a selection, not an average. An empty selected axis raises an
    indexing error; plotting callers check for empty selections earlier.
    """
    return np.take(data,0,axis)

def mean_amp_dB(data,axis):
   """Accept linear amplitudes, average their dB levels, and return linear amplitude.

   Converts to 20*log10(data) average along axis, then returns 10**(data*0.05).

   Data below -400 dB is set to -400 dB before averaging.
   """
   data = np.asarray(data)

   if np.iscomplexobj(data):
        raise ValueError("Amplitude data must be real")

   if np.any(data < 0):
        raise ValueError("Amplitude data must be nonnegative")
   
   return 10**(np.mean(20*np.log10(np.maximum(data,1e-20)),axis=axis)*0.05)

def mean_pwr_dB(data,axis):
   """Accept linear power, average their dB levels, and return linear power.

   Converts to 10*log10(data) average along axis, then returns 10**(data*0.1).

   Data below -400 dB is set to -400 dB before averaging.
   """
   data = np.asarray(data)
   
   if np.iscomplexobj(data):
        raise ValueError("PSD data must be real")

   if np.any(data < 0):
        raise ValueError("PSD data must be nonnegative")


   return 10**(np.mean(10*np.log10(np.maximum(data,1e-40)),axis=axis)*0.1)


def circular_mean(data, axis):
    """Mean direction of angles in radians; return angles in [-pi, pi].

    Handles -179,179 degree boundary, mean would be 180 degrees (inputs and outputs are radians)
    """
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ValueError("circular_mean expects real phase angles in radians")
    if data.shape[axis] == 0:
        raise ValueError("circular_mean requires at least one angle")
    
    # Unit phasors avoid averaging across the artificial -pi/pi boundary.
    return np.angle(np.mean(np.exp(1j * data), axis=axis))


def circular_median(data, axis):
    """Return a circular median by selecting an observed angle that minimizes the sum of the
    shortest angular distances.

    Can't use np.angle(np.median(np.exp(1j * data), axis=axis)) since Numpy orders
    complex numbers by real part and then imaginary part so ordering around a circle
    is not preserved.

    Input/output are radians; output is wrapped to [-pi, pi]. If several
    samples minimize the distance, choose the first (not their midpoint).
    Nonfinite values propagate as NaN within their slice. The search costs
    O(n**2) time per slice, but uses O(n) temporary memory.
    """
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ValueError("circular_median expects real phase angles in radians")
    if data.shape[axis] == 0:
        raise ValueError("circular_median requires at least one angle")

    def median_1d(angles):
        """Score observed candidate directions for one reduction slice."""
        if not np.all(np.isfinite(angles)):
            return np.nan
        # Normalize equivalent angles before comparing shortest arc distances.
        angles = np.angle(np.exp(1j * angles))
        scores = np.empty(angles.size)
        for i, candidate in enumerate(angles):
            # angle(exp(i*delta)) gives the signed shortest angular difference.
            # Score one candidate at a time to avoid an n-by-n distance matrix.
            distances = np.abs(np.angle(np.exp(1j * (angles - candidate))))
            scores[i] = np.sum(distances)
        # Treat numerical roundoff in equal distance sums as a tie.
        tied = np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-12)
        return angles[np.flatnonzero(tied)[0]]

    # Apply independently to every combination of the remaining dimensions.
    return np.apply_along_axis(median_1d, axis, data)


# Keep supported names and implementations together. Store callable objects,
# not calls; custom functions must be defined before this dictionary is built.
_COLLAPSE_FUNCTIONS = {None: first_slice,"mean": np.mean,"median": np.median,"min": np.min,"max": np.max,
    "mean_pwr_dB": mean_pwr_dB,"mean_amp_dB": mean_amp_dB,"circular_mean": circular_mean,
    "circular_median": circular_median}

def validate_collapse_func(collapse_func):
    """Accept an exact registry key (including None), otherwise raise ValueError.

    The supported input is a string or None. This validates names only, not
    array shape, physical units, or suitability for a plotting component.
    """
    if collapse_func not in _COLLAPSE_FUNCTIONS:
        choices = ", ".join(repr(name) for name in _COLLAPSE_FUNCTIONS)
        raise ValueError(f"collapse_func must be one of {choices}; got {collapse_func!r}")
    
def collapse_data(data, dim, collapse_func):
    """Reduce data along dim using a registered function and return its result.

    dim is a NumPy axis number, including supported negative axis indices.
    collapse_func is an exact registry name or None for first_slice. The
    output has one fewer dimension, with other singleton axes preserved.
    Units follow the selected reducer.

    Raises ValueError if the input contains NaN or +/-Inf, or if the reducer
    produces any of these values from finite input. Nonfinite values are
    rejected, not skipped, including when collapse_func is None.

    Example: collapse_data(data_QJT, 2, "mean") returns an array in QJ order.
    """
    validate_collapse_func(collapse_func)

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Cannot collapse data containing NaN or +/-Inf. "
            "Check the upstream processing.")

    function = _COLLAPSE_FUNCTIONS[collapse_func]
    # Public API uses dim; every registry callable receives the axis keyword.
    result = function(data, axis=dim)

    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"Collapse function {collapse_func!r} produced NaN or +/-Inf "
            "from finite input.")

    return result
