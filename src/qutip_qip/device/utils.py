from collections.abc import Iterable
import numbers
import numpy as np


def _to_array(params, num_qubits):
    """
    Transfer a parameter to an array.
    """
    if isinstance(params, numbers.Real):
        return np.asarray([params] * num_qubits)
    elif isinstance(params, Iterable):
        return np.asarray(params)


def _pulse_interpolate(pulse, tlist):
    """
    A function that calls Scipy interpolation routine. Used for plotting.
    """
    if pulse.tlist is None and pulse.coeff is None:
        coeff = np.zeros(len(tlist))
        return coeff
    if isinstance(pulse.coeff, bool):
        if pulse.coeff:
            coeff = np.ones(len(tlist))
        else:
            coeff = np.zeros(len(tlist))
        return coeff
    coeff = pulse.coeff
    if len(coeff) == len(pulse.tlist) - 1:  # for discrete pulse
        coeff = np.concatenate([coeff, [0]])

    from scipy import interpolate

    if pulse.spline_kind == "step_func":
        kind = "previous"
    else:
        kind = "cubic"
    inter = interpolate.interp1d(
        pulse.tlist, coeff, kind=kind, bounds_error=False, fill_value=0.0
    )
    return inter(tlist)
