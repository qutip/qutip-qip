import numpy as np
from scipy.interpolate import CubicSpline

def fill_coeff(old_coeffs, old_tlist, full_tlist, args=None, tol=1.0e-10):
    """
    Make a step function coefficients compatible with a longer ``tlist`` by
    filling the empty slot with the nearest left value.
    The returned ``coeff`` always have the same size as the ``tlist``.
    If `step_func`, the last element is 0.
    """
    if args is None:
        args = {}

    if "_step_func_coeff" in args and args["_step_func_coeff"]:
        if len(old_coeffs) == len(old_tlist) - 1:
            old_coeffs = np.concatenate([old_coeffs, [0]])

        new_n = len(full_tlist)
        old_ind = 0  # index for old coeffs and tlist
        new_coeff = np.zeros(new_n)

        for new_ind in range(new_n):
            t = full_tlist[new_ind]

            if old_tlist[0] - t > tol:
                new_coeff[new_ind] = 0.0
                continue

            if t - old_tlist[-1] > tol:
                new_coeff[new_ind] = 0.0
                continue

            # tol is required because of the floating-point error
            if old_tlist[old_ind + 1] <= t + tol:
                old_ind += 1
            new_coeff[new_ind] = old_coeffs[old_ind]

    else:
        sp = CubicSpline(old_tlist, old_coeffs)
        new_coeff = sp(full_tlist)
        new_coeff *= full_tlist <= old_tlist[-1]
        new_coeff *= full_tlist >= old_tlist[0]
        
    return new_coeff
