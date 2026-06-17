"""
For the compatibility between qutip-v5 and v4.
"""

from packaging.version import parse as parse_version
import qutip


is_qutip5 = parse_version(qutip.__version__) >= parse_version("5.dev")


def to_scalar(qobj_or_scalar):
    if isinstance(qobj_or_scalar, qutip.Qobj):
        if qobj_or_scalar.dims == [[1], [1]]:
            return qobj_or_scalar[0, 0]
    return qobj_or_scalar


def solver_options(**kwargs):
    if is_qutip5:
        return kwargs
    return qutip.Options(**kwargs)
