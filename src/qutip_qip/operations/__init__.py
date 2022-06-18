"""
Operations on quantum circuits.
"""
import warnings
from .gates import *
from .gateclass import *
from .measurement import *


def _add_deprecation(fun, msg):
    def newfun(*args, **kwargs):
        warnings.warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        return fun(*args, **kwargs)

    return newfun
