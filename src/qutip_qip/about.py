"""
Command line output of information on QuTiP QIP and dependencies.
"""
__all__ = ["about"]

import qutip


def about():
    """
    About box for QuTiP QIP. Gives version numbers for QuTiP, QuTiP QIP, NumPy,
    SciPy, Cython, and MatPlotLib.
    """
    qutip.about("qutip_qip")


if __name__ == "__main__":
    about()
