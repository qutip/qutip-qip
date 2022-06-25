"""
Command line output of information on QuTiP QIP and dependencies.
"""

import qutip
import qutip_qip
import os
import inspect


def _about():
    """
    Meant to be called from qutip's about function
    through qutip-qip's entrypoint.
    """
    title = "QuTiP QIP: QuTiP Quantum Information Processing"
    lines = []
    lines.append("=" * len(title))
    lines.append("QuTiP QIP Version: %s" % qutip_qip.__version__)
    qutip_qip_install_path = os.path.dirname(inspect.getsourcefile(qutip_qip))
    lines.append("Installation path: %s" % qutip_qip_install_path)
    return title, lines


def about():
    """About box for QuTiP QIP."""
    qutip.about()
