"""QuTiP family package entry point."""

from . import __version__


def version():
    """Return information to include in qutip.about()."""
    return "qutip-qip", __version__
