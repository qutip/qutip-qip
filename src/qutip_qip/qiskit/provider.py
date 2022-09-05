"""Provider for the simulator backends."""

from qiskit.providers.provider import ProviderV1
from .backend import QiskitCircuitSimulator, QiskitPulseSimulator


class Provider(ProviderV1):
    """
    Provides access to qutip_qip based qiskit backends.

    Attributes
    ----------
    name: str
        Name of the provider
    """

    def __init__(self):
        super().__init__()

        self.name = "qutip_provider"
        self._backends = {
            QiskitCircuitSimulator.BACKEND_NAME: QiskitCircuitSimulator(),
            QiskitPulseSimulator.BACKEND_NAME: QiskitPulseSimulator(),
        }

    def backends(self, name: str = None, filters=None, **kwargs) -> list:
        """
        Returns the available backends

        Parameters
        ----------
        name: str
            Name of required backend

        Returns
        -------
        list
            List of available backends
        """
        backends = list(self._backends.values())
        if name:
            try:
                backends = [self._backends[name]]
            except LookupError:
                print(
                    "The '{}' backend is not installed in your system.".format(
                        name
                    )
                )

        return backends
