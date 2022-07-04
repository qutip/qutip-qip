from qiskit.providers.provider import ProviderV1
from .backend import QiskitCircuitSimulator


class Provider(ProviderV1):
    """
    Provides access to qutip_qip based qiskit backends. 
    """

    def __init__(self):
        super().__init__()

        self.name = "qutip_provider"
        self._backends = {"circuit_simulator": QiskitCircuitSimulator()}

    def backends(self, name=None, filters=None, **kwargs):
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
