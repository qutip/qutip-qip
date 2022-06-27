from qiskit.providers.provider import ProviderV1 as QiskitProvider
from .backend import TestSimulator

SIMULATORS = [TestSimulator]


class Provider(QiskitProvider):

    def __init__(self):
        super().__init__()

        self.name = "qutip_provider"
        self._backends = {"test_backend": TestSimulator()}

    def backends(self, name=None, filters=None, **kwargs):
        backends = list(self._backends.values())
        if name:
            try:
                backends = [self._backends[name]]
            except LookupError:
                print("The '{}' backend is not installed in your system.".format(name))

        return backends
