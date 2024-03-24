"""Backends for simulating circuits."""

from .provider import IonQProvider


class IonQBackend:
    def __init__(self, provider: IonQProvider, backend: str, gateset: str):
        self.provider = provider
        self.provider.backend = backend
        self.provider.gateset = gateset

    def run(self, circuit: dict, shots: int = 1024):
        return self.provider.run(circuit, shots=shots)


class IonQSimulator(IonQBackend):
    def __init__(
        self,
        provider: IonQProvider,
        gateset: str = "qis",
    ):
        super().__init__(provider, "simulator", gateset)


class IonQQPU(IonQBackend):
    def __init__(
        self,
        provider: IonQProvider,
        qpu: str = "harmony",
        gateset: str = "qis",
    ):
        qpu_name = ".".join(("qpu", qpu)).lower()
        super().__init__(provider, qpu_name, gateset)
