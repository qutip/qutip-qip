"""Provider for the IonQ backends."""

from .converter import convert_qutip_circuit
from .job import Job
from ..version import version as __version__
from os import getenv


class IonQProvider:
    """
    Provides access to qutip_qip based IonQ backends.

    Attributes
    ----------
    name: str
        Name of the provider
    """

    def __init__(
        self,
        token: str = None,
        url: str = "https://api.ionq.co/v0.3",
    ):
        token = token or getenv("IONQ_API_KEY")
        if not token:
            raise ValueError("No token provided")
        self.headers = self.create_headers(token)
        self.url = url
        self.backend = None

    def run(self, circuit, shots: int = 1024) -> Job:
        """
        Run a circuit.

        Parameters
        ----------
        circuit: QubitCircuit
            The circuit to be run.
        shots: int
            The number of shots.

        Returns
        -------
        Job
            The running job.
        """
        ionq_circuit = convert_qutip_circuit(circuit)
        job = Job(
            ionq_circuit,
            shots,
            self.backend,
            self.gateset,
            self.headers,
            self.url,
        )
        job.submit()
        return job

    def create_headers(self, token: str):
        return {
            "Authorization": f"apiKey {token}",
            "Content-Type": "application/json",
            "User-Agent": f"qutip-qip/{__version__}",
        }
