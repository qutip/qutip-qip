"""Class for a running job."""

from .converter import create_job_body, convert_ionq_response_to_circuitresult
from qutip_qip.circuit import CircuitResult
import requests
import time


class Job:
    """
    Class for a running job.

    Attributes
    ----------
    body: dict
        The body of the job request.
    """

    def __init__(
        self,
        circuit: dict,
        shots: int,
        backend: str,
        gateset: str,
        headers: dict,
        url: str,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.backend = backend
        self.gateset = gateset
        self.headers = headers
        self.url = url
        self.id = None
        self.results = None

    def submit(self) -> None:
        """
        Submit the job.
        """
        json = create_job_body(
            self.circuit,
            self.shots,
            self.backend,
            self.gateset,
        )
        response = requests.post(
            f"{self.url}/jobs",
            json=json,
            headers=self.headers,
        )
        response.raise_for_status()
        self.id = response.json()["id"]

    def get_status(self) -> dict:
        """
        Get the status of the job.

        Returns
        -------
        dict
            The status of the job.
        """
        response = requests.get(
            f"{self.url}/jobs/{self.id}",
            headers=self.headers,
        )
        response.raise_for_status()
        self.status = response.json()
        return self.status

    def get_results(self, polling_rate: int = 1) -> CircuitResult:
        """
        Get the results of the job.

        Returns
        -------
        dict
            The results of the job.
        """
        while self.get_status()["status"] not in (
            "canceled",
            "completed",
            "failed",
        ):
            time.sleep(polling_rate)
        response = requests.get(
            f"{self.url}/jobs/{self.id}/results",
            headers=self.headers,
        )
        response.raise_for_status()
        return convert_ionq_response_to_circuitresult(response.json())
