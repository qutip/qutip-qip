"""Class for a running job."""

from typing import List

from qiskit import QuantumCircuit
from qiskit.providers import JobV1, JobStatus
from qiskit.providers.backend import Backend
from qiskit.result import Result


class Job(JobV1):
    """
    Stores information about an ongoing job.

    Parameters
    ----------
    backend : :class:`.QiskitCircuitSimulator`
        The backend used to simulate a
        circuit in the job.

    job_id : str
        Unique ID identifying a job.

    result : :class:`qiskit.result.Result`
        The result of a simulation run.
    """

    def __init__(
        self, backend: Backend, job_id: str, circuit: list[QuantumCircuit]
    ):
        super().__init__(backend, job_id)
        self._run_input = circuit
        self._status = JobStatus.INITIALIZING

    def submit(self):
        """Submit the job to the backend for execution."""
        self._status = JobStatus.QUEUED
        self._result = self.backend()._run_job(self.job_id, self._run_input)
        self._status = JobStatus.DONE
        return

    def status(self) -> JobStatus:
        """Returns job status"""
        return self._status

    def cancel(self):
        pass

    def result(self) -> Result:
        """Return the job's result"""
        return self._result
