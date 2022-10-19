"""Class for a running job."""

from qiskit.providers import JobV1, JobStatus
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

    def __init__(self, backend, job_id: str, result: Result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        """Submit the job to the backend for execution."""
        return

    def status(self):
        """Returns job status"""
        return JobStatus.DONE

    def result(self):
        """Return the job's result"""
        return self._result
