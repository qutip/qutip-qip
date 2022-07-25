from qiskit.providers import JobV1, JobStatus
from qiskit.result import Result


class Job(JobV1):
    """
    Stores information about a qiskit job.

    Parameters
    ----------
    backend : QiskitCircuitSimulator
        The backend used to simulate a
        circuit in the job.

    job_id : str
        Unique ID identifying a job.

    result : qiskit.result.Result
        The result of a simulation run.
    """

    def __init__(self, backend, job_id: str, result: Result, warnings=[]):
        super().__init__(backend, job_id)
        self._result = result
        self._warnings = warnings

    def submit(self):
        """Submit the job to the backend for execution."""
        return

    def status(self):
        """Returns job status"""
        return JobStatus.DONE

    def result(self):
        """Return the job's result"""
        return self._result

    def warnings(self):
        """Display the warnings encountered while execution"""
        if not self._warnings:
            print("Executed with no warnings.")
        else:
            for warning in self._warnings:
                print(warning)
