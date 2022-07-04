class Job:
    """
    Stores information about a qiskit job.

    Parameters
    ----------
    backend : Union[QiskitCircuitSimulator]
        The backend used to simulate a circuit in the job.

    job_id : str
        Unique ID identifying a job.

    result : -
        The result of a simulation run. 
    """

    def __init__(self, backend, job_id, result):
        # super().__init__(backend, job_id)
        self._result = result

    def result(self):
        return self._result
