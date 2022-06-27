from .job import Job
from qiskit import qobj


class TestSimulator():

    def __init__(self, configuration=None, provider=None, **fields):
        pass
        # super().__init__(configuration=configuration, provider=provider, **fields)

    def run(self, qobj: qobj.Qobj, **backend_options):
        job_id = 123
        job = Job(self, job_id, self._run_job(job_id, qobj))
        return job

    def _run_job(self, job_id, qobj):
        test_ret = {
            "success": True,
            "data": [1, 0, 0]
        }
        return test_ret
