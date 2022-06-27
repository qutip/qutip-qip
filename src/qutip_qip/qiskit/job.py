
class Job():

    def __init__(self, backend, job_id, result):
        # super().__init__(backend, job_id)
        self._result = result

    def result(self):
        return self._result
