from qutip import tensor, basis
import numpy as np
import uuid
from .job import Job
from .converter import qiskit_to_qutip


class QiskitSimulatorBase:
    """
    The base class for qutip_qip based qiskit backends.
    """

    def __init__(self, configuration=None, provider=None, **fields):
        pass

    def run(self, circuit, **backend_options):
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        circuit : QuantumCircuit
            The qiskit circuit to be simulated.
        """
        qutip_circ = qiskit_to_qutip(circuit)
        job_id = str(uuid.uuid4())
        job = Job(
            backend=self,
            job_id=job_id,
            result=self._run_job(job_id, qutip_circ),
        )
        return job


class QiskitCircuitSimulator(QiskitSimulatorBase):
    """
    Qiskit backend dealing with operator-level
    circuit simulation using qutip_qip's CircuitSimulator.
    """

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(configuration=None, provider=None, **fields)

    def _run_job(self, job_id, circuit):
        """
        Run a QubitCircuit on the CircuitSimulator.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.

        circuit : QubitCircuit
            The circuit obtained after conversion
            from QuantumCircuit to QubitCircuit. 
        """
        zero_state = basis(2, 0)
        for i in range(circuit.N - 1):
            zero_state = tensor(zero_state, basis(2, 0))
        result = circuit.run(state=zero_state)
        return result
