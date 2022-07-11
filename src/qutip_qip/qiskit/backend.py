from sre_parse import State
from qutip import tensor, basis
import numpy as np
import uuid

from qutip_qip.operations.gateclass import Z
from .job import Job
from .converter import qiskit_to_qutip
from qiskit.providers import BackendV1, Options
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector


class QiskitSimulatorBase(BackendV1):
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

    _configuration = {
        "backend_name": "circuit_simulator",
        "backend_version": "0.1",
    }

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(configuration=configuration, provider=provider, **fields)

    def _parse_results(self, statevector, statistics, job_id, circuit):
        """
        Returns a parsed object of type qiskit.result.Result
        for the CircuitSimulator

        Parameters
        ----------
        statevector : qutip.qobj.Qobj
            The result obtained by running a circuit on CircuitSimulator

        statistics : qutip_qip.circuit.
                    circuitsimulator.CircuitResult
            The result obtained from `run_statistics` on
            a circuit on CircuitSimulator

        job_id : str
            Unique ID identifying a job.

        circuit : QubitCircuit
            The circuit being simulated
        """
        counts = {}

        def convert_to_hex(count):
            return hex(int("".join(str(i) for i in count), 2))

        if statistics.cbits[0] is not None:
            for i, count in enumerate(statistics.cbits):
                counts[convert_to_hex(count)] = round(
                    statistics.probabilities[i], 3)
        else:
            counts = None

        exp_res_data = ExperimentResultData(
            counts=counts,
            statevector=Statevector(data=np.array(statevector))
        )

        exp_res = ExperimentResult(
            shots=1,
            success=True,
            data=exp_res_data
        )

        result = Result(
            backend_name=self._configuration["backend_name"],
            backend_version=self._configuration["backend_version"],
            qobj_id=id(circuit),
            job_id=job_id,
            success=True,
            results=[exp_res],
        )

        return result

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

        statevector = circuit.run(state=zero_state)
        statistics = circuit.run_statistics(state=zero_state)

        return self._parse_results(statevector=statevector,
                                   statistics=statistics,
                                   job_id=job_id,
                                   circuit=circuit)

    @classmethod
    def _default_options(cls):
        """
        Default options for the backend. To be updated.
        """
        return Options()
