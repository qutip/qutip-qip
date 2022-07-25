from email.header import Header
from qutip import tensor, basis
import numpy as np
import uuid
import random
from qutip_qip import circuit

from qutip_qip.operations.gateclass import Z
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit.circuitsimulator import CircuitResult

from .job import Job
from .converter import convert_qiskit_circuit
from qiskit.providers import BackendV1, Options
from qiskit.providers.models import (
    BackendConfiguration,
    QasmBackendConfiguration,
)
from qiskit.result import Result, Counts
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit
from qiskit.qobj import QobjExperimentHeader


class QiskitSimulatorBase(BackendV1):
    """
    The base class for qutip_qip based qiskit backends.
    """

    def __init__(self, configuration=None, provider=None, **fields):
        pass

    def run(self, qiskit_circuit: QuantumCircuit, **backend_options) -> Job:
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        qiskit_circuit : QuantumCircuit
            The qiskit circuit to be simulated.

        Result
        ------
        qutip_qip.qiskit.job.Job
            Job that stores results and execution data
        """
        qutip_circ = convert_qiskit_circuit(qiskit_circuit)
        job_id = str(uuid.uuid4())
        job = Job(
            backend=self,
            job_id=job_id,
            result=self._run_job(job_id, qutip_circ),
            warnings=qutip_circ._warnings if hasattr(
                qutip_circ, "_warnings") else []
        )
        return job


class QiskitCircuitSimulator(QiskitSimulatorBase):
    """
    Qiskit backend dealing with operator-level
    circuit simulation using qutip_qip's CircuitSimulator.


    """

    MAX_QUBITS_MEMORY = 10
    _configuration = {
        "backend_name": "circuit_simulator",
        "backend_version": "0.1",
        "n_qubits": MAX_QUBITS_MEMORY,
        "url": "https://github.com/qutip/qutip-qip",
        "simulator": True,
        "local": True,
        "conditional": False,
        "open_pulse": False,
        "memory": False,
        "max_shots": 1,
        "coupling_map": None,
        "description": "A qutip-qip based operator-level circuit simulator.",
        "basis_gates": [],
        "gates": [],
    }

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(
            configuration=(
                configuration
                or BackendConfiguration.from_dict(self._configuration)
            ),
            provider=provider,
            **fields
        )

    def _parse_results(
        self,
        statistics: CircuitResult,
        job_id: str,
        qutip_circuit: QubitCircuit,
    ) -> Result:
        """
        Returns a parsed object of type qiskit.result.Result
        for the CircuitSimulator

        Parameters
        ----------
        statistics : qutip_qip.circuit.
                    circuitsimulator.CircuitResult
            The result obtained from `run_statistics` on
            a circuit on CircuitSimulator

        job_id : str
            Unique ID identifying a job.

        qutip_circuit : QubitCircuit
            The circuit being simulated

        Returns
        -------
        qiskit.result.Result
            Result of the simulation
        """
        counts = {}

        def convert_to_hex(count):
            return hex(int("".join(str(i) for i in count), 2))

        if statistics.cbits[0] is not None:
            for i, count in enumerate(statistics.cbits):
                counts[convert_to_hex(count)] = statistics.probabilities[i]
            counts = Counts(counts)
        else:
            counts = None

        statevector = random.choices(
            statistics.final_states, weights=statistics.probabilities
        )[0]

        exp_res_data = ExperimentResultData(
            counts=counts, statevector=Statevector(data=np.array(statevector))
        )

        header = QobjExperimentHeader.from_dict({
            "name": qutip_circuit.name if hasattr(
                qutip_circuit, "name") else "",
            "n_qubits": qutip_circuit.N,
        })

        exp_res = ExperimentResult(shots=1, success=True, data=exp_res_data,
                                   header=header)

        result = Result(
            backend_name=self._configuration["backend_name"],
            backend_version=self._configuration["backend_version"],
            qobj_id=id(qutip_circuit),
            job_id=job_id,
            success=True,
            results=[exp_res],
        )

        return result

    def _run_job(self, job_id: str, qutip_circuit: QubitCircuit) -> Result:
        """
        Run a QubitCircuit on the CircuitSimulator.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.

        qutip_circuit : QubitCircuit
            The circuit obtained after conversion
            from QuantumCircuit to QubitCircuit.

        Returns
        -------
        qiskit.result.Result
            Result of the simulation
        """
        zero_state = basis([2]*qutip_circuit.N, [0]*qutip_circuit.N)
        statistics = qutip_circuit.run_statistics(state=zero_state)

        return self._parse_results(
            statistics=statistics, job_id=job_id, qutip_circuit=qutip_circuit
        )

    @classmethod
    def _default_options(cls):
        """
        Default options for the backend. To be updated.
        """
        return Options()
