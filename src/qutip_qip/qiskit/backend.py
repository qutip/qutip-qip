from email.header import Header
from qutip import tensor, basis
import numpy as np
import uuid
import random
from collections import Counter
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
        super().__init__(configuration=configuration, provider=provider)
        self.options.set_validator(
            "shots", (1, self.configuration().max_shots)
        )

    def run(self, qiskit_circuit: QuantumCircuit, **run_options) -> Job:
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        qiskit_circuit : QuantumCircuit
            The qiskit circuit to be simulated.

        **run_options:
            Additional run options for the backend.

            Valid options are:

            shots : int
                Number of times to perform the simulation
            allow_custom_gate: bool
                Allow conversion of circuit using unitary matrices
                for custom gates.

        Result
        ------
        qutip_qip.qiskit.job.Job
            Job that stores results and execution data
        """
        # configure the options
        self.set_options(
            shots=run_options["shots"]
            if "shots" in run_options
            else self._default_options().shots,
            allow_custom_gate=run_options["allow_custom_gate"]
            if "allow_custom_gate" in run_options
            else self._default_options().allow_custom_gate,
        )
        qutip_circ = convert_qiskit_circuit(
            qiskit_circuit, allow_custom_gate=self.options.allow_custom_gate
        )
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

    MAX_QUBITS_MEMORY = 10
    _DEFAULT_CONFIGURATION = {
        "backend_name": "circuit_simulator",
        "backend_version": "0.1",
        "n_qubits": MAX_QUBITS_MEMORY,
        "url": "https://github.com/qutip/qutip-qip",
        "simulator": True,
        "local": True,
        "conditional": False,
        "open_pulse": False,
        "memory": False,
        "max_shots": int(1e6),
        "coupling_map": None,
        "description": "A qutip-qip based operator-level circuit simulator.",
        "basis_gates": [],
        "gates": [],
    }

    def __init__(self, configuration=None, provider=None, **fields):

        if configuration is None:
            configuration = QasmBackendConfiguration.from_dict(
                QiskitCircuitSimulator._DEFAULT_CONFIGURATION
            )

        super().__init__(
            configuration=configuration, provider=provider, **fields
        )

    def _sample_shots(self, count_probs: dict) -> Counts:
        """
        Sample measurements from a given probability distribution

        Parameters
        ----------
        count_probs: dict
            Probability distribution corresponding
            to different classical outputs.

        Returns
        -------
        qiskit.result.Counts
            Returns the Counts object sampled according to
            the given probabilities and configured shots.
        """
        shots = self.options.shots
        samples = random.choices(
            list(count_probs.keys()), list(count_probs.values()), k=shots
        )
        return Counts(Counter(samples))

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
        count_probs = {}
        counts = None

        def convert_to_hex(count):
            return hex(int("".join(str(i) for i in count), 2))

        if statistics.cbits[0] is not None:
            for i, count in enumerate(statistics.cbits):
                count_probs[convert_to_hex(count)] = statistics.probabilities[
                    i
                ]
            # sample the shots from obtained probabilities
            counts = self._sample_shots(count_probs)

        statevector = random.choices(
            statistics.final_states, weights=statistics.probabilities
        )[0]

        exp_res_data = ExperimentResultData(
            counts=counts, statevector=Statevector(data=np.array(statevector))
        )

        header = QobjExperimentHeader.from_dict(
            {
                "name": qutip_circuit.name
                if hasattr(qutip_circuit, "name")
                else "",
                "n_qubits": qutip_circuit.N,
            }
        )

        exp_res = ExperimentResult(
            shots=self.options.shots,
            success=True,
            data=exp_res_data,
            header=header,
        )

        result = Result(
            backend_name=self.configuration().backend_name,
            backend_version=self.configuration().backend_version,
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
        zero_state = basis([2] * qutip_circuit.N, [0] * qutip_circuit.N)
        statistics = qutip_circuit.run_statistics(state=zero_state)

        return self._parse_results(
            statistics=statistics, job_id=job_id, qutip_circuit=qutip_circuit
        )

    @classmethod
    def _default_options(cls):
        """
        Default options for the backend. To be updated.
        """
        return Options(shots=1024, allow_custom_gate=True)
