import random

from qutip import basis
from qutip_qip.circuit import CircuitResult, QubitCircuit
from qutip_qip.qiskit import QiskitSimulatorBase

import qiskit
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector

class QiskitCircuitSimulator(QiskitSimulatorBase):
    """
    ``qiskit`` backend dealing with operator-level
    circuit simulation using ``qutip_qip``'s :class:`.CircuitSimulator`.

    Parameters
    ----------
    configuration : dict
        Configurable attributes of the backend.
    """

    MAX_QUBITS_MEMORY = 10
    BACKEND_NAME = "circuit_simulator"
    _DEFAULT_CONFIGURATION = {
        "backend_name": BACKEND_NAME,
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

    def __init__(self, configuration: dict = None, **fields):
        self.configuration = configuration
        if configuration is None:
            self.configuration = self._DEFAULT_CONFIGURATION

        super().__init__(
            name = configuration.backend_name,
            description = configuration.description,
        )

    def target(self):
        pass

    def max_circuits(self):
        pass

    def _parse_results(
        self,
        job_id: str,
        qutip_circuit: QubitCircuit,
        statistics: CircuitResult,
    ) -> qiskit.result.Result:
        """
        Returns a parsed object of type :class:`qiskit.result.Result`
        from the results of simulation.

        Parameters
        ----------
        statistics : :class:`.CircuitResult`
            The result obtained from ``run_statistics`` on
            a circuit on :class:`.CircuitSimulator`.

        job_id : str
            Unique ID identifying a job.

        qutip_circuit : :class:`.QubitCircuit`
            The circuit being simulated

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the simulation.
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
            counts=counts, statevector=Statevector(data=statevector.full())
        )

        header = {
            "name": (
                qutip_circuit.name
                if hasattr(qutip_circuit, "name")
                else ""
            ),
            "n_qubits": qutip_circuit.N,
        }

        exp_res = ExperimentResult(
            shots=self._options.shots,
            success=True,
            data=exp_res_data,
            header=header,
        )

        result = Result(
            backend_name=self.BACKEND_NAME,
            backend_version=0.1,
            qobj_id=id(qutip_circuit),
            job_id=job_id,
            success=True,
            results=[exp_res],
        )

        return result

    def _run_job(self, job_id: str, qutip_circuit: QubitCircuit) -> Result:
        """
        Run a :class:`.QubitCircuit` on the :class:`.CircuitSimulator`.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.

        qutip_circuit : :class:`.QubitCircuit`
            The circuit obtained after conversion
            from :class:`qiskit.circuit.QuantumCircuit`
            to :class:`.QubitCircuit`.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the simulation
        """
        zero_state = basis([2] * qutip_circuit.N, [0] * qutip_circuit.N)
        statistics = qutip_circuit.run_statistics(state=zero_state)

        return self._parse_results(
            statistics=statistics, job_id=job_id, qutip_circuit=qutip_circuit
        )
