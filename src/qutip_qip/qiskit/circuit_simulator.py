import random
from typing import List

from qutip import basis
from qutip_qip.circuit import CircuitResult, QubitCircuit
from qutip_qip.qiskit import QiskitSimulatorBase
from qutip_qip.qiskit.converter import convert_qiskit_circuit_to_qutip

from qiskit import QuantumCircuit, transpile
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector


class QiskitCircuitSimulator(QiskitSimulatorBase):
    """
    ``qiskit`` backend dealing with operator-level
    circuit simulation using ``qutip_qip``'s :class:`.CircuitSimulator`.

    Parameters
    ----------
    name : str
        Name of the backend circuit simulator.
    """

    def __init__(
        self,
        name: str = "circuit_simulator",
        description: str = "A qutip-qip based operator-level circuit simulator.",
        version: str = "0.1",
        num_qubits: int = 10,
        basis_gates: List[str] = None,
        max_shots: int = 1e6,
        max_circuits: int = 1,
    ):
        super().__init__(
            num_qubits=num_qubits,
            basis_gates=basis_gates,
            max_shots=max_shots,
            max_circuits=max_circuits,
            name=name,
            description=description,
            version=version,
        )

    @property
    def meas_map(self) -> list[list[int]]:
        """Simulator allows measuring any qubit independently"""
        return [[i] for i in range(self.target.num_qubits)]

    def _run_job(
        self, job_id: str, qiskit_circuit: list[QuantumCircuit]
    ) -> Result:
        """
        Run a :class:`.QubitCircuit` on the :class:`.CircuitSimulator`.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.

        qiskit_circuit : :class:`qiskit.QuantumCircuit`
            The qiskits circuit to be simulated.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the simulation
        """
        qutip_circuits = []
        statistics = []

        for circuit in qiskit_circuit:
            if not isinstance(circuit, QuantumCircuit):
                raise ValueError(
                    "List must only consist of qiskit QuantumCircuit"
                )

            transpiled_circuit = transpile(circuit, backend=self)
            qutip_circuit = convert_qiskit_circuit_to_qutip(transpiled_circuit)
            qutip_circuits.append(qutip_circuit)

            zero_state = basis([2] * qutip_circuit.N, [0] * qutip_circuit.N)
            statistics.append(qutip_circuit.run_statistics(state=zero_state))

        return self._parse_results(
            statistics=statistics, job_id=job_id, qutip_circuits=qutip_circuits
        )

    def _parse_results(
        self,
        job_id: str,
        qutip_circuits: list[QubitCircuit],
        statistics: list[CircuitResult],
    ) -> Result:
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

        if len(statistics) != len(qutip_circuits):
            raise ValueError(
                "Number of statistics must be = to number of qutip circuits"
            )

        exp_res = []
        num_circuits = len(qutip_circuits)

        for i in range(num_circuits):
            count_probs = {}
            counts = None
            statistic = statistics[i]
            qutip_circuit = qutip_circuits[i]

            def convert_to_hex(count):
                return hex(int("".join(str(i) for i in count), 2))

            if statistic.cbits[0] is not None:
                for i, count in enumerate(statistic.cbits):
                    count_probs[convert_to_hex(count)] = (
                        statistic.probabilities[i]
                    )
                # sample the shots from obtained probabilities
                counts = self._sample_shots(count_probs)

            statevector = random.choices(
                statistic.final_states, weights=statistic.probabilities
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

            exp_res.append(
                ExperimentResult(
                    shots=self._options.shots,
                    success=True,
                    data=exp_res_data,
                    header=header,
                )
            )

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=exp_res,
        )
