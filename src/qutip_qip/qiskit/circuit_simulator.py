import random
import uuid
from typing import List

from qutip import basis
from qutip_qip.circuit import CircuitResult, QubitCircuit
from qutip_qip.qiskit import Job, QiskitSimulatorBase
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

    DESCRIPTION = "A qutip-qip based operator-level circuit simulator."
    VERSION = 0.1
    URL = "https://github.com/qutip/qutip-qip"
    MAX_CIRCUITS = 1

    def __init__(
        self,
        name: str = "circuit_simulator",
        description: str = DESCRIPTION,
        num_qubits: int = 10,
        basis_gates: List[str] = None,
        max_shots: int = 1e6,
        max_circuits: int = 1,
    ):
        super().__init__(
            name=name,
            description=description,
            num_qubits=num_qubits,
            basis_gates=basis_gates,
            max_shots=max_shots,
            max_circuits=max_circuits,
        )

    @property
    def meas_map(self) -> list[list[int]]:
        """Simulator allows measuring any qubit independently"""
        return [[i] for i in range(self.target.num_qubits)]

    def run(self, run_input: List[QuantumCircuit], **run_options) -> Job:
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        run_input : List[:class:`qiskit.circuit.QuantumCircuit`]
            List of ``qiskit`` circuits to be simulated.

        **run_options:
            Additional run options for the backend.

            Valid options are:

            shots : int
                Number of times to sample the results.

        Returns
        -------
        :class:`.Job`
            Job object that stores results and execution data.
        """
        # Set the no. of shots
        if "shots" in run_options:
            shots = run_options["shots"]
            if shots <= 0:
                raise ValueError(f"shots ${shots} must be a positive number")

            self.set_options(shots=shots)

        job_id = str(uuid.uuid4())
        job = Job(
            backend=self,
            job_id=job_id,
            circuit=run_input,
        )
        job.submit()
        return job

    def _run_job(self, job_id: str, qiskit_circuit: QuantumCircuit) -> Result:
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
        transpiled_circuit = transpile(qiskit_circuit, backend=self)
        qutip_circuit = convert_qiskit_circuit_to_qutip(transpiled_circuit)

        zero_state = basis([2] * qutip_circuit.N, [0] * qutip_circuit.N)
        statistics = qutip_circuit.run_statistics(state=zero_state)

        return self._parse_results(
            statistics=statistics, job_id=job_id, qutip_circuit=qutip_circuit
        )

    def _parse_results(
        self,
        job_id: str,
        qutip_circuit: QubitCircuit,
        statistics: CircuitResult,
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
                qutip_circuit.name if hasattr(qutip_circuit, "name") else ""
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
            backend_name=self.name,
            backend_version=self.VERSION,
            job_id=job_id,
            success=True,
            results=[exp_res],
        )

        return result
