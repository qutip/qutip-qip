import random
from typing import List, Union

from qutip import basis
from qutip_qip.circuit import CircuitResult, QubitCircuit
from qutip_qip.qiskit import QiskitSimulatorBase
from qutip_qip.qiskit.converter import convert_qiskit_circuit_to_qutip
from qutip_qip.qiskit.utils import QUTIP_TO_QISKIT_MAP

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Measure
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector
from qiskit.transpiler.target import Target

class QiskitCircuitSimulator(QiskitSimulatorBase):
    """
    ``qiskit`` backend dealing with operator-level
    circuit simulation using ``qutip_qip``'s :class:`.CircuitSimulator`.

    Parameters
    ----------
    name : str
        Name of the backend circuit simulator.
    """

    NAME = "circuit_simulator"
    DESCRIPTION = "A qutip-qip based operator-level circuit simulator."
    VERSION = 0.1
    URL = "https://github.com/qutip/qutip-qip"
    MAX_SHOTS = 1e6
    MAX_CIRCUITS = 1

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        num_qubits: int = 10,
        basis_gates: List[str] = None,
        max_shots: int = MAX_SHOTS,
        max_circuits: int = MAX_CIRCUITS,
    ):
        super().__init__(
            name = name,
            description = description,
        )

        self._target = self._build_target(
            num_qubits=num_qubits,
            basis_gates=basis_gates
        )

        self.max_shots = max_shots
        self.max_circuits = max_circuits


    @property
    def max_shots(self) -> int:
        """The maximum number of shots that can be used by the sampler."""
        return self._max_shots
    
    @max_shots.setter
    def max_shots(self, value) -> None:
        """Python Setter function for the max_shots property"""
        self._max_shots = value

    @property
    def max_circuits(self) -> Union[int | None]:
        """The maximum number of circuits that can be
        run in a single job.

        If there is no limit this will return None."""
        return self._max_circuits
    
    @max_circuits.setter
    def max_circuits(self, value) -> None:
        """Python Setter function for the max_circuits property"""
        self._max_circuits = value

    @property
    def target(self) -> Target:
        return self._target

    def _build_target(self, num_qubits:int = 10, basis_gates=None) -> Target:
        """Builds a :class:`qiskit.transpiler.Target` object for the backend.
        
        :rtype: Target
        """

        target = Target(num_qubits=num_qubits)
        if basis_gates is None:
            basis_gates = list(QUTIP_TO_QISKIT_MAP.keys())

        # Adding the basis gates
        # Passing properties=None means "This gate works on ALL qubits with NO error"
        for gate in basis_gates:
            target.add_instruction(QUTIP_TO_QISKIT_MAP[gate], properties=None)

        # Essential primitives
        target.add_instruction(Measure(), properties=None)
        
        # TODO: Add Barrier implementation to QuTiP.
        #target.add_instruction(Barrier(), properties=None)
        return target


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
            counts=counts,
            statevector=Statevector(data=statevector.full())
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
            backend_name=self.name,
            backend_version=self.VERSION,
            qobj_id=id(qutip_circuit),
            job_id=job_id,
            success=True,
            results=[exp_res],
        )

        return result
