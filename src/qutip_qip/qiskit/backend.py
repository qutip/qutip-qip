"""Backends for simulating qiskit circuits."""

from abc import abstractmethod
from typing import List
import uuid

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import Measure
from qiskit.providers import BackendV2, Options
from qiskit.result import Result
from qiskit.transpiler.target import Target

from qutip_qip.qiskit import Job
from qutip_qip.qiskit.utils import QUTIP_TO_QISKIT_MAP


class QiskitSimulatorBase(BackendV2):
    """
    The base class for ``qutip_qip`` based ``qiskit`` backends.
    This class must always be inherited, never instantiated as the 
    implementation of abstract method `_run_job` is left to the child class.
    
    Parameters
    ----------
    name : str
        Backend name.
    description : str
        Backend description.
    version : float
        Backend version
    num_qubits : int
        Number of qubits supported by the backend.
    basis_gates : list[str]
        QuTiP Basis Gates supported by the backend.
    max_shots : int
        Maximum number of sampling shots supported by the backend.
        Defaults to 1e6
    max_circuits : int
        Maximum number of circuits which can be passed to the
        backend in a single job. Defaults to 1.

    Notes
    -----
    Inherits all attributes from `BackendV2`.
    """

    def __init__(
        self,
        name: str = None,
        description: str = None,
        version: str = "0.1",
        num_qubits: int = 10,
        basis_gates: List[str] = None,
        max_shots: int = 1e6,
        max_circuits: int = 1,
    ):
        super().__init__(
            name=name,
            description=description,
            backend_version=version,
        )

        self._target = self._build_target(
            num_qubits=num_qubits, basis_gates=basis_gates
        )
        self.max_shots = max_shots
        self.max_circuits = max_circuits

    @property
    def max_shots(self) -> int:
        """The maximum number of shots that can be used by the sampler."""
        return self._max_shots

    @max_shots.setter
    def max_shots(self, shots: int) -> None:
        """Python Setter function for the max_shots property"""
        self._max_shots = shots

    @property
    def max_circuits(self) -> (int | None):
        """The maximum number of circuits that can be
        run in a single job.

        If there is no limit this will return None."""
        return self._max_circuits

    @max_circuits.setter
    def max_circuits(self, circuit_count: int) -> None:
        """Python Setter function for the max_circuits property"""
        self._max_circuits = circuit_count

    @property
    def target(self) -> Target:
        return self._target

    def _build_target(
        self, num_qubits: int = 10, basis_gates: list[str] = None
    ) -> Target:
        """
        Builds a :class:`qiskit.transpiler.Target` object for the backend.

        Parameters
        ----------
        num_qubits: int
            Number of qubits in the backend processor

        basis_gates: list[str]
            QuTiP Basis Quantum Gates supported by the backend.

        Returns
        -------
        :class:`qiskit.transpiler.Target`
            Target object that defines the configuration for the backend.
            `num_qubits`, `qubit_properties`, `basis_gates`,
            `concurrent_measurements` etc.
        """

        DEFAULT_BASIS_GATE_SET = QUTIP_TO_QISKIT_MAP.keys()
        if basis_gates is not None:
            for gate in basis_gates:
                if gate not in DEFAULT_BASIS_GATE_SET:
                    raise ValueError(
                        f"Invalid basis gate set, contains ${gate}"
                    )

        target = Target(num_qubits=num_qubits)
        if basis_gates is None:
            basis_gates = list(DEFAULT_BASIS_GATE_SET)

        # Adding the basis gates
        # Passing properties=None means "This gate works on ALL qubits with NO error"
        for gate in basis_gates:
            target.add_instruction(QUTIP_TO_QISKIT_MAP[gate], properties=None)

        # Essential primitives
        target.add_instruction(Measure(), properties=None)

        # TODO: Add Barrier implementation to QuTiP.
        # target.add_instruction(Barrier(), properties=None)
        return target

    @classmethod
    def _default_options(cls) -> Options:
        """
        Default options for the backend.

        Returns
        -------
        :class:`qiskit.providers.Option`
            Option object that stores stores the different options
            (e.g. shots) for the backend.
        """
        options = Options()
        options.shots = 1024
        options.set_validator("shots", int)
        return options

    def run(
        self, run_input: QuantumCircuit | list[QuantumCircuit], **run_options
    ) -> Job:
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        run_input : list[:class:`qiskit.circuit.QuantumCircuit`]
            List of ``qiskit`` circuits to be simulated.

        **run_options : dict[str, Any]
            Additional run options for the backend. Valid options are:
            shots : int
                Number of times to sample the results.

        Returns
        -------
        :class:`.Job`
            Job object that stores results and execution data.
        """

        if not isinstance(run_input, list):
            run_input = [run_input]

        if len(run_input) > self.max_circuits:
            raise ValueError(
                f"Passed ${len(run_input)} circuits to the backend,\
                while max_cicruits is defined as ${self.max_circuits}"
            )

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

    @abstractmethod
    def _run_job(
        self, job_id: str, qiskit_circuits: List[QuantumCircuit]
    ) -> Result:
        """
        Given the `job_id` and `qiskit_circuits` list implement the
        simulation logic and return :class:`qiskit.result.Result` object.
        """
        pass
