"""Backends for simulating qiskit circuits."""

from collections import Counter
from abc import abstractmethod
from typing import List, Union
import uuid
import random
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import Measure
from qiskit.providers import BackendV2, Options
from qiskit.result import Counts, Result
from qiskit.transpiler.target import Target

from qutip import Qobj
from qutip_qip.qiskit import Job
from qutip_qip.qiskit.utils import QUTIP_TO_QISKIT_MAP


class QiskitSimulatorBase(BackendV2):
    """
    The base class for ``qutip_qip`` based ``qiskit`` backends.
    This class must always be inherited, never instantiated as
    abstract method run, _run_job is left to the child class.
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
            provider="QuTiP-qip",
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
    def max_circuits(self) -> Union[int | None]:
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
        """Builds a :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: :class:`qiskit.transpiler.Target`
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

        Options
        -------
        shots : int
            Number of times to sample the results.
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
        pass

    def _get_probabilities(self, state: Qobj) -> np.ndarray:
        """
        Given a state, return an array of corresponding probabilities.

        Parameters
        ----------
        state: Qobj
            Qobj (type - density matrix, ket) state
            obtained after circuit application.

        Returns
        -------
        :class:`np.ndarray`
            Returns the ``numpy`` corresponding to the basis state.
        """
        if state.type == "oper":
            # diagonal elements of a density matrix are
            # the probabilities
            return state.diag()

        # squares of coefficients are the probabilities for a ket vector
        return np.square(np.real(state.data_as("ndarray", copy=False)))

    def _sample_shots(self, count_probs: dict) -> Counts:
        """
        Sample measurements from a given probability distribution.

        Parameters
        ----------
        count_probs: dict
            Probability distribution corresponding
            to different classical outputs.

        Returns
        -------
        :class:`qiskit.result.Counts`
            Returns the ``Counts`` object sampled according to
            the given probabilities and configured shots.
        """
        weights = []
        for p in count_probs.values():
            if hasattr(p, "item"):
                weights.append(float(p.item()))  # For multiple choice
            else:
                weights.append(float(p))  # For a trivial circuit with output 1

        samples = random.choices(
            list(count_probs.keys()), weights, k=self._options["shots"]
        )
        return Counts(Counter(samples))
