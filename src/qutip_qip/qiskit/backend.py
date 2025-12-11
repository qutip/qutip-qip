"""Backends for simulating qiskit circuits."""

from collections import Counter
from abc import abstractmethod
from typing import List
import uuid
import random
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.result import Counts, Result

from qutip import Qobj
from qutip_qip.qiskit.job import Job


class QiskitSimulatorBase(BackendV2):
    """
    The base class for ``qutip_qip`` based ``qiskit`` backends.
    This class must always be inherited, never instantiated as
    abstract methods target and max_circuits are left to parent class.
    """

    def __init__(self, name=None, description=None):
        super().__init__(
            provider="QuTiP-qip",
            name=name, 
            description=description
        )
        
    @classmethod
    def _default_options(cls):
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

    @abstractmethod
    def _run_job(self, job_id: str, qiskit_circuits: List[QuantumCircuit]) -> Result:
        pass

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
            if (shots <= 0):
                raise ValueError(f'shots ${shots} must be a positive number')

            self.set_options(shots=shots)

        job_id = str(uuid.uuid4())
        job = Job(
            backend=self,
            job_id=job_id,
            circuit = run_input,
        )
        job.submit()
        return job

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
        return np.square(np.real(state.data_as('ndarray', copy=False)))

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
                weights.append(float(p.item())) # For multiple choice
            else:
                weights.append(float(p)) # For a trivial circuit with output 1

        samples = random.choices(
            list(count_probs.keys()), weights, k=self._options["shots"]
        )
        return Counts(Counter(samples))
    