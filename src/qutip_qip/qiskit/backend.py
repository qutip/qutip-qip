"""Backends for simulating qiskit circuits."""

import numpy as np
import uuid
import random
from collections import Counter

from .job import Job
from .converter import convert_qiskit_circuit_to_qutip
from qiskit.providers import BackendV2
from qiskit.providers import Options
from qiskit.result import Counts
from qiskit.circuit import QuantumCircuit


class QiskitSimulatorBase(BackendV2):
    """
    The base class for ``qutip_qip`` based ``qiskit`` backends.
    """

    def __init__(self, name=None, description=None):
        super().__init__(
            name=name, 
            description=description
        )
        self._provider = "QuTiP-qip"
        
    def _default_options(self):
        """
        This method is defined in BackendV2 and will run during __init__ to set self._options
        """
        options = Options()
        options.shots = 1000
        options.allow_custom_gate = True

        options.set_validator("shots", int)
        options.set_validator("allow_custom_gate", bool)

        return options


    def run(self, run_input: QuantumCircuit, **run_options) -> Job:
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        # In V2 this can be a list of QuantumCircuits to be simulated
        run_input : :class:`qiskit.circuit.QuantumCircuit`
            The ``qiskit`` circuit to be simulated.

        **run_options:
            Additional run options for the backend.

            Valid options are:

            shots : int
                Number of times to sample the results.
            allow_custom_gate: bool
                Allow conversion of circuit using unitary matrices
                for custom gates.

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

            self.set_options({"shots": shots})

        # Set allow_custom_gate
        if ("allow_custom_gate" in run_options):
            allow_custom_gate = run_options["allow_custom_gate"]
            self.set_options({"allow_custom_gate": allow_custom_gate})

        qutip_circ = convert_qiskit_circuit_to_qutip(
            run_input,
            allow_custom_gate=self.options.allow_custom_gate,
        )

        job_id = str(uuid.uuid4())

        job = Job(
            backend=self,
            job_id=job_id,
            result=self._run_job(job_id, qutip_circ),
        )
        return job

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
        shots = self._options["shots"]
        samples = random.choices(
            list(count_probs.keys()), list(count_probs.values()), k=shots
        )
        return Counts(Counter(samples))

    def _get_probabilities(self, state):
        """
        Given a state, return an array of corresponding probabilities.
        """
        if state.type == "oper":
            # diagonal elements of a density matrix are
            # the probabilities
            return state.diag()

        # squares of coefficients are the probabilities
        # for a ket vector
        return np.array([np.abs(coef) ** 2 for coef in state])
