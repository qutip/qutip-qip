"""Backends for simulating qiskit circuits."""

import numpy as np
import uuid
import random
from collections import Counter

import qutip
import qiskit
from qutip import basis
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit.circuitsimulator import CircuitResult
from qutip_qip.device import Processor

from .job import Job
from .converter import convert_qiskit_circuit
from qiskit.providers import BackendV1, Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result, Counts
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit import QuantumCircuit
from qiskit.qobj import QobjExperimentHeader


class QiskitSimulatorBase(BackendV1):
    """
    The base class for ``qutip_qip`` based ``qiskit`` backends.
    """

    def __init__(self, configuration=None, **fields):
        if configuration is None:
            configuration_dict = self._DEFAULT_CONFIGURATION
        else:
            configuration_dict = self._DEFAULT_CONFIGURATION.copy()
            for k, v in configuration.items():
                configuration_dict[k] = v

        configuration = QasmBackendConfiguration.from_dict(configuration_dict)

        super().__init__(configuration=configuration)

        self.options.set_validator(
            "shots", (1, self.configuration().max_shots)
        )

    def run(self, qiskit_circuit: QuantumCircuit, **run_options) -> Job:
        """
        Simulates a circuit on the required backend.

        Parameters
        ----------
        qiskit_circuit : :class:`qiskit.circuit.QuantumCircuit`
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
        # configure the options
        self.set_options(
            shots=(
                run_options["shots"]
                if "shots" in run_options
                else self._default_options().shots
            ),
            allow_custom_gate=(
                run_options["allow_custom_gate"]
                if "allow_custom_gate" in run_options
                else self._default_options().allow_custom_gate
            ),
        )
        qutip_circ = convert_qiskit_circuit(
            qiskit_circuit,
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
        shots = self.options.shots
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

    def __init__(self, configuration=None, **fields):
        super().__init__(configuration=configuration, **fields)

    def _parse_results(
        self,
        statistics: CircuitResult,
        job_id: str,
        qutip_circuit: QubitCircuit,
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

        header = QobjExperimentHeader.from_dict(
            {
                "name": (
                    qutip_circuit.name
                    if hasattr(qutip_circuit, "name")
                    else ""
                ),
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

    @classmethod
    def _default_options(cls):
        """
        Default options for the backend.

        Options
        -------
        shots : int
            Number of times to sample the results.

        allow_custom_gate : bool
            Allow conversion of circuit using unitary matrices
            for custom gates.
        """
        return Options(shots=1024, allow_custom_gate=True)


class QiskitPulseSimulator(QiskitSimulatorBase):
    """
    ``qiskit`` backend dealing with pulse-level simulation.

    Parameters
    ----------
    processor : :class:`.Processor`
        The processor model to be used for simulation.
        An instance of the required :class:`.Processor`
        object is to be provided after initialising
        it with the required parameters.

    configuration : dict
        Configurable attributes of the backend.

    Attributes
    ----------
    processor : :class:`.Processor`
        The processor model to be used for simulation.
    """

    processor = None
    MAX_QUBITS_MEMORY = 10
    BACKEND_NAME = "pulse_simulator"
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
        "description": "A qutip-qip based pulse-level \
            simulator based on the open system solver.",
        "basis_gates": [],
        "gates": [],
    }

    def __init__(self, processor: Processor, configuration=None, **fields):
        self.processor = processor
        super().__init__(configuration=configuration, **fields)

    def _parse_results(
        self, final_state: qutip.Qobj, job_id: str, qutip_circuit: QubitCircuit
    ) -> qiskit.result.Result:
        """
        Returns a parsed object of type :class:`qiskit.result.Result`
        for the pulse simulators.

        Parameters
        ----------
        density_matrix : :class:`.Qobj`
            The resulting density matrix obtained from `run_state` on
            a circuit using the Pulse simulator processors.

        job_id : str
            Unique ID identifying a job.

        qutip_circuit : :class:`.QubitCircuit`
            The circuit being simulated.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the pulse simulation.
        """
        count_probs = {}
        counts = None

        # calculate probabilities of required states
        if final_state:
            for i, prob in enumerate(self._get_probabilities(final_state)):
                if not np.isclose(prob, 0):
                    count_probs[hex(i)] = prob
            # sample the shots from obtained probabilities
            counts = self._sample_shots(count_probs)

        exp_res_data = ExperimentResultData(
            counts=counts,
            statevector=(
                Statevector(data=final_state.full())
                if final_state.type == "ket"
                else DensityMatrix(data=final_state.full())
            ),
        )

        header = QobjExperimentHeader.from_dict(
            {
                "name": (
                    qutip_circuit.name
                    if hasattr(qutip_circuit, "name")
                    else ""
                ),
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
        Run a :class:`.QubitCircuit` on the Pulse Simulator.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.

        qutip_circuit : :class:`.QubitCircuit`
            The circuit obtained after conversion
            from :class:`.QuantumCircuit` to :class:`.QubitCircuit`.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the simulation.
        """
        zero_state = self.processor.generate_init_processor_state()

        self.processor.load_circuit(qutip_circuit)
        result = self.processor.run_state(zero_state)

        final_state = self.processor.get_final_circuit_state(result.states[-1])

        return self._parse_results(
            final_state=final_state, job_id=job_id, qutip_circuit=qutip_circuit
        )

    @classmethod
    def _default_options(cls):
        """
        Default options for the backend.

        Options
        -------
        shots : int
            Number of times to sample the results.

        allow_custom_gate : bool
            Allow conversion of circuit using unitary matrices
            for custom gates.
        """
        return Options(shots=1024, allow_custom_gate=True)
