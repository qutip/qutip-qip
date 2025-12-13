from typing import List
import uuid
import numpy as np

from qutip import Qobj
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import Processor
from qutip_qip.qiskit import Job, QiskitSimulatorBase
from qutip_qip.qiskit.converter import convert_qiskit_circuit_to_qutip

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector, DensityMatrix


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

    def __init__(self, processor: Processor, configuration=None):
        if configuration is None:
            configuration = self._DEFAULT_CONFIGURATION

        super().__init__(
            name=configuration["backend_name"],
            description=configuration["description"],
        )

        self.processor = processor

    def target(self):
        pass

    def max_circuits(self):
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

    def _run_job(
        self, job_id: str, qiskit_circuit: List[QuantumCircuit]
    ) -> Result:
        """
        Run a :class:`.QubitCircuit` on the Pulse Simulator.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.

        qiskit_circuit : :class:`.QuantumCircuit`
            The circuit obtained after conversion
            from :class:`.QuantumCircuit` to :class:`.QubitCircuit`.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the simulation.
        """
        qutip_circuit = convert_qiskit_circuit_to_qutip(qiskit_circuit)
        self.processor.load_circuit(qutip_circuit)

        zero_state = self.processor.generate_init_processor_state()
        result = self.processor.run_state(zero_state)
        final_state = self.processor.get_final_circuit_state(result.states[-1])

        return self._parse_results(
            final_state=final_state, job_id=job_id, qutip_circuit=qutip_circuit
        )

    def _parse_results(
        self,
        job_id: str,
        qutip_circuit: QubitCircuit,
        final_state: Qobj,
    ) -> Result:
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
            backend_name=self.BACKEND_NAME,
            backend_version=0.1,
            qobj_id=id(qutip_circuit),
            job_id=job_id,
            success=True,
            results=[exp_res],
        )

        return result
