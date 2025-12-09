import numpy as np

import qutip
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import Processor
from qutip_qip.qiskit import QiskitSimulatorBase

import qiskit
from qiskit.providers import Options
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.qobj import QobjExperimentHeader
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
