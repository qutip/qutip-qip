import numpy as np

from qutip import Qobj
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import Processor
from qutip_qip.qiskit import QiskitSimulatorBase
from qutip_qip.qiskit.utils import (
    QUTIP_TO_QISKIT_GATE_MAP,
    convert_qiskit_circuit_to_qutip,
    get_probabilities,
    sample_shots,
)

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.quantum_info import Statevector, DensityMatrix

DEFAULT_BASIS_GATE_LIST = list(QUTIP_TO_QISKIT_GATE_MAP.keys())


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
    num_qubits : int, Optional
        num_qubits for the Pulse Simulator Backend.
        Defaults to 10 qubits.
    basis_gates : list[str], Optional
        The basis gates names in QuTip.
        Defaults to (PHASEGATE, X, Y, Z, RX, RY, RZ,
        Hadamard, S, T, SWAP, QASMU, CX, CY, CZ,
        CRX, CRY, CRZ, CPHASE)
    max_shots : int, Optional
        Maximum number of shots the Backend support
        while sampling.
    max_circuits : int, Optional
        The maximum number of circuits that can be
        run in a single job.
    name : str, Optional
        Name of the Pulse Simulator Backend
    description : str, Optional
        Description of the Pulse Simulator Backend
    version : str, Optional
        Version of Pulse Simulator Backend

    Attributes
    ----------
    processor : :class:`.Processor`
        The processor model to be used for simulation.
    """

    def __init__(
        self,
        processor: Processor,
        num_qubits: int = 10,
        basis_gates: list[str] = DEFAULT_BASIS_GATE_LIST,
        max_shots: int = 1e6,
        max_circuits: int = 1,
        name: str = "pulse_simulator",
        description: str = "A qutip-qip based pulse-level \
            simulator based on the open system solver.",
        version: str = "0.1",
    ):
        super().__init__(
            name=name,
            description=description,
            version=version,
            num_qubits=num_qubits,
            basis_gates=basis_gates,
            max_shots=max_shots,
            max_circuits=max_circuits,
        )

        self._processor = processor

    @property
    def processor(self) -> Processor:
        return self._processor

    def _run_job(
        self, job_id: str, qiskit_circuit: list[QuantumCircuit]
    ) -> Result:
        """
        Run a :class:`.QubitCircuit` on the Pulse Simulator.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.
        qiskit_circuit : list[:class:`.QuantumCircuit`]
            The circuit obtained after conversion
            from :class:`.QuantumCircuit` to :class:`.QubitCircuit`.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the simulation.
        """
        final_states = []
        qutip_circuits = []

        for circuit in qiskit_circuit:
            qutip_circuit = convert_qiskit_circuit_to_qutip(circuit)
            qutip_circuits.append(qutip_circuit)

            self.processor.load_circuit(qutip_circuit)
            zero_state = self.processor.generate_init_processor_state()
            result = self.processor.run_state(zero_state)
            final_states.append(
                self.processor.get_final_circuit_state(result.states[-1])
            )

        return self._parse_results(
            job_id=job_id,
            final_states=final_states,
            qutip_circuits=qutip_circuits,
        )

    def _parse_results(
        self,
        job_id: str,
        qutip_circuits: list[QubitCircuit],
        final_states: list[Qobj],
    ) -> Result:
        """
        Returns a parsed object of type :class:`qiskit.result.Result`
        for the pulse simulators.

        Parameters
        ----------
        job_id : str
            Unique ID identifying a job.
        qutip_circuits : list[:class:`.QubitCircuit`]
            The circuits being simulated.
        final_states : list[:class:`.Qobj`]
            The resulting density matrices obtained from `run_state` on
            circuits using the Pulse simulator processors.

        Returns
        -------
        :class:`qiskit.result.Result`
            Result of the pulse simulation.
        """

        if len(final_states) != len(qutip_circuits):
            raise ValueError(
                "Number of final_states must be = to number of qutip circuits"
            )

        exp_res = []
        num_circuits = len(final_states)

        for i in range(num_circuits):
            count_probs = {}
            counts = None
            final_state = final_states[i]
            qutip_circuit = qutip_circuits[i]

            # calculate probabilities of required states
            if final_state:
                for i, prob in enumerate(get_probabilities(final_state)):
                    if not np.isclose(prob, 0):
                        count_probs[hex(i)] = prob
                # sample the shots from obtained probabilities
                counts = sample_shots(count_probs, self.options["shots"])

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
                    header=header,
                    data=exp_res_data,
                )
            )

        result = Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=exp_res,
        )
        return result
