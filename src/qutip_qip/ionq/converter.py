from qutip import Qobj
from qutip_qip.circuit import QubitCircuit, CircuitResult
import numpy as np


def convert_qutip_circuit(qc: QubitCircuit) -> dict:
    """
    Convert a qutip_qip circuit to an IonQ circuit.

    Parameters
    ----------
    qc: QubitCircuit
        The qutip_qip circuit to be converted.

    Returns
    -------
    dict
        The IonQ circuit.
    """
    ionq_circuit = []
    for gate in qc.gates:
        g = {"gate": gate.name}
        # Map target(s) and control(s) depending on the number of qubits
        for attr, key in (("targets", "target"), ("controls", "control")):
            items = getattr(gate, attr, None)
            if items:
                g[key if len(items) == 1 else key + "s"] = (
                    items[0] if len(items) == 1 else items
                )
        # Include arg_value as angle, phase, and rotation if it exists
        if getattr(gate, "arg_value", None) is not None:
            g.update(
                {
                    "angle": gate.arg_value,
                    "phase": gate.arg_value,
                    "rotation": gate.arg_value,
                }
            )
        ionq_circuit.append(g)
    return ionq_circuit


def convert_ionq_response_to_circuitresult(ionq_response: dict):
    """
    Convert an IonQ response to a CircuitResult.

    Parameters
    ----------
    ionq_response: dict
        The IonQ response {state: probability, ...}.

    Returns
    -------
    CircuitResult
        The CircuitResult.
    """
    # Calculate the number of qubits based on the binary representation of the highest state
    num_qubits = max(len(state) for state in ionq_response.keys())

    # Initialize an empty density matrix for the mixed state
    density_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

    # Iterate over the measurement outcomes and their probabilities
    for state, probability in ionq_response.items():
        # Ensure state string is correctly padded for single-qubit cases
        binary_state = format(int(state, base=10), f"0{num_qubits}b")
        index = int(binary_state, 2)  # Convert binary string back to integer

        # Update the density matrix to include this measurement outcome
        state_vector = np.zeros((2**num_qubits,), dtype=complex)
        state_vector[index] = (
            1.0  # Pure state corresponding to the measurement outcome
        )
        density_matrix += probability * np.outer(
            state_vector, state_vector.conj()
        )  # Add weighted outer product

    # Convert the numpy array to a Qobj density matrix
    qobj_density_matrix = Qobj(
        density_matrix, dims=[[2] * num_qubits, [2] * num_qubits]
    )

    return CircuitResult(
        [qobj_density_matrix], [1.0]
    )  # Return the density matrix wrapped in CircuitResult


def create_job_body(
    circuit: dict,
    shots: int,
    backend: str,
    gateset: str,
    format: str = "ionq.circuit.v0",
) -> dict:
    """
    Create the body of a job request.

    Parameters
    ----------
    circuit: dict
        The IonQ circuit.
    shots: int
        The number of shots.
    backend: str
        The simulator or QPU backend.
    gateset: str
        Either native or compiled gates.
    format: str
        The format of the circuit.

    Returns
    -------
    dict
        The body of the job request.
    """
    return {
        "target": backend,
        "shots": shots,
        "input": {
            "format": format,
            "gateset": gateset,
            "circuit": circuit,
            "qubits": len(
                {
                    q
                    for g in circuit
                    for q in g.get("targets", [])
                    + g.get("controls", [])
                    + [g.get("target")]
                    + [g.get("control")]
                    if q is not None
                }
            ),
        },
    }
