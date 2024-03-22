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
        if (
            hasattr(gate, "name")
            and hasattr(gate, "targets")
            and hasattr(gate, "controls")
        ):
            g = {
                "gate": gate.name,
            }
            if gate.targets is not None:
                if len(gate.targets) == 1:
                    g["target"] = gate.targets[0]
                else:
                    g["targets"] = gate.targets
            if gate.controls is not None:
                if len(gate.controls) == 1:
                    g["control"] = gate.controls[0]
                else:
                    g["controls"] = gate.controls
            if hasattr(gate, "arg_value") and gate.arg_value is not None:
                g["angle"] = gate.arg_value
                g["phase"] = gate.arg_value
                g["rotation"] = gate.arg_value
            ionq_circuit.append(g)
    return ionq_circuit


def convert_ionq_response_to_circuitresult(
    ionq_response: dict,
) -> CircuitResult:
    """
    Convert an IonQ response to a CircuitResult.

    Parameters
    ----------
    ionq_response: dict
        The IonQ response.

    Returns
    -------
    CircuitResult
        The CircuitResult.
    """
    states = list(ionq_response.keys())
    # probabilities = list(ionq_response.values())

    max_state = max(int(state) for state in states)
    num_qubits = int(np.ceil(np.log2(max_state + 1))) if max_state > 0 else 1

    final_states = []
    final_probabilities = []

    for state in states:
        binary_state = format(int(state), "0{}b".format(num_qubits))
        state_vector = np.zeros((2**num_qubits,), dtype=complex)
        index = int(binary_state, 2)
        state_vector[index] = 1.0

        qobj_state = Qobj(
            state_vector, dims=[[2] * num_qubits, [1] * num_qubits]
        )
        final_states.append(qobj_state)
        final_probabilities.append(ionq_response[state])

    return CircuitResult(final_states, final_probabilities)


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
