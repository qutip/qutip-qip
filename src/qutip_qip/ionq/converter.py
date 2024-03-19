from qutip.qip.circuit import QubitCircuit, Gate


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
        if isinstance(gate, Gate):
            ionq_circuit.append(
                {
                    "gate": gate.name,
                    "targets": gate.targets,
                    "controls": gate.controls or [],
                }
            )
    return ionq_circuit


def create_job_body(
    circuit: dict,
    shots: int,
    backend: str,
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
    format: str
        The format of the circuit.

    Returns
    -------
    dict
        The body of the job request.
    """
    return {
        "target": backend,
        "input": {
            "format": format,
            "qubits": len(
                {
                    q
                    for g in circuit
                    for q in g.get("targets", []) + g.get("controls", [])
                }
            ),
            "circuit": circuit,
        },
        "shots": shots,
    }
