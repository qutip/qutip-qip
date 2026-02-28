import numpy as np
from typing import Union
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate

__all__ = ["dj_oracle", "deutsch_jozsa"]


def dj_oracle(num_qubits: int, case: str) -> QubitCircuit:
    """
    Utility to create a Deutsch-Jozsa oracle.

    Parameters
    ----------
    num_qubits : int
        The number of input qubits (excluding the auxiliary qubit).
    case : str
        Case should be either "constant" or "balanced", denoting whether
        the state of the oracle (f(x)) is constant or balanced.

    Returns
    -------
    QubitCircuit
        Either a constant or a balanced oracle for Deutsch-Jozsa algorithm,
    """

    total_qubits = num_qubits + 1
    qc = QubitCircuit(total_qubits)

    if case == "constant":
        if np.random.randint(2) == 1:
            qc.add_gate("X", targets=num_qubits)

    elif case == "balanced":
        b_mask = np.random.randint(1, 2**num_qubits)
        binary_rep = format(b_mask, f"0{num_qubits}b")

        for i, bit in enumerate(binary_rep):
            if bit == "1":
                qc.add_gate("CNOT", controls=i, targets=num_qubits)

    return qc


def deutsch_jozsa(
    num_qubits: int, oracle: Union[QubitCircuit, Gate]
) -> QubitCircuit:
    """
    Construct the Deutsch-Josza algorithm circuit

    The algorithim determines if a hidden boolean function is constant
    or balanced in a single query.

    Parameters
    ----------
    num_qubits : int
        The number of input qubits (excluding the auxiliary qubit).
    oracle : QubitCircuit or Gate
        The quantum oracle representing the function f(x).
        It must act on (num_qubits + 1) qubits, where the last
        qubit is the auxiliary workspace.

    Returns
    -------
    QubitCircuit
        The circuit implementing the Deutsch-Jozsa algorithm.

    References
    ----------
    .. [1] Deutsch, D. and Jozsa, R., "Rapid solution of problems by
       quantum computation," Proc. R. Soc. Lond. A, 439 (1992).
    """

    total_qubits = num_qubits + 1
    qc = QubitCircuit(num_qubits + 1)

    # Initialize auxilary Qubit:
    qc.add_gate("X", targets=num_qubits)

    # Apply Hadamard to all gates(including auxillary):
    for i in range(total_qubits):
        qc.add_gate("SNOT", targets=i)

    # Apply phase oracle:
    if isinstance(oracle, QubitCircuit):
        qc.gates.extend(oracle.gates)
    else:
        qc.add_gate(oracle)

    # Apply Hadamard Gate to input qubits only:
    for i in range(num_qubits):
        qc.add_gate("SNOT", targets=i)

    return qc
