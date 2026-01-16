import numpy as np
from qutip_qip.operations import Gate, ControlledGate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.algorithms import qft_gate_sequence

__all__ = ["qpe"]


class CustomGate(Gate):
    """
    Custom gate that wraps an arbitrary quantum operator.
    """

    latex_str = r"U"

    def __init__(self, targets, U, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.targets = targets if isinstance(targets, list) else [targets]
        self._U = U
        self.kwargs = kwargs

    def get_compact_qobj(self):
        return self._U


def qpe(U, num_counting_qubits, target_qubits=None, to_cnot=False):
    """
    Quantum Phase Estimation circuit implementation for QuTiP.

    Parameters
    ----------
    U : Qobj
        Unitary operator whose eigenvalue we want to estimate.
        Should be a unitary quantum operator.
    num_counting_qubits : int
        Number of counting qubits to use for the phase estimation.
        More qubits provide higher precision.
    target_qubits : int or list, optional
        Index or indices of the target qubit(s) where the eigenstate is prepared.
        If None, target_qubits is set automatically based on U's dimension.
    to_cnot : bool, optional
        Flag to decompose controlled phase gates to CNOT gates (default: False)

    Returns
    -------
    qc : instance of QubitCircuit
        Gate sequence implementing Quantum Phase Estimation.
    """
    if num_counting_qubits < 1:
        raise ValueError("Minimum value of counting qubits must be 1")

    # Handle target qubits specification
    if target_qubits is None:
        dim = U.dims[0][0]
        num_target_qubits = int(np.log2(dim))
        if 2**num_target_qubits != dim:
            raise ValueError(
                f"Unitary operator dimension {dim} is not a power of 2"
            )
        target_qubits = list(
            range(num_counting_qubits, num_counting_qubits + num_target_qubits)
        )
    elif isinstance(target_qubits, int):
        target_qubits = [target_qubits]
        num_target_qubits = 1
    else:
        num_target_qubits = len(target_qubits)

    total_qubits = num_counting_qubits + num_target_qubits
    qc = QubitCircuit(total_qubits)

    # Apply Hadamard gates to all counting qubits
    for i in range(num_counting_qubits):
        qc.add_gate("SNOT", targets=[i])

    # Apply controlled-U gates with increasing powers
    for i in range(num_counting_qubits):
        power = 2 ** (num_counting_qubits - i - 1)
        # Create U^power
        U_power = U if power == 1 else U**power

        # Add controlled-U^power gate
        controlled_u = ControlledGate(
            controls=[i],
            targets=target_qubits,
            control_value=1,
            target_gate=CustomGate(target_qubits, U_power),
        )
        qc.add_gate(controlled_u)

    # Add inverse QFT on counting qubits
    inverse_qft_circuit = qft_gate_sequence(
        num_counting_qubits, swapping=True, to_cnot=to_cnot
    ).reverse_circuit()
    for gate in inverse_qft_circuit.gates:
        qc.add_gate(gate)

    return qc
