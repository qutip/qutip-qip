import numpy as np
from typing import List, Union, Optional
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, Z, ControlledGate, CSIGN

__all__ = ["grover", "grover_oracle"]


class OracleGate(Gate):
    """
    Custom gate that wraps an arbitrary quantum oracle operator.
    """

    def __init__(self, targets, U, **kwargs):
        """
        Initialize an OracleGate.

        Parameters
        ----------
        targets : int or list of int
            The qubit(s) on which the oracle gate acts.
        U : Qobj
            The unitary operator representing the oracle.
        **kwargs
            Additional keyword arguments to pass to the parent Gate class.
        """
        super().__init__(targets=targets, **kwargs)
        self.U = U
        self.latex_str = r"Oracle"

    def get_compact_qobj(self):
        return self.U


def grover_oracle(
    qubits: Union[int, List[int]], marked_states: Union[int, List[int]]
) -> QubitCircuit:
    """
    Constructs a Phase Oracle circuit for Grover's algorithm.

    Parameters
    ----------
    qubits : int or list of int
        The specific qubit indices the oracle acts on.
    marked_states : int or list of int
        The states to mark (integer representation).

    Returns
    -------
    QubitCircuit
        The oracle circuit.
    """
    if isinstance(qubits, int):
        qubits = list(range(qubits))

    if isinstance(marked_states, int):
        marked_states = [marked_states]

    n_qubits = len(qubits)
    # We need a circuit big enough to hold the highest qubit index
    qc = QubitCircuit(max(qubits) + 1)

    for state in marked_states:
        # Safety check
        if state >= 2**n_qubits:
            raise ValueError(
                f"Marked state {state} is out of bounds for {n_qubits} qubits."
            )

        binary_rep = format(state, f"0{n_qubits}b")

        # Flip 0 bits:
        for i, char in enumerate(binary_rep):
            if char == "0":
                qc.add_gate("X", targets=qubits[i])

        # Change state by Multi Controlled Z Gate
        ctrls = qubits[:-1]
        tgt = qubits[-1]

        ctrl_val = 2 ** len(ctrls) - 1

        if len(ctrls) == 1:
            qc.add_gate("CSIGN", controls=ctrls, targets=tgt)
        else:
            qc.add_gate(
                ControlledGate(
                    controls=ctrls,
                    targets=tgt,
                    control_value=ctrl_val,
                    target_gate=Z,
                )
            )

        # uncompute by X
        for i, char in enumerate(binary_rep):
            if char == "0":
                qc.add_gate("X", targets=qubits[i])

    return qc


def grover(
    oracle: QubitCircuit,
    qubits: Union[int, List[int]],
    num_solutions: int,
    num_iterations: Optional[int] = None,
) -> QubitCircuit:
    """
    Construct the Grover search algorithim's circuit

    Parameters
    ----------
    oracle : QubitCircuit, Gate, or Qobj
        The oracle that flips the phase of marked states.
    qubits : int or list of int
        The qubits to run the search on.
    num_iterations : int, optional
        Number of iterations. Defaults to optimal for 1 solution.

    Returns
    -------
    QubitCircuit
        Quantum Circuit implementing Grover's search algorithim

    Raises
    ------
    ValueError
        If the oracle gate targets do not match the algorithm qubits.

    Notes
    -----
    The algorithm performs the following steps:
    1. Apply Hadamard gates to all qubits to create superposition
    2. For each iteration:
       a. Apply the oracle (phase flip on marked states)
       b. Apply the diffusion operator (inversion about the mean):
          - Hadamard gates
          - X gates
          - Multi-controlled Z gate
          - X gates
          - Hadamard gates

    References
    ----------
    .. [1] L. K. Grover, "A fast quantum mechanical algorithm for database
       search," Proceedings of the 28th Annual ACM Symposium on Theory of
       Computing (1996).

    Examples
    --------
    Search for state |01⟩ using 2 qubits:

        >>> oracle = grover_oracle([0, 1], marked_states=1)
        >>> qc = grover(oracle, qubits=[0, 1])

    """
    if isinstance(qubits, int):
        qubits = list(range(qubits))

    n_qubits = len(qubits)
    qc = QubitCircuit(max(qubits) + 1)

    # Superposition:
    for q in qubits:
        qc.add_gate("SNOT", targets=q)

    # Calculate optimal Iterations if none provided:
    if num_iterations is None:
        N = 2**n_qubits
        if num_solutions >= N:
            num_iterations = 0
        else:
            calc = (np.pi / 4) * np.sqrt(N / num_solutions)
            num_iterations = max(1, int(np.floor(calc)))

    # Grover Iterations:
    for _ in range(num_iterations):
        # Oracle
        if isinstance(oracle, QubitCircuit):
            qc.gates.extend(oracle.gates)
        elif isinstance(oracle, Gate):
            if set(oracle.targets) != set(qubits):
                raise ValueError(
                    f"Oracle gate targets {oracle.targets} do not match algorithm qubits {qubits}"
                )
            qc.add_gate(oracle)
        else:
            qc.add_gate(OracleGate(qubits, oracle))

        # Diffusion (Inversion about the mean)
        for q in qubits:
            qc.add_gate("SNOT", targets=q)

        for q in qubits:
            qc.add_gate("X", targets=q)

        # Projection (Multi-Controlled Z)
        ctrls = qubits[:-1]
        tgt = qubits[-1]
        ctrl_val = 2 ** (len(ctrls)) - 1

        if len(ctrls) == 1:
            qc.add_gate("CSIGN", controls=ctrls, targets=tgt)
        else:
            qc.add_gate(
                ControlledGate(
                    controls=ctrls,
                    targets=tgt,
                    control_value=ctrl_val,
                    target_gate=Z,
                )
            )

        for q in qubits:
            qc.add_gate("X", targets=q)

        for q in qubits:
            qc.add_gate("SNOT", targets=q)

    return qc
