import numpy as np
from typing import Sequence
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, Z, ControlledGate

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
    qubits: int | Sequence[int], marked_states: int | Sequence[int]
) -> QubitCircuit:
    """
    Constructs a Phase Oracle circuit for Grover's algorithm.

    Parameters
    ----------
    qubits : int or sequence of int
        The specific qubit indices the oracle acts on.
    marked_states : int or sequence of int
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
        if state < 0 or state >= 2**n_qubits:
            raise ValueError(
                f"Marked state {state} is out of bounds for {n_qubits} qubits. Valid range is [0, {2**n_qubits-1}]."
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
    qubits: int | Sequence[int],
    num_solutions: int,
    num_iterations: int | None = None,
    N: int | None = None,
) -> QubitCircuit:
    """
    Construct the Grover search algorithm's circuit.

    Parameters
    ----------
    oracle : :class:`~.circuit.QubitCircuit` or :class:`~.operations.Gate` or :class:`qutip.Qobj`
        The oracle that flips the phase of marked states.
    qubits : int or sequence of int
        The qubits to run the search on.
    num_solutions : int
        The number of expected solutions M.
    num_iterations : int, optional
        Number of iterations. Defaults to optimal for M solutions.
    N: int,optional
        Total number of qubits in the system.

    Returns
    -------
    QubitCircuit
        Quantum Circuit implementing Grover's search algorithm.

    Raises
    ------
    ValueError
        If the oracle gate targets do not match the algorithm qubits.

    Notes
    -----
    The algorithm performs the following steps:

    1. Apply Hadamard gates to all qubits to create superposition.
    2. For each iteration:
        a. Apply the oracle (phase flip on marked states).
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
    Search for state :math:`|01\\rangle` using 2 qubits:

    >>> from qutip_qip.algorithms.grover import grover, grover_oracle
    >>> oracle = grover_oracle([0, 1], marked_states=1)
    >>> qc = grover(oracle, qubits=[0, 1], num_solutions=1)

    """
    if isinstance(qubits, int):
        qubits = list(range(qubits))

    n_qubits = len(qubits)

    if N is not None:
        if N <= 0:
            raise ValueError(f"N must be a positive integer, got {N}.")
        min_required = max(qubits) + 1
        if N < min_required:
            raise ValueError(
                f"N={N} is too small. The search qubits {qubits} "
                f"require at least {min_required} total qubits."
            )

    total_qubits = N if N is not None else (max(qubits) + 1)
    qc = QubitCircuit(total_qubits)

    # Superposition:
    for q in qubits:
        qc.add_gate("SNOT", targets=q)

    # Calculate optimal Iterations if none provided:
    if num_iterations is not None and num_iterations <= 0:
        raise ValueError(
            f"num_iterations must be a positive integer, got {num_iterations}."
        )

    if num_iterations is None:
        N = 2**n_qubits

        if num_solutions <= 0:
            raise ValueError("num_solutions must be greater than 0.")
        elif num_solutions >= N:
            raise ValueError(
                "Number of solutions is equal/greater to the search space."
            )
        else:
            calc = (np.pi / 4) * np.sqrt(N / num_solutions)
            num_iterations = int(np.floor(calc))

    # Grover Iterations:
    for _ in range(num_iterations):
        # Oracle
        if isinstance(oracle, QubitCircuit):
            qc.gates.extend(oracle.gates)
        elif isinstance(oracle, Gate):
            if not set(oracle.targets).issubset(qubits):
                raise ValueError(
                    f"Oracle gate targets {oracle.targets} are not a subset of algorithm qubits {qubits}"
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
