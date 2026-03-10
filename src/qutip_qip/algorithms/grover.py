import numpy as np
from typing import Sequence
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, Z, ControlledGate

__all__ = ["grover", "grover_oracle"]


def grover_oracle(
    search_qubits: int | Sequence[int], marked_states: int | Sequence[int]
) -> QubitCircuit:
    """
    Constructs a Phase Oracle circuit for Grover's algorithm.

    Parameters
    ----------
    search_qubits : int or sequence of int
        The specific qubit indices the oracle acts on.
    marked_states : int or sequence of int
        The states to mark (integer representation).

    Returns
    -------
    QubitCircuit
        The oracle circuit.
    """
    if isinstance(search_qubits, int):
        search_qubits = list(range(search_qubits))

    if isinstance(marked_states, int):
        marked_states = [marked_states]

    n_qubits = len(search_qubits)
    # We need a circuit big enough to hold the highest qubit index
    qc = QubitCircuit(max(search_qubits) + 1)

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
                qc.add_gate("X", targets=search_qubits[i])

        # Change state by Multi Controlled Z Gate
        if n_qubits == 1:
            qc.add_gate("Z", targets=search_qubits[0])
        else:
            ctrls = search_qubits[:-1]
            tgt = search_qubits[-1]

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
                    qc.add_gate("X", targets=search_qubits[i])

    return qc


def grover(
    oracle: QubitCircuit | Gate,
    search_qubits: int | Sequence[int],
    num_solutions: int,
    num_iterations: int | None = None,
    num_qubits: int | None = None,
) -> QubitCircuit:
    """
    Construct the Grover search algorithm's circuit.

    Parameters
    ----------
    oracle : :class:`~.circuit.QubitCircuit` or :class:`~.operations.Gate`
        The oracle that flips the phase of marked states.
    search_qubits : int or sequence of int
        The qubits to run the search on.
    num_solutions : int
        The number of expected solutions M.
    num_iterations : int, optional
        Number of iterations. Defaults to optimal for M solutions.
    num_qubits: int,optional
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
    >>> qc = grover(oracle, search_qubits=[0, 1], num_solutions=1)

    """
    if isinstance(search_qubits, int):
        search_qubits = list(range(search_qubits))

    n_qubits = len(search_qubits)
    search_space_size = 2**n_qubits

    # Validation check for N
    if num_qubits is not None:
        if num_qubits <= 0:
            raise ValueError(
                f"N must be a positive integer, got {num_qubits}."
            )
        min_required = max(search_qubits) + 1
        if num_qubits < min_required:
            raise ValueError(
                f"Total Qubits={num_qubits} is too small. The search qubits {search_qubits} "
                f"require at least {min_required} total qubits."
            )

    # Validation check for num_solutions:
    if num_solutions <= 0:
        raise ValueError("num_solutions must be greater than 0.")
    if num_solutions >= search_space_size:
        raise ValueError(
            "Number of solutions is equal/greater to the search space."
        )

    num_qubits = (max(search_qubits) + 1) if num_qubits is None else num_qubits
    qc = QubitCircuit(num_qubits)

    # Superposition:
    for q in search_qubits:
        qc.add_gate("SNOT", targets=q)

    # Calculate optimal Iterations if none provided:
    if num_iterations is not None and num_iterations < 0:
        raise ValueError(
            f"num_iterations must not be a negative integer, got {num_iterations}."
        )

    if num_iterations is None:
        calc = (np.pi / 4) * np.sqrt(search_space_size / num_solutions)
        num_iterations = int(np.floor(calc))

    # Grover Iterations:
    for _ in range(num_iterations):
        # Oracle
        if isinstance(oracle, QubitCircuit):
            qc.gates.extend(oracle.gates)
        elif isinstance(oracle, Gate):
            if not set(oracle.targets).issubset(search_qubits):
                raise ValueError(
                    f"Oracle gate targets {oracle.targets} are not a subset of algorithm qubits {search_qubits}"
                )
            qc.add_gate(oracle)

        # Diffusion (Inversion about the mean)
        for q in search_qubits:
            qc.add_gate("SNOT", targets=q)

        for q in search_qubits:
            qc.add_gate("X", targets=q)

        # Projection (Multi-Controlled Z)
        if n_qubits == 1:
            qc.add_gate("Z", targets=search_qubits[0])
        else:
            ctrls = search_qubits[:-1]
            tgt = search_qubits[-1]
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

        for q in search_qubits:
            qc.add_gate("X", targets=q)

        for q in search_qubits:
            qc.add_gate("SNOT", targets=q)

    return qc
