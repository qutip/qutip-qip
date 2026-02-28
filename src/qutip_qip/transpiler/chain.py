from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.std import SWAP


def to_chain_structure(qc: QubitCircuit, setup="linear"):
    """
    Method to resolve 2 qubit gates with non-adjacent control/s or target/s
    in terms of gates with adjacent interactions for linear/circular spin
    chain system.

    Parameters
    ----------
    qc : :class:`.QubitCircuit`
        The circular spin chain circuit to be resolved

    setup: Boolean
        Linear of Circular spin chain setup

    Returns
    -------
    qc : :class:`.QubitCircuit`
        Returns QubitCircuit of resolved gates for the qubit circuit in the
        desired basis.
    """
    # FIXME This huge block has been here for a long time.
    # It could be moved to the new compiler section and carefully
    # splitted into smaller peaces.
    N = qc.num_qubits
    qc_t = QubitCircuit(N)
    qc_t.add_global_phase(qc.global_phase)
    swap_gates = [
        SWAP,
        "ISWAP",
        "SQRTISWAP",
        "SQRTSWAP",
        "BERKELEY",
        "SWAPalpha",
    ]

    for circ_instruction in qc.instructions:
        gate = circ_instruction.operation
        controls = circ_instruction.controls
        targets = circ_instruction.targets

        if gate.name in ["CNOT", "CX", "CSIGN", "CZ"]:
            start = min([targets[0], controls[0]])
            end = max([targets[0], controls[0]])

            if setup == "linear" or (
                setup == "circular" and (end - start) <= N // 2
            ):
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        # Apply required gate if control and target are
                        # adjacent to each other, provided |control-target|
                        # is even.
                        if end == controls[0]:
                            qc_t.add_gate(gate, targets=[i], controls=[i + 1])
                        else:
                            qc_t.add_gate(gate, targets=[i + 1], controls=[i])

                    elif (
                        start + end - i - i == 2 and (end - start + 1) % 2 == 1
                    ):
                        # Apply a swap between i and its adjacent gate,
                        # then the required gate if and then another swap
                        # if control and target have one qubit between
                        # them, provided |control-target| is odd.
                        qc_t.add_gate(SWAP, targets=[i, i + 1])
                        if end == controls[0]:
                            qc_t.add_gate(
                                gate, targets=[i + 1], controls=[i + 2]
                            )
                        else:
                            qc_t.add_gate(
                                gate, targets=[i + 2], controls=[i + 1]
                            )
                        qc_t.add_gate(SWAP, targets=[i, i + 1])
                        i += 1

                    else:
                        # Swap the target/s and/or control with their
                        # adjacent qubit to bring them closer.
                        qc_t.add_gate(SWAP, targets=[i, i + 1])
                        qc_t.add_gate(
                            SWAP,
                            targets=[start + end - i - 1, start + end - i],
                        )
                    i += 1

            elif (end - start) < N - 1:
                """
                If the resolving has to go backwards, the path is first
                mapped to a separate circuit and then copied back to the
                original circuit.
                """

                temp = QubitCircuit(N - end + start + 1)
                i = 0
                while i < (N - end + start):
                    if (
                        N + start - end - i - i == 1
                        and (N - end + start + 1) % 2 == 0
                    ):
                        if end == controls[0]:
                            temp.add_gate(gate, targets=[i], controls=[i + 1])
                        else:
                            temp.add_gate(gate, targets=[i + 1], controls=[i])

                    elif (
                        N + start - end - i - i == 2
                        and (N - end + start + 1) % 2 == 1
                    ):
                        temp.add_gate(SWAP, targets=[i, i + 1])
                        if end == controls[0]:
                            temp.add_gate(
                                gate, targets=[i + 2], controls=[i + 1]
                            )
                        else:
                            temp.add_gate(
                                gate, targets=[i + 1], controls=[i + 2]
                            )
                        temp.add_gate(SWAP, targets=[i, i + 1])
                        i += 1

                    else:
                        temp.add_gate(SWAP, targets=[i, i + 1])
                        temp.add_gate(
                            SWAP,
                            targets=[
                                N + start - end - i - 1,
                                N + start - end - i,
                            ],
                        )
                    i += 1

                j = 0

                for circ_instruction in temp.instructions:
                    gate = circ_instruction.operation
                    targets = circ_instruction.targets
                    controls = circ_instruction.controls

                    if j < N - end - 2:
                        if gate.name in ["CNOT", "CX" "CSIGN", "CZ"]:
                            qc_t.add_gate(
                                gate,
                                targets=end + targets[0],
                                controls=end + controls[0],
                            )
                        else:
                            qc_t.add_gate(
                                gate,
                                targets=[
                                    (end + targets[0]) % N,
                                    (end + targets[1]) % N,
                                ],
                            )
                    elif j == N - end - 2:
                        if gate.name in ["CNOT", "CX", "CSIGN", "CZ"]:
                            qc_t.add_gate(
                                gate,
                                targets=end + targets[0],
                                controls=(end + controls[0]) % N,
                            )
                        else:
                            qc_t.add_gate(
                                gate,
                                targets=[
                                    end + targets[0],
                                    (end + targets[1]) % N,
                                ],
                            )
                    else:
                        if gate.name in ["CNOT", "CX", "CSIGN", "CZ"]:
                            qc_t.add_gate(
                                gate,
                                targets=(end + targets[0]) % N,
                                controls=(end + controls[0]) % N,
                            )
                        else:
                            qc_t.add_gate(
                                gate,
                                targets=[
                                    (end + targets[0]) % N,
                                    (end + targets[1]) % N,
                                ],
                            )
                    j = j + 1

            elif (end - start) == N - 1:
                qc_t.add_gate(gate, targets=targets, controls=controls)

        elif gate.name in swap_gates:
            start = min(targets)
            end = max(targets)

            if setup == "linear" or (
                setup == "circular" and (end - start) <= N // 2
            ):
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        qc_t.add_gate(gate, targets=[i, i + 1])

                    elif (start + end - i - i) == 2 and (
                        end - start + 1
                    ) % 2 == 1:
                        qc_t.add_gate(SWAP, targets=[i, i + 1])
                        qc_t.add_gate(gate, targets=[i + 1, i + 2])
                        qc_t.add_gate(SWAP, targets=[i, i + 1])
                        i += 1

                    else:
                        qc_t.add_gate(SWAP, targets=[i, i + 1])
                        qc_t.add_gate(
                            SWAP,
                            targets=[start + end - i - 1, start + end - i],
                        )
                    i += 1

            else:
                temp = QubitCircuit(N - end + start + 1)
                i = 0
                while i < (N - end + start):
                    if (
                        N + start - end - i - i == 1
                        and (N - end + start + 1) % 2 == 0
                    ):
                        temp.add_gate(gate, targets=[i, i + 1])

                    elif (
                        N + start - end - i - i == 2
                        and (N - end + start + 1) % 2 == 1
                    ):
                        temp.add_gate(SWAP, targets=[i, i + 1])
                        temp.add_gate(gate, targets=[i + 1, i + 2])
                        temp.add_gate(SWAP, targets=[i, i + 1])
                        i += 1

                    else:
                        temp.add_gate(SWAP, targets=[i, i + 1])
                        temp.add_gate(
                            SWAP,
                            targets=[
                                N + start - end - i - 1,
                                N + start - end - i,
                            ],
                        )
                    i += 1

                j = 0

                for circ_instruction in temp.instructions:
                    gate = circ_instruction.operation
                    targets = circ_instruction.targets

                    if j < N - end - 2:
                        qc_t.add_gate(
                            gate,
                            targets=[end + targets[0], end + targets[1]],
                        )
                    elif j == N - end - 2:
                        qc_t.add_gate(
                            gate,
                            targets=[
                                (end + targets[0]) % N,
                                (end + targets[1]) % N,
                            ],
                        )
                    else:
                        qc_t.add_gate(
                            gate,
                            targets=[
                                (end + targets[0]) % N,
                                (end + targets[1]) % N,
                            ],
                        )
                    j = j + 1

        else:
            # This gate can be general quantum operations
            # such as measurement or global phase.
            qc_t.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
                style=circ_instruction.style,
            )

    return qc_t
