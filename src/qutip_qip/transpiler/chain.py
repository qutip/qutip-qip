from copy import deepcopy
from qutip_qip.circuit import QubitCircuit


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
    qc_t = deepcopy(qc)
    qc_t.gates = []
    swap_gates = [
        "SWAP",
        "ISWAP",
        "SQRTISWAP",
        "SQRTSWAP",
        "BERKELEY",
        "SWAPalpha",
    ]
    num_qubits = qc.num_qubits

    for gate in qc.gates:
        if gate.name == "CNOT" or gate.name == "CSIGN":
            start = min([gate.targets[0], gate.controls[0]])
            end = max([gate.targets[0], gate.controls[0]])

            if setup == "linear" or (
                setup == "circular" and (end - start) <= num_qubits // 2
            ):
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        # Apply required gate if control and target are
                        # adjacent to each other, provided |control-target|
                        # is even.
                        if end == gate.controls[0]:
                            qc_t.add_gate(
                                gate.name, targets=[i], controls=[i + 1]
                            )
                        else:
                            qc_t.add_gate(
                                gate.name, targets=[i + 1], controls=[i]
                            )

                    elif (
                        start + end - i - i == 2 and (end - start + 1) % 2 == 1
                    ):
                        # Apply a swap between i and its adjacent gate,
                        # then the required gate if and then another swap
                        # if control and target have one qubit between
                        # them, provided |control-target| is odd.
                        qc_t.add_gate("SWAP", targets=[i, i + 1])
                        if end == gate.controls[0]:
                            qc_t.add_gate(
                                gate.name, targets=[i + 1], controls=[i + 2]
                            )
                        else:
                            qc_t.add_gate(
                                gate.name, targets=[i + 2], controls=[i + 1]
                            )
                        qc_t.add_gate("SWAP", [i, i + 1])
                        i += 1

                    else:
                        # Swap the target/s and/or control with their
                        # adjacent qubit to bring them closer.
                        qc_t.add_gate("SWAP", [i, i + 1])
                        qc_t.add_gate(
                            "SWAP", [start + end - i - 1, start + end - i]
                        )
                    i += 1

            elif (end - start) < num_qubits - 1:
                """
                If the resolving has to go backwards, the path is first
                mapped to a separate circuit and then copied back to the
                original circuit.
                """

                temp = QubitCircuit(num_qubits - end + start)
                i = 0
                while i < (num_qubits - end + start):
                    if (
                        num_qubits + start - end - i - i == 1
                        and (num_qubits - end + start + 1) % 2 == 0
                    ):
                        if end == gate.controls[0]:
                            temp.add_gate(
                                gate.name, targets=[i], controls=[i + 1]
                            )
                        else:
                            temp.add_gate(
                                gate.name, targets=[i + 1], controls=[i]
                            )

                    elif (
                        num_qubits + start - end - i - i == 2
                        and (num_qubits - end + start + 1) % 2 == 1
                    ):
                        temp.add_gate("SWAP", targets=[i, i + 1])
                        if end == gate.controls[0]:
                            temp.add_gate(
                                gate.name, targets=[i + 2], controls=[i + 1]
                            )
                        else:
                            temp.add_gate(
                                gate.name, targets=[i + 1], controls=[i + 2]
                            )
                        temp.add_gate("SWAP", [i, i + 1])
                        i += 1

                    else:
                        temp.add_gate("SWAP", [i, i + 1])
                        temp.add_gate(
                            "SWAP",
                            [
                                num_qubits + start - end - i - 1,
                                num_qubits + start - end - i,
                            ],
                        )
                    i += 1

                j = 0
                for gate in temp.gates:
                    if j < num_qubits - end - 2:
                        if gate.name in ["CNOT", "CSIGN"]:
                            qc_t.add_gate(
                                gate.name,
                                end + gate.targets[0],
                                end + gate.controls[0],
                            )
                        else:
                            qc_t.add_gate(
                                gate.name,
                                [end + gate.targets[0], end + gate.targets[1]],
                            )
                    elif j == num_qubits - end - 2:
                        if gate.name in ["CNOT", "CSIGN"]:
                            qc_t.add_gate(
                                gate.name,
                                end + gate.targets[0],
                                (end + gate.controls[0]) % num_qubits,
                            )
                        else:
                            qc_t.add_gate(
                                gate.name,
                                [
                                    end + gate.targets[0],
                                    (end + gate.targets[1]) % num_qubits,
                                ],
                            )
                    else:
                        if gate.name in ["CNOT", "CSIGN"]:
                            qc_t.add_gate(
                                gate.name,
                                (end + gate.targets[0]) % num_qubits,
                                (end + gate.controls[0]) % num_qubits,
                            )
                        else:
                            qc_t.add_gate(
                                gate.name,
                                [
                                    (end + gate.targets[0]) % num_qubits,
                                    (end + gate.targets[1]) % num_qubits,
                                ],
                            )
                    j = j + 1

            elif (end - start) == num_qubits - 1:
                qc_t.add_gate(gate.name, gate.targets, gate.controls)

        elif gate.name in swap_gates:
            start = min([gate.targets[0], gate.targets[1]])
            end = max([gate.targets[0], gate.targets[1]])

            if setup == "linear" or (
                setup == "circular" and (end - start) <= num_qubits // 2
            ):
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        qc_t.add_gate(gate.name, [i, i + 1])
                    elif (start + end - i - i) == 2 and (
                        end - start + 1
                    ) % 2 == 1:
                        qc_t.add_gate("SWAP", [i, i + 1])
                        qc_t.add_gate(gate.name, [i + 1, i + 2])
                        qc_t.add_gate("SWAP", [i, i + 1])
                        i += 1
                    else:
                        qc_t.add_gate("SWAP", [i, i + 1])
                        qc_t.add_gate(
                            "SWAP", [start + end - i - 1, start + end - i]
                        )
                    i += 1

            else:
                temp = QubitCircuit(num_qubits - end + start)
                i = 0
                while i < (num_qubits - end + start):
                    if (
                        num_qubits + start - end - i - i == 1
                        and (num_qubits - end + start + 1) % 2 == 0
                    ):
                        temp.add_gate(gate.name, [i, i + 1])

                    elif (
                        num_qubits + start - end - i - i == 2
                        and (num_qubits - end + start + 1) % 2 == 1
                    ):
                        temp.add_gate("SWAP", [i, i + 1])
                        temp.add_gate(gate.name, [i + 1, i + 2])
                        temp.add_gate("SWAP", [i, i + 1])
                        i += 1

                    else:
                        temp.add_gate("SWAP", [i, i + 1])
                        temp.add_gate(
                            "SWAP",
                            [
                                num_qubits + start - end - i - 1,
                                num_qubits + start - end - i,
                            ],
                        )
                    i += 1

                j = 0
                for gate in temp.gates:
                    if j < num_qubits - end - 2:
                        qc_t.add_gate(
                            gate.name,
                            [end + gate.targets[0], end + gate.targets[1]],
                        )
                    elif j == num_qubits - end - 2:
                        qc_t.add_gate(
                            gate.name,
                            [
                                end + gate.targets[0],
                                (end + gate.targets[1]) % num_qubits,
                            ],
                        )
                    else:
                        qc_t.add_gate(
                            gate.name,
                            [
                                (end + gate.targets[0]) % num_qubits,
                                (end + gate.targets[1]) % num_qubits,
                            ],
                        )
                    j = j + 1

        else:
            # This gate can be general quantum operations
            # such as measurement or global phase.
            qc_t.add_gate(gate)

    return qc_t
