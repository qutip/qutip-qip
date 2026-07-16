from qutip_qip.circuit.conditional import Cbnz, Cbz, Conditional, Label


def _is_unconditional_pair(instructions, index):
    """Check for the Cbz/Cbnz pair used to represent an unconditional jump."""
    if index + 1 >= len(instructions):
        return False

    first = instructions[index]
    second = instructions[index + 1]
    return (
        {type(first.op), type(second.op)} == {Cbz, Cbnz}
        and first.op.label is second.op.label
        and first.creg == second.creg
    )


def infer_classical_controls(instructions):
    """Infer the classical controls active at each operation.

    Conditional control flow is represented by branches around a sequence of
    operations instead of metadata on each gate.  This function maps that
    control flow back to the classical wires that control each operation for
    use by circuit renderers.
    """
    instructions = tuple(instructions)
    controls = [set() for _ in instructions]
    label_positions = {
        id(instruction.op): index
        for index, instruction in enumerate(instructions)
        if isinstance(instruction.op, Label)
    }

    unconditional_pairs = []
    unconditional_indices = set()
    for index in range(len(instructions) - 1):
        if _is_unconditional_pair(instructions, index):
            target = label_positions.get(id(instructions[index].op.label))
            if target is not None:
                unconditional_pairs.append((index, target))
                unconditional_indices.update((index, index + 1))

    # A regular forward branch controls every operation it can skip.
    for index, instruction in enumerate(instructions):
        if not isinstance(instruction.op, Conditional):
            continue
        if index in unconditional_indices:
            continue

        target = label_positions.get(id(instruction.op.label))
        if target is None or target <= index:
            continue

        for controlled_index in range(index + 1, target):
            controls[controlled_index].update(instruction.creg)

    # NEQ/GT/LT enter their body through an intermediate label and use an
    # unconditional pair to skip to the final label.  Include every condition
    # evaluated in that prelude on each operation in the body.
    for pair_index, exit_index in unconditional_pairs:
        entry_index = pair_index + 2
        if entry_index >= exit_index:
            continue
        if not isinstance(instructions[entry_index].op, Label):
            continue

        condition_controls = set()
        prelude_index = pair_index - 1
        while prelude_index >= 0 and isinstance(
            instructions[prelude_index].op, Conditional
        ):
            if prelude_index not in unconditional_indices:
                condition_controls.update(instructions[prelude_index].creg)
            prelude_index -= 1

        if not condition_controls:
            continue

        for controlled_index in range(entry_index + 1, exit_index):
            controls[controlled_index].update(condition_controls)

    return [tuple(sorted(control)) for control in controls]
