import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import uuid4

from qutip_qip.operations import OpInstruction
from qutip_qip.operations.conditional import Cbz, Cbnz, ClassicalControlCheck, Label
from qutip_qip.typing import Int
from qutip_qip.utils import check_limit


@dataclass(frozen=True, slots=True)
class Bloq:
    uuid: str = field(default_factory=lambda: uuid4().hex)
    qreg_dim: tuple[Int, ...] = ()
    aux_qreg_dim: tuple[Int, ...] = ()

    @property
    def num_qreg(self) -> int:
        return len(self.qreg_dim)

    @property
    def num_aux_qreg(self) -> int:
        return len(self.aux_qreg_dim)

    num_creg: Int = 0
    num_aux_creg: Int = 0
    global_phase: float = 0.0
    instructions: tuple[OpInstruction, ...] = ()

    # Op and Bloq are kept separate and qreg_dim, qreg_count, creg_count etc. have been copied here.
    # The only linking between them happens via the BloqRepository.


class BloqBuilder:
    __slots__ = (
        "_qreg_dim",
        "_aux_qreg_dim",
        "_num_creg",
        "_num_aux_creg",
        "_global_phase",
        "_op_instructions",
    )

    def __init__(
        self, num_qreg: int, num_creg: int = 0, qreg_dim: tuple[int, ...] = ()
    ) -> None:
        if num_qreg < 0:
            raise ValueError("num_qreg must be greater than or equal to 0.")

        if num_creg < 0:
            raise ValueError("num_creg must be greater than or equal to 0.")

        if len(qreg_dim) and len(qreg_dim) != num_qreg:
            raise ValueError(
                f"Length of qreg_dim={qreg_dim} must be equal to num_qreg={num_qreg}"
            )

        self._qreg_dim = qreg_dim
        if len(self._qreg_dim) == 0:
            self._qreg_dim = (2,) * num_qreg

        self._aux_qreg_dim = []
        self._num_creg = num_creg
        self._num_aux_creg = 0
        self._global_phase = 0.0
        self._op_instructions = []

    @property
    def num_qreg(self) -> int:
        return len(self._qreg_dim)

    @property
    def num_aux_qreg(self) -> int:
        return len(self._aux_qreg_dim)

    @property
    def num_creg(self) -> int:
        return self._num_creg

    @property
    def num_aux_creg(self) -> int:
        return self._num_aux_creg

    @property
    def instructions(self) -> tuple[OpInstruction, ...]:
        return tuple(self._op_instructions)

    @property
    def global_phase(self) -> float:
        return self._global_phase

    def add_global_phase(self, phase: float) -> None:
        self._global_phase += phase
        self._global_phase %= 2 * math.pi

    def add_aux_qreg(self, count: Int = 1, dim: Int = 2) -> None:
        if not (isinstance(count, Int) and count > 0):
            raise TypeError(f"count must be of type int, got {count}")

        if not (isinstance(dim, Int) and dim > 0):
            raise TypeError(f"dim must be of type int, got {dim}")

        self._aux_qreg_dim.extend([dim] * count)

    def add_aux_creg(self, count: Int = 1) -> None:
        if not (isinstance(count, Int) and count > 0):
            raise TypeError(f"count must be of type int, got {count}")
        self._num_aux_creg += count

    @property
    def qreg(self) -> tuple[int, ...]:
        return tuple(range(self.num_qreg))

    @property
    def aux_qreg(self) -> tuple[int, ...]:
        return tuple(range(self.num_qreg, self.num_qreg + self.num_aux_qreg))

    @property
    def creg(self) -> tuple[int, ...]:
        return tuple(range(self.num_creg))

    @property
    def aux_creg(self) -> tuple[int, ...]:
        return tuple(range(self.num_creg, self.num_creg + self.num_aux_creg))

    def add_op(self, op, qreg=(), creg=(), style: dict = None) -> None:
        # Type checking is handled internally within OpInstruction
        # We just check each element of qreg, creg are within the limit
        if isinstance(qreg, Int):
            qreg = [qreg]
        if isinstance(creg, Int):
            creg = [creg]

        check_limit("qreg", qreg, 0, self.num_qreg + self.num_aux_qreg - 1)
        check_limit("creg", creg, 0, self.num_creg + self.num_aux_creg - 1)

        self._op_instructions.append(
            OpInstruction(op=op, qreg=tuple(qreg), creg=tuple(creg), style=style)
        )

    @contextmanager
    def if_test(self, creg, value: Int, check: ClassicalControlCheck = "EQ") -> None:
        if isinstance(creg, Int):
            creg = [creg]

        if check == ClassicalControlCheck.GTE:
            check = ClassicalControlCheck.GT
            value -= 1

        if check == ClassicalControlCheck.LTE:
            check = ClassicalControlCheck.LT
            value += 1

        check_limit("creg", creg, 0, self.num_creg + self.num_aux_creg - 1)
        num_bits = len(creg)
        lg_check_value = 2**num_bits
        label = Label(uuid4().hex)

        if check == ClassicalControlCheck.EQ:
            if (value < 0) or (value >= lg_check_value):
                return  # Useless if_test condition

            else:
                for index, cbit in enumerate(creg):
                    if (value >> (num_bits - 1 - index)) & 1 == 1:  # MSB first ordering
                        # If does not match for cbit_value=1, then branch to label (don't execute the conditional if)
                        self.add_op(Cbz(label=label), creg=cbit)
                    else:
                        # If does not match for cbit_value=0, then branch to label
                        self.add_op(Cbnz(label=label), creg=cbit)

        elif check == ClassicalControlCheck.NEQ:
            neq_label = Label(uuid4().hex)
            if (value < 0) or (value >= lg_check_value):
                # This is an unconditional jump essentially
                self.add_op(Cbz(label=neq_label), creg=0)
                self.add_op(Cbnz(label=neq_label), creg=0)

            else:
                for index, cbit in enumerate(creg):
                    target_bit_value = (value >> (num_bits - 1 - index)) & 1

                    if target_bit_value == 1:
                        # If a mismatch match for cbit_value=1, then branch to neqlabel
                        self.add_op(Cbz(label=neq_label), creg=cbit)
                    else:
                        self.add_op(Cbnz(label=neq_label), creg=cbit)

                # This will only execute if non of the earlier conditional branching executes.
                # means NEQ is FALSE. We must skip the conditional block.

                # This must be preferably replaced Jump statement (unconditional)
                self.add_op(Cbz(label=label), creg=0)
                self.add_op(Cbnz(label=label), creg=0)

            # Successful entry point for the NEQ condition
            self.add_op(neq_label)

        elif check == ClassicalControlCheck.GT:
            # Check for redundant conditions
            if value >= lg_check_value:  # Never true
                return

            elif value >= 0:  # for value less than 0, condition is always true
                gt_label = Label(uuid4().hex)

                for index, cbit in enumerate(creg):
                    target_bit_value = (value >> (num_bits - 1 - index)) & 1

                    # We break at first point of discontinuity (but to different labels)
                    if target_bit_value == 1:
                        self.add_op(Cbz(label=label), creg=cbit)
                    else:
                        self.add_op(Cbnz(label=gt_label), creg=cbit)

                # If execution falls through the entire loop without jumping,
                # it means every single bit matched exactly.
                # self.add_op(Jump(label=label))
                self.add_op(Cbz(label=label), creg=0)
                self.add_op(Cbnz(label=label), creg=0)

                # Entry point for the GT conditional block
                self.add_op(gt_label)

        elif check == ClassicalControlCheck.LT:
            # Check for redundant conditions
            if value <= 0:  # Never true
                return

            elif (
                value < lg_check_value
            ):  # for value larger than 2^m, condition is always true
                lt_label = Label(uuid4().hex)
                for index, cbit in enumerate(creg):
                    target_bit_value = (value >> (num_bits - 1 - index)) & 1

                    # We break at first point of discontinuity (but to different labels)
                    if target_bit_value == 1:
                        self.add_op(Cbz(label=lt_label), creg=cbit)
                    else:
                        self.add_op(Cbnz(label=label), creg=cbit)

                # If execution falls through the entire loop without jumping,
                # it means every single bit matched exactly.
                # self.add_op(Jump(label=label))
                self.add_op(Cbz(label=label), creg=0)
                self.add_op(Cbnz(label=label), creg=0)

                # Entry point for the GT conditional block
                self.add_op(lt_label)

        else:
            raise ValueError(f"Invalid check {check}")

        try:
            yield
        finally:
            self.add_op(label)

    def build(self) -> Bloq:
        return Bloq(
            qreg_dim=self._qreg_dim,
            aux_qreg_dim=tuple(self._aux_qreg_dim),
            num_creg=self.num_creg,
            num_aux_creg=self.num_aux_creg,
            global_phase=self.global_phase,
            instructions=self.instructions,
        )


class BloqRepository:
    pass
