import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4
from typing import Self

from qutip_qip.operations.conditional import Cbz, Cbnz, ClassicalControlCheck, Label
from qutip_qip.operations.op import Op, OpInstruction
from qutip_qip.typing import Int
from qutip_qip.utils import check_limit


@dataclass(frozen=True, slots=True)
class Bloq:
    qreg_dim: tuple[Int, ...] = ()
    aux_qreg_dim: tuple[Int, ...] = ()

    @property
    def num_qreg(self: Self) -> int:
        return len(self.qreg_dim)

    @property
    def num_aux_qreg(self: Self) -> int:
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
        "_clean_aux_qreg",
        "_global_phase",
        "_op_instructions",
    )

    # Note: We don't require an ancilla like clean mechanism for aux_creg because
    # classical registers can be overwritten in measurement. So a creg is reusable
    # as soon as its bloq ends.

    def __init__(
        self: Self, num_qreg: int, num_creg: int = 0, qreg_dim: tuple[int, ...] = ()
    ) -> None:
        if num_qreg < 0:
            raise ValueError("num_qreg must be greater than or equal to 0.")

        if num_creg < 0:
            raise ValueError("num_creg must be greater than or equal to 0.")

        if len(qreg_dim) and len(qreg_dim) != num_qreg:
            raise ValueError(
                f"Length of qreg_dim={qreg_dim} must be equal to num_qreg={num_qreg}"
            )

        self._qreg_dim: tuple[int, ...] = qreg_dim
        if len(self._qreg_dim) == 0:
            self._qreg_dim = (2,) * num_qreg

        self._aux_qreg_dim: list[int] = []
        self._num_creg: int = num_creg
        self._num_aux_creg: int = 0
        self._clean_aux_qreg: defaultdict[int, list[int]] = defaultdict(list)
        self._global_phase: float = 0.0
        self._op_instructions: list[OpInstruction] = []

    @property
    def qreg_dim(self: Self) -> tuple[int, ...]:
        return self._qreg_dim

    @property
    def num_qreg(self: Self) -> int:
        return len(self._qreg_dim)

    @property
    def num_aux_qreg(self: Self) -> int:
        return len(self._aux_qreg_dim)

    @property
    def num_creg(self: Self) -> int:
        return self._num_creg

    @property
    def num_aux_creg(self: Self) -> int:
        return self._num_aux_creg

    @property
    def instructions(self: Self) -> tuple[OpInstruction, ...]:
        return tuple(self._op_instructions)

    @property
    def global_phase(self: Self) -> float:
        return self._global_phase

    def add_global_phase(self: Self, phase: float) -> None:
        self._global_phase += phase
        self._global_phase %= 2 * math.pi

    def add_aux_qreg(self: Self, count: Int = 1, dim: Int = 2) -> tuple[int, ...]:
        if not (isinstance(count, Int) and count > 0):
            raise TypeError(f"count must be of type int, got {count}")

        if not (isinstance(dim, Int) and dim > 0):
            raise TypeError(f"dim must be of type int, got {dim}")

        clean = self._clean_aux_qreg[dim]
        return tuple(
            clean.pop() if clean else self._new_aux_qreg(dim) for _ in range(count)
        )

    def _new_aux_qreg(self: Self, dim: Int) -> int:
        self._aux_qreg_dim.append(dim)
        return self.aux_qreg[-1]

    def release_aux_qreg(self: Self, qreg: tuple[Int, ...]) -> None:
        for q in qreg:
            self._clean_aux_qreg[self._aux_qreg_dim[q - self.num_qreg]].append(q)

    def add_aux_creg(self: Self, count: Int = 1) -> tuple[int, ...]:
        if not (isinstance(count, Int) and count > 0):
            raise TypeError(f"count must be of type int, got {count}")

        start = self.num_creg + self.num_aux_creg
        self._num_aux_creg += count
        return tuple(range(start, start + count))

    @property
    def qreg(self: Self) -> tuple[int, ...]:
        return tuple(range(self.num_qreg))

    @property
    def creg(self: Self) -> tuple[int, ...]:
        return tuple(range(self.num_creg))

    @property
    def aux_qreg(self: Self) -> tuple[int, ...]:
        return tuple(range(self.num_qreg, self.num_qreg + self.num_aux_qreg))

    @property
    def aux_creg(self: Self) -> tuple[int, ...]:
        return tuple(range(self.num_creg, self.num_creg + self.num_aux_creg))

    def add(
        self: Self,
        op: Op,
        qreg: tuple[Int, ...] = (),
        creg: tuple[Int, ...] = (),
        style: dict = None,
    ) -> None:
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

    def _jump(self: Self, label: Label):
        # Unconditional jump
        self.add(Cbz(label), creg=0)
        self.add(Cbnz(label), creg=0)

    @contextmanager
    def if_test(
        self: Self,
        creg: Int | tuple[Int, ...],
        value: Int,
        check: ClassicalControlCheck = "EQ",
    ):
        if isinstance(creg, Int):
            creg = [creg]

        num_bits = len(creg)
        check_limit("creg", creg, 0, self.num_creg + self.num_aux_creg - 1)
        check_limit("value", [value], 0, 2**num_bits - 1)
        label = Label(uuid4().hex)

        if check == ClassicalControlCheck.GTE:
            check = ClassicalControlCheck.GT
            value -= 1

        if check == ClassicalControlCheck.LTE:
            check = ClassicalControlCheck.LT
            value += 1

        if check == ClassicalControlCheck.EQ:
            for index, cbit in enumerate(creg):
                if (value >> (num_bits - 1 - index)) & 1 == 1:  # MSB first ordering
                    # If does not match for cbit_value=1, then branch to label (don't execute the conditional if)
                    self.add(Cbz(label=label), creg=cbit)
                else:
                    # If does not match for cbit_value=0, then branch to label
                    self.add(Cbnz(label=label), creg=cbit)

        elif check == ClassicalControlCheck.NEQ:
            neq_label = Label(uuid4().hex)
            for index, cbit in enumerate(creg):
                target_bit_value = (value >> (num_bits - 1 - index)) & 1

                if target_bit_value == 1:
                    # If a mismatch match for cbit_value=1, then branch to neqlabel
                    self.add(Cbz(label=neq_label), creg=cbit)
                else:
                    self.add(Cbnz(label=neq_label), creg=cbit)

            # Every bit matched, so NEQ is false i.e. skip the block.
            self._jump(label)

            # Successful entry point for the NEQ condition
            self.add(neq_label)

        elif check == ClassicalControlCheck.GT:
            if value >= 0:
                gt_label = Label(uuid4().hex)

                for index, cbit in enumerate(creg):
                    target_bit_value = (value >> (num_bits - 1 - index)) & 1

                    # We break at first point of discontinuity (but to different labels)
                    if target_bit_value == 1:
                        self.add(Cbz(label=label), creg=cbit)
                    else:
                        self.add(Cbnz(label=gt_label), creg=cbit)

                # If execution falls through the entire loop without jumping,
                # it means every single bit matched exactly.
                self._jump(label)

                # Entry point for the GT conditional block
                self.add(gt_label)

        elif check == ClassicalControlCheck.LT:
            if value < 2**num_bits:
                lt_label = Label(uuid4().hex)
                for index, cbit in enumerate(creg):
                    target_bit_value = (value >> (num_bits - 1 - index)) & 1

                    # We break at first point of discontinuity (but to different labels)
                    if target_bit_value == 1:
                        self.add(Cbz(label=lt_label), creg=cbit)
                    else:
                        self.add(Cbnz(label=label), creg=cbit)

                # If execution falls through the entire loop without jumping,
                # it means every single bit matched exactly.
                self._jump(label)

                # Entry point for the GT conditional block
                self.add(lt_label)

        else:
            raise ValueError(f"Invalid check {check}")

        try:
            yield
        finally:
            self.add(label)

    def to_bloq(self: Self) -> Bloq:
        return Bloq(
            qreg_dim=self._qreg_dim,
            aux_qreg_dim=tuple(self._aux_qreg_dim),
            num_creg=self.num_creg,
            num_aux_creg=self.num_aux_creg,
            global_phase=self.global_phase,
            instructions=tuple(self._op_instructions),
        )
