import inspect
from functools import lru_cache

from qutip import Qobj
from qutip_qip.operations.modifier import Modifier
from qutip_qip.operations.op import Unitary, is_op, op_factory
from qutip_qip.operations.utils import class_or_instance_method


class InverseUnitary(Modifier):
    _derived = {
        "num_qreg": lambda c, t: t.num_qreg,
        "qreg_dim": lambda c, t: t.qreg_dim,
        "num_creg": lambda c, t: t.num_creg,
        "self_inverse": lambda c, t: t.self_inverse,
        "is_clifford": lambda c, t: t.is_clifford,
    }

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            cls._key = ("inverse", cls.target)
        super().__init_subclass__()

    @class_or_instance_method
    def inverse(cls_ins) -> Unitary | type[Unitary]:
        return cls_ins.target_unitary()

    @class_or_instance_method
    def compute_qobj(cls_ins, dtype="dense") -> Qobj:
        return cls_ins.target_unitary().get_qobj(dtype).dag()


@op_factory
@lru_cache(maxsize=128)
def inverse(
    op: Unitary | type[Unitary], name: str | None = None
) -> type[InverseUnitary]:
    if not is_op(op, Unitary):
        raise TypeError(f"inverse() takes an op class, e.g. inverse(RX), got {op!r}.")

    _name, _target = name or f"{op.name}dag", op

    class _Inverse(InverseUnitary):
        name = _name
        target = _target

    _Inverse.__name__ = _Inverse.__qualname__ = _name
    return _Inverse
