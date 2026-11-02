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
