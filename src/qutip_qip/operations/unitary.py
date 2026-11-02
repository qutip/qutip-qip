import inspect
from typing import ClassVar

from qutip import Qobj
from qutip_qip.operations.op import Op, is_op, op_factory
from qutip_qip.operations.utils import (
    class_or_instance_method,
    cached_class_or_instance_method,
)
from qutip_qip.utils import convert_type_input_to_sequence
from qutip_qip.typing import Real


class Unitary(Op):
    self_inverse: ClassVar[bool] = False
    is_unitary: ClassVar[bool] = True
    is_clifford: ClassVar[bool] = False
    is_controlled: ClassVar[bool] = False
    config = Op.config | {"self_inverse", "is_clifford", "is_controlled"}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not cls.is_unitary:
            raise TypeError(f"{cls.name} is not unitary, so it has no inverse.")

        if "dtype" not in inspect.signature(cls.compute_qobj).parameters:
            raise TypeError(f"{cls.name}.compute_qobj must accept a dtype parameter.")

        for attr in ("self_inverse", "is_clifford"):
            if not isinstance(inspect.getattr_static(cls, attr), bool | property):
                raise TypeError(f"{cls.name}: {attr} must be a bool.")

        if "inverse" in vars(cls):
            raise TypeError(f"{cls.name}: override compute_inverse, not inverse.")

        if cls.self_inverse is True and "compute_inverse" in vars(cls):
            raise TypeError(
                f"{cls.name} is self_inverse, so it must not define compute_inverse."
            )

    @cached_class_or_instance_method
    def get_qobj(cls_ins, dtype: str = "dense") -> Qobj:
        if not is_op(cls_ins, Unitary):
            raise TypeError("For parametric Unitary, they must be instantiated.")
        return cls_ins.compute_qobj(dtype)

    @class_or_instance_method
    def compute_qobj(cls_ins, dtype: str = "dense") -> Qobj:
        """The op's matrix. Any op may define one, from X to QFT(n); the default is to have none."""
        raise NotImplementedError(
            f"{cls_ins.name} has no matrix, decompose it instead."
        )

    @cached_class_or_instance_method
    def inverse(cls_ins) -> "Unitary | type[Unitary]":
        # To disallow something like RX.inverse()
        if not is_op(cls_ins):
            raise TypeError(
                f"{cls_ins.name} is parametric, build an instance first e.g. {cls_ins.name}(...).inverse()."
            )

        if cls_ins.self_inverse:
            return cls_ins
        return cls_ins.compute_inverse()

    @class_or_instance_method
    def compute_inverse(cls_ins) -> "Unitary | type[Unitary]":
        """Ops with a closed form override this; the default wraps the op in Inverse."""
        from qutip_qip.operations.inverse import _inverse_class

        return _inverse_class(cls_ins)


class AngleParametricOp(Unitary):
    """Parametric gate whose parameters are all real angles."""

    def validate_params(self) -> None:
        for v in self.params:
            if isinstance(v, bool) or not isinstance(
                v, Real
            ):  # bool is an int in Python, but not an angle from our perspective
                raise TypeError(f"{self.name}: parameters must be real, got {v!r}.")


@op_factory
def get_unitary_gate(name: str, U: Qobj) -> type[Op]:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    _name = name
    if not U.isoper or U.dims[0] != U.dims[1]:
        raise ValueError(
            "U must be a square operator with equal input and output dims."
        )

    if not U.isunitary:
        raise ValueError("U must be unitary.")

    class _CustomGate(Unitary):
        name = _name
        num_qreg = len(U.dims[0])
        qreg_dim = tuple(U.dims[0])
        self_inverse = U == U.dag()

        @staticmethod
        def compute_qobj(dtype=U.dtype):
            return U.to(dtype)

    _CustomGate.__name__ = _CustomGate.__qualname__ = _name
    return _CustomGate
