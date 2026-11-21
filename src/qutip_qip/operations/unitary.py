import inspect
from abc import abstractmethod
from typing import ClassVar

from qutip import Qobj
from qutip_qip.operations.op import Op, op_factory
from qutip_qip.operations.utils import cached_method
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

        if cls.self_inverse is True and "inverse" in vars(cls):
            raise TypeError(
                f"{cls.name} is self_inverse, so it must not define inverse."
            )

    @cached_method
    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(dtype)

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        """The op's matrix. Any op may define one, from X to QFT(n); the default is to have none."""
        raise NotImplementedError(f"{self.name} has no matrix, decompose it instead.")

    def inverse(self) -> "Unitary | None":
        """Ops with a closed form should overwrite this."""
        return self if self.self_inverse else None

    @property
    def has_matrix(self) -> bool:
        return type(self).compute_qobj is not Unitary.compute_qobj


class AngleUnitary(Unitary):
    """Parametric gate whose parameters are all real angles."""

    def validate_params(self) -> None:
        for v in self.params:
            if isinstance(v, bool) or not isinstance(
                v, Real
            ):  # bool is an int in Python, but not an angle from our perspective
                raise TypeError(f"{self.name}: parameters must be real, got {v!r}.")


@op_factory
def get_unitary_gate(name: str, U: Qobj) -> Unitary:
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
        __qualname__ = _name
        name = _name
        num_qreg = len(U.dims[0])
        qreg_dim = tuple(U.dims[0])
        self_inverse = U == U.dag()

        def compute_qobj(self, dtype=U.dtype):
            return U.to(dtype)

    return _CustomGate


class Modifier(Unitary):
    @property
    @abstractmethod
    def target(self) -> Unitary | type[Unitary]: ...

    @property
    def has_matrix(self) -> bool:
        return self.target.has_matrix
