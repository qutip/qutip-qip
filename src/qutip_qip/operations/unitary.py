import copy
import inspect
import weakref
from abc import abstractmethod
from dataclasses import fields
from typing import ClassVar, Self

from qutip import Qobj
from qutip_qip.operations.op import Op, get_op_class, op_factory
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


type UnitaryFamily = Unitary | type[Unitary]


def get_unitary_class(op: UnitaryFamily) -> type[Unitary]:
    """
    Return the class of a unitary, e.g. RX for both RX and RX(0.5).

    Raises
    ------
    TypeError
        If op is neither a unitary value nor a unitary family.
    """
    family = op if isinstance(op, type) else type(op)
    if not issubclass(family, Unitary) or inspect.isabstract(family):
        raise TypeError(
            f"Expected a unitary value like X or a family like RX, got {op!r}."
        )
    return family


def _from_target(config_name: str) -> property:
    """A config value that a modifier takes from its target unchanged."""
    return property(lambda self: getattr(self.target, config_name))


class Modifier(Unitary):
    """
    Base class of the ops that are built from another op: :class:`.Inverse`,
    :class:`.Controlled` and :class:`.Power`. The op they are built from is
    called the target.

    A modifier can be applied to a value or to a family, and the target is
    of the same kind as the result.

    >>> Inverse(QFT(3)).target
    QFT(3)
    >>> Inverse(QFT).target is QFT
    True

    A family made by a modifier takes the same parameters as its target, and
    filling them in before or after makes no difference.

    >>> Controlled(RX)(0.5) == Controlled(RX(0.5))
    True

    Anything else a modifier is given, such as the k in Power(op, k),
    is a setting. Settings are stored on the class, so Power(RX, 2) and
    Power(RX, 3) are two classes, and each takes theta like RX.

    Attributes
    ----------
    target : Unitary or type[Unitary]
        The op being modified.

    Notes
    -----
    To write a new modifier, subclass Modifier and

    * list the names of its settings in _settings;
    * define _apply_modifier, which is what calling the modifier runs;
    * write its config (name, num_qreg, ...) as properties that read
      only self.target and the settings, because they are also run on
      the class. num_qreg, qreg_dim, num_creg, self_inverse
      and is_clifford are taken from the target unless written;
    * define compute_qobj and inverse as for any other unitary.
    """

    config = Unitary.config | {"target"}

    # Names of the settings, in the order the modifier takes them, e.g. ("num_ctrl_qreg", "ctrl_value", "ctrl_dim") in Controlled.
    _settings: ClassVar[tuple[str, ...]] = ()

    # The values of those settings for one family, e.g. (3, 5, (2, 2, 2)) for Controlled(X, num_ctrl_qreg=3, ctrl_val=5).
    _setting_values: ClassVar[tuple]

    # Inverse, Controlled or Power, whichever one the class comes from.
    _modifier: ClassVar[type["Modifier"]]

    _cache: ClassVar[weakref.WeakValueDictionary]
    num_qreg = _from_target("num_qreg")
    qreg_dim = _from_target("qreg_dim")
    num_creg = _from_target("num_creg")
    self_inverse = _from_target("self_inverse")
    is_clifford = _from_target("is_clifford")

    @property
    @abstractmethod
    def target(self) -> UnitaryFamily:
        """The Unitary being modified."""

    def __init_subclass__(cls) -> None:
        if Modifier in cls.__bases__:
            cls._modifier = cls
            cls._cache = weakref.WeakValueDictionary()

        target = cls.target
        if isinstance(target, property):
            super().__init_subclass__()
            return

        if isinstance(target, type):
            # A family takes the same parameters as its target,
            # e.g. Controlled(RX) takes theta because RX does.
            parameters = fields(target)
            cls.__annotations__ = {
                parameter.name: parameter.type for parameter in parameters
            } | cls.__annotations__
            for parameter in parameters:
                setattr(cls, parameter.name, copy.copy(parameter))

        # If the config of the target is fixed (RX always acts on 1 qubit), the property can
        # be run on the class itself, so we can work the value out of that property once and store
        # it. If it depends on the parameters (QFT needs n first), keep the
        # property and every value works it out for itself.
        needs_parameters = isinstance(target, type) and target.is_derived
        for config_name in cls.config - vars(cls).keys():
            config_value = getattr(cls, config_name)
            if isinstance(config_value, property):
                if not needs_parameters:
                    config_value = config_value.fget(cls)
                setattr(cls, config_name, config_value)

        super().__init_subclass__()

        cls._setting_values = tuple(getattr(cls, name) for name in cls._settings)
        cls._cache.setdefault((target, *cls._setting_values), cls)

    def __new__(cls, *args, **kwargs) -> UnitaryFamily:
        if cls is cls._modifier:
            return cls._apply_modifier(*args, **kwargs)

        if not cls.is_parametric:
            return super().__new__(cls)

        target = cls.target(*args, **kwargs)
        if (replacement := cls._find_simpler_form(target)) is not None:
            return replacement

        value = super().__new__(cls)
        # Ops are frozen, so a plain value.target = target would raise.
        object.__setattr__(value, "target", target)
        return value

    @classmethod
    @abstractmethod
    def _apply_modifier(cls, op: UnitaryFamily, *settings) -> UnitaryFamily:
        """
        What calling the modifier does: Power(op, 3) runs Power._apply_modifier(op, 3).

        Each modifier must write its own. It checks its arguments, returns a
        simpler Unitary if it knows one (e.g. Inverse(T) is Tdag) or otherwise returns cls._make_modifier(...).
        """

    @classmethod
    def _find_simpler_form(cls, target: Unitary) -> Unitary | None:
        if type(target) is not cls.target:
            return cls._modifier(target, *cls._setting_values)
        return None

    @classmethod
    def _make_modifier(
        cls,
        op: UnitaryFamily,
        class_name: str,
        **class_attributes,
    ) -> Self | type[Self]:
        """
        Return op with the modifier around it, without looking for a simpler
        form.

        Parameters
        ----------
        op : Unitary or type[Unitary]
            The target. A family gives back a family, a value gives back a
            value.
        class_name : str
            The name of the new class, used only if it does not exist yet.
        """
        # RX(0.5) shares the class made for RX.
        # X has no parameters, so X itself is the target.
        target = get_op_class(op)
        settings = tuple(class_attributes[name] for name in cls._settings)

        family = cls._cache.get((target, *settings))
        if family is None:
            # The new class cannot be imported by name, so pickle saves the call that makes it.
            namespace = {
                "target": target,
                "_factory": (cls, (target, *settings)),
                "__module__": cls.__module__,
            }

            # A class without parameters gives back its one value when it is
            # created, like the class statement of X does.
            created = type(cls)(class_name, (cls,), namespace | class_attributes)
            family = get_unitary_class(created)

        if not family.is_parametric:
            return family._value  # e.g. CX
        if isinstance(op, type):
            return family
        return family(*op.params)

    @property
    def has_matrix(self) -> bool:
        return self.target.has_matrix

    @property
    def _key(self) -> tuple:
        # Used for == and hash. Two modified ops are equal if they apply the
        # same modifier, with the same settings, to equal targets.
        return (self._modifier, self.target, *self._setting_values)
