import copy
import inspect
import warnings
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import fields
from typing import ClassVar

from qutip_qip.operations.op import Unitary
from qutip_qip.operations.utils import cached_class_or_instance_method


class Modifier(Unitary):
    """An op defined by a target op and a rule applied to it, e.g. Controlled, Inverse."""

    config = Unitary.config | {"target"}

    # Abstract class attribute i.e. subclasses need to define target Op for e.g. RX
    @property
    @abstractmethod
    def target(self) -> type[Unitary]: ...

    _derived: ClassVar[dict[str, Callable]] = {}

    def __init_subclass__(cls) -> None:
        if clash := vars(cls).keys() & cls._derived.keys():
            warnings.warn(
                f"{cls.__name__}: cannot set {', '.join(clash)}, derived from the target.",
                SyntaxWarning,
            )

        if not inspect.isabstract(cls):
            target = cls.target
            if not (
                isinstance(target, type)
                and issubclass(target, Unitary)
                and not inspect.isabstract(target)
            ):
                raise TypeError(
                    f"{cls.__name__}: target must be a concrete Unitary, got {target!r}."
                )
            if not target.is_unitary:
                raise TypeError(f"{cls.__name__}: target {target.name} is not unitary.")

            for attr, rule in cls._derived.items():
                if target.is_derived:
                    setattr(
                        cls,
                        attr,
                        property(
                            lambda self, rule=rule: rule(self, self.target_unitary())
                        ),
                    )
                else:
                    setattr(cls, attr, rule(cls, target))

            # The target Op's parameters e.g. theta in RX, become parameter in a Controlled Op
            cls.__annotations__ = {f.name: f.type for f in fields(target)} | dict(
                cls.__annotations__
            )
            for f in fields(target):
                setattr(cls, f.name, copy.copy(f))

        super().__init_subclass__()

    def validate_params(self) -> None:
        self.target_unitary()  # building the target runs its own validation

    @cached_class_or_instance_method
    def target_unitary(cls_ins) -> Unitary | type[Unitary]:
        return (
            cls_ins.target
            if isinstance(cls_ins, type)
            else cls_ins.target(*cls_ins.params)
        )
