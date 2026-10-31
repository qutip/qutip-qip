import copy
import inspect
import warnings
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import fields
from typing import ClassVar

from qutip_qip.operations.helper import class_or_instance_method
from qutip_qip.operations.op import Op


class Modifier(Op):
    """An op defined by a target op and a rule applied to it, e.g. Controlled, Inverse."""

    config = Op.config | {"target"}

    # Abstract class attribute i.e. subclasses need to define target Op for e.g. RX
    @property
    @abstractmethod
    def target(self) -> type[Op]: ...

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
                and issubclass(target, Op)
                and not inspect.isabstract(target)
            ):
                raise TypeError(
                    f"{cls.__name__}: target must be a concrete Op, got {target!r}."
                )
            if not target.unitary:
                raise TypeError(f"{cls.__name__}: target {target.name} is not unitary.")

            for attr, rule in cls._derived.items():
                if target.is_derived:
                    setattr(
                        cls,
                        attr,
                        property(lambda self, rule=rule: rule(self, self.target_op())),
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
        self.target_op()  # building the target runs its own validation

    @class_or_instance_method
    def target_op(cls_ins) -> Op | type[Op]:
        return (
            cls_ins.target
            if isinstance(cls_ins, type)
            else cls_ins.target(*cls_ins.params)
        )
