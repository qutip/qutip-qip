from qutip import Qobj
from qutip_qip.operations.unitary import (
    Modifier,
    Unitary,
    UnitaryFamily,
    get_unitary_class,
)


class Inverse(Modifier):
    """
    Inverse() gives the inverse of a value like U or of a family like QFT.
    It is also the base class of every inverse that has no closed form.

    >>> Inverse(T)
    Tdag
    >>> Inverse(RX(0.5))
    RX(-0.5)
    >>> Inverse(RX)(0.5)
    RX(-0.5)

    Parameters
    ----------
    op : Unitary or type[Unitary]
        The value or family to invert.
    """

    @classmethod
    def _apply_modifier(cls, op: UnitaryFamily) -> UnitaryFamily:
        unitary_class = get_unitary_class(op)

        if isinstance(op, Unitary):
            closed_form = op.inverse()

        elif issubclass(unitary_class, Inverse):
            closed_form = unitary_class.target  # Inverse(Inverse(QFT)) is QFT

        elif unitary_class.self_inverse is True:
            closed_form = unitary_class
        else:
            closed_form = None

        if closed_form:
            return closed_form
        return cls._make_modifier(op, f"{unitary_class.__name__}dag")

    @classmethod
    def _find_simpler_form(cls, target: Unitary) -> Unitary | None:
        # Inverse(RX)(0.5) is RX(-0.5), not an Inverse around RX(0.5).
        closed_form = target.inverse()
        if closed_form is not None:
            return closed_form
        return super()._find_simpler_form(target)

    @property
    def name(self) -> str:
        return f"{self.target.name}dag"

    @property
    def latex_str(self) -> str:
        return rf"{{{self.target.latex_str}}}^\dagger"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return self.target.get_qobj(dtype).dag()

    def inverse(self) -> Unitary:
        return self.target

    def __repr__(self) -> str:
        return f"{self.target!r}dag"
