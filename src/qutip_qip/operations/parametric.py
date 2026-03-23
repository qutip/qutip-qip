import inspect
from abc import abstractmethod

from qutip import Qobj
from qutip_qip.operations import Gate
from qutip_qip.typing import Real, SequenceLike


class ParametricGate(Gate):
    r"""
    Abstract base class for parametric quantum gates.

    Parameters
    ----------
    arg_value : float or Sequence
        The argument value(s) for the gate. If a single float is provided,
        it is converted to a list. These values are saved as attributes
        and can be accessed or modified later.

    arg_label : str, optional
        Label for the argument to be shown in the circuit plot.

        Example:
        If ``arg_label="\phi"``, the LaTeX name for the gate in the circuit
        plot will be rendered as ``$U(\phi)$``.

    Attributes
    ----------
    num_params : int
        The number of parameters required by the gate. This is a mandatory
        class attribute for subclasses.

    arg_value : Sequence
        The numerical values of the parameters provided to the gate.

    arg_label : str, optional
        The LaTeX string representing the parameter variable in circuit plots.

    Raises
    ------
    ValueError
        If the number of provided arguments does not match `num_params`.
    """

    __slots__ = ("_arg_value", "arg_label")
    num_params: int

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Validates the subclass definition.

        Ensures that `num_params` is defined as a positive integer.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        # Assert num_params is a positive integer
        num_params = getattr(cls, "num_params", None)
        if (type(num_params) is not int) or (num_params < 1):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_params' must be a positive integer, "
                f"got {type(num_params)} with value {num_params}."
            )

        # Validate params must take only one argument 'args'
        validate_params_func = getattr(cls, "validate_params")
        if len(inspect.signature(validate_params_func).parameters) > 1:
            raise SyntaxError(
                f"Class '{cls.name}' method 'validate_params()' must take exactly 1 "
                f"additional arguments (only the implicit 'args'),"
                f" but it takes {len(inspect.signature(validate_params_func).parameters)}."
            )

        if not cls.is_parametric():
            raise ValueError(
                f"Class '{cls.name}' method 'is_parametric()' must always return True."
            )

        if cls.is_controlled():
            raise ValueError(
                f"Class '{cls.name}' method 'is_controlled()' must always return False."
            )

    def __init__(self, arg_value, arg_label: str | None = None) -> None:
        # This auto triggers a call to arg_value setter (where checks happen)
        self.arg_value = arg_value
        self.arg_label = arg_label

    @property
    def arg_value(self) -> tuple[any, ...]:
        return self._arg_value

    @arg_value.setter
    def arg_value(self, new_args: SequenceLike) -> None:
        if not isinstance(new_args, SequenceLike):
            new_args = [new_args]

        if len(new_args) != self.num_params:
            raise ValueError(
                f"Requires {self.num_params} parameters, got {len(new_args)}"
            )

        self.validate_params(new_args)
        self._arg_value = tuple(new_args)

    @staticmethod
    @abstractmethod
    def validate_params(arg_value):
        r"""
        Validate the provided parameters.

        This method should be implemented by subclasses to check if the
        parameters are valid type and within valid range (e.g., $0 \le \theta < 2\pi$).

        Parameters
        ----------
        arg_value : list of float
            The parameters to validate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_qobj(self, dtype: str = "dense") -> Qobj:
        """
        Get the QuTiP quantum object representation using the current parameters.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the gate with the specific `arg_value`.
        """
        raise NotImplementedError

    def inverse(self) -> Gate:
        if self.self_inverse:
            return self
        raise NotImplementedError

    @staticmethod
    def is_parametric() -> bool:
        return True

    def __str__(self) -> str:
        return f"""
            Gate({self.name}, arg_value={self.arg_value},
            arg_label={self.arg_label}),
        """

    def __eq__(self, other) -> bool:
        # Returns false for RX(0.5), RY(0.5)
        if type(self) is not type(other):
            return False

        # Returns false for RX(0.5), RX(0.6)
        if self.arg_value != other.arg_value:
            return False

        return True

    def __hash__(self) -> int:
        return hash((type(self), self.arg_value))


class AngleParametricGate(ParametricGate):
    __slots__ = ()

    @staticmethod
    def validate_params(arg_value) -> None:
        for arg in arg_value:
            if not isinstance(arg, Real):
                raise TypeError(f"Invalid arg {arg} in arg_value")
