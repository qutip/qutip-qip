import inspect
from abc import abstractmethod
from collections.abc import Iterable

from qutip import Qobj
from qutip_qip.operations import Gate


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

    __slots__ = ("arg_value", "arg_label")
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
                f"Class '{cls.__name__}' attribute 'num_params' must be a postive integer, "
                f"got {type(num_params)} with value {num_params}."
            )

    def __init__(self, arg_value: float, arg_label: str | None = None):
        if not isinstance(arg_value, Iterable):
            arg_value = [arg_value]

        if len(arg_value) != self.num_params:
            raise ValueError(
                f"Requires {self.num_params} parameters, got {len(arg_value)}"
            )

        self.validate_params(arg_value)
        self.arg_value = list(arg_value)
        self.arg_label = arg_label

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
        pass

    @abstractmethod
    def get_qobj(self) -> Qobj:
        """
        Get the QuTiP quantum object representation using the current parameters.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the gate with the specific `arg_value`.
        """
        pass

    @staticmethod
    def is_parametric_gate():
        return True

    def __str__(self):
        return f"""
            Gate({self.name}, arg_value={self.arg_value},
            arg_label={self.arg_label}),
        """

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False

        if self.arg_value != other.arg_value:
            return False
        return True


class AngleParametricGate(ParametricGate):
    __slots__ = ()

    @staticmethod
    def validate_params(arg_value):
        for arg in arg_value:
            try:
                float(arg)
            except TypeError:
                raise ValueError(f"Invalid arg {arg} in arg_value")
