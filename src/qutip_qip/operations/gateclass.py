from abc import ABC, ABCMeta, abstractmethod
import warnings
import inspect

import numpy as np
from qutip import Qobj
from qutip_qip.operations import controlled_gate

"""
.. testsetup::

   import numpy as np
   np.set_printoptions(5)
"""


# The prupose of the meta class is to prevent certain class attribute from being overwitten without inheritance.
# class X(Gate):
#   num_qubits = 1
#
# will work.
# But X.num_qubits = 2 will throw an error.
class GateReadOnlyMeta(ABCMeta):
    _read_only = ["num_qubits", "num_ctrl_qubits", "num_param"]

    def __setattr__(cls, name, value):
        for attribute in cls._read_only:
            if name == attribute and hasattr(cls, attribute):
                raise AttributeError(f"{attribute} is read-only!")
            super().__setattr__(name, value)


class Gate(ABC, metaclass=GateReadOnlyMeta):
    r"""
    Base class for a quantum gate,
    concrete gate classes need to be defined as subclasses.

    Parameters
    ----------
    arg_value : object
        Argument value of the gate. It will be saved as an attributes and
        can be accessed when generating the `:obj:qutip.Qobj`.
    control_value : int, optional
        The decimal value of controlling bits for executing
        the unitary operator on the target qubits.
        E.g. if the gate should be executed when the zero-th bit is 1,
        ``controll_value=1``;
        If the gate should be executed when the two bits are 1 and 0,
        ``controll_value=2``.
    arg_label : string
        Label for the argument, it will be shown in the circuit plot,
        representing the argument value provided to the gate, e.g,
        if ``arg_label="\phi" the latex name for the gate in the circuit plot
        will be ``$U(\phi)$``.
    name : string, optional
        The name of the gate. This is kept for backward compatibility
        to identify different gates.
        In most cases it is identical to the class name,
        but that is not guaranteed.
        It is recommended to use ``isinstance``
        or ``issubclass`` to identify a gate rather than
        comparing the name string.
    """
    num_qubits: int = None
    num_ctrl_qubits: int = 0
    num_param: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls): # Skip the below check for an abstract class
            return

        # If name attribute is not set in subclass it to the Name of the subclass
        # e.g. class Hadamard(Gate):
        #         pass
        #
        #      print(Hadamard.name) -> 'Hadamard'
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

        if "latex_str" not in cls.__dict__:
            cls.latex_str = cls.__name__

        num_qubits = getattr(cls, "num_qubits", None)
        num_ctrl_qubits = getattr(cls, "num_ctrl_qubits", None)

        if type(num_qubits) is not int:
            raise TypeError(f"Class '{cls.__name__}' must define class attribute 'num_qubits' as an integer.")

        if type(num_ctrl_qubits) is not int:
            raise TypeError(f"Class '{cls.__name__}' must define class attribute 'num_ctrl_qubits' as an integer got {num_ctrl_qubits}.")

        if num_qubits < 0:
            raise ValueError(f"Class '{cls.__name__}' class attribute 'num_qubits' can't be negative.")

        if  num_ctrl_qubits >= num_qubits:
            raise ValueError("For a Gate num_ctrl_qubits be strictly less than the num_qubits")

    def get_compact_qobj(self) -> Qobj:
        """
        Get the compact :class:`qutip.Qobj` representation of the gate
        operator, ignoring the controls and targets.
        In the unitary representation,
        it always assumes that the first few qubits are controls,
        then targets.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        warnings.warn(
            "get_compact_qobj method has been deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2
        )
        self.get_qobj()

    @abstractmethod
    def get_qobj(self, qubits, dims=None):
        """
        Get the :class:`qutip.Qobj` representation of the gate operator.
        The operator is expanded to the full Hilbert space according to
        the controls and targets qubits defined for the gate.

        Parameters
        ----------
        num_qubits : int, optional
            The number of qubits.
            If not given, use the minimal number of qubits required
            by the target and control qubits.
        dims : list, optional
            A list representing the dimensions of each quantum system.
            If not given, it is assumed to be an all-qubit system.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        raise NotImplementedError

    def __str__(self):
        return f"""
            Gate({self.name},
        """

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)


# Make this an abstract class
class ControlledGate(Gate):
    num_qubits = 1
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls): # Skip the num_ctrl_qubits check for an abstract class
            return

        num_ctrl_qubits = getattr(cls, "num_ctrl_qubits", None)
        if type(num_ctrl_qubits) is not int:
            raise TypeError(
                f"Class '{cls.__name__}' must define 'num_ctrl_qubits' as an integer class attribute.\
                 got {num_ctrl_qubits}."
            )

        if num_ctrl_qubits < 1:
            raise ValueError(
                f"Class '{cls.__name__}' class attribute 'num_ctrl_qubits' must be atleast 1."
            )

    def __init__(self, target_gate=None, control_value=None):
        if target_gate is None:
            if self._target_gate_class is not None:
                target_gate = self._target_gate_class
            else:
                raise ValueError(
                    "target_gate must be provided either as argument or class attribute."
                )

        self.target_gate = target_gate
        if control_value is None:
            self._control_value = 2**self.num_ctrl_qubits - 1
        else:
            self._validate_control_value(control_value)
            self._control_value = control_value
        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        self.latex_str = target_gate.latex_str
        self.num_qubits = self.target_gate.num_qubits + self.num_ctrl_qubits

    @property
    def control_value(self) -> int:
        return self._control_value

    def _validate_control_value(self, control_value: int):
        if type(control_value) is not int:
            raise TypeError(
                f"Control value must be an int, got {control_value}"
            )

        if control_value < 0:
            raise ValueError(
                f"Control value can't be negative, got {control_value}"
            )

        if control_value > 2**self.num_ctrl_qubits - 1:
            raise ValueError(
                f"""Control value can't be greater than 2^num_ctrl_qubits - 1,
                     got {control_value}"""
            )

    def __str__(self):
        return f"""
            Gate({self.name},
            control_value={self.control_value},
        """

    def get_qobj(self):
        return controlled_gate(
            U=self.target_gate.get_qobj(),
            control_value=self.control_value,
        )


class ParametrizedGate(Gate):
    def __init__(
        self,
        arg_value: float,
        arg_label: str = None,
    ):
        if type(arg_value) is float or type(arg_value) is np.float64:
            arg_value = [arg_value]

        # if len(arg_value) != self.num_param:
        #     raise ValueError(f"Requires {self.num_param} parameters, got {len(arg_value)}")

        self.arg_label = arg_label
        self.arg_value = arg_value
        # self.validate_params()

    # def __init_subclass__(cls, **kwargs):
    #     if (cls.num_param < 1):
    #         raise ValueError(f"For a Parametric Gate no. of parameters but be atleast 1, got {cls.num_param}")

    @abstractmethod
    def validate_params(self):
        pass

    def __str__(self):
        return f"""
            Gate({self.name}, arg_value={self.arg_value},
            arg_label={self.arg_label},
        """


class ControlledParamGate(ParametrizedGate, ControlledGate, ABC):
    def __init__(
        self,
        arg_value,
        arg_label=None,
        target_gate=None,
        control_value=1,
    ):
        if target_gate is None:
            if self._target_gate_class is not None:
                target_gate = self._target_gate_class
            else:
                raise ValueError(
                    "target_gate must be provided either as argument or class attribute."
                )

        if type(arg_value) is float:
            arg_value = [arg_value]

        ControlledGate.__init__(
            self,
            target_gate=target_gate(arg_value=arg_value, arg_label=arg_label),
            control_value=control_value,
        )
        # These can be commented out I believe
        self.arg_label = arg_label
        self.arg_value = arg_value

    def __str__(self):
        return f"""
            Gate({self.name},
            arg_value={self.arg_value}, arg_label={self.arg_label},
            control_value={self.control_value},
        """


def custom_gate_factory(gate_name: str, U: Qobj) -> Gate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class CustomGate(Gate):
        latex_str = r"U"
        name = gate_name
        num_qubits = int(np.log2(U.shape[0]))

        def __init__(
            self,
        ):
            self._U = U

        @staticmethod
        def get_qobj():
            return U

    return CustomGate


def controlled_gate_factory(
    target_gate: Gate,
    n_ctrl_qubits: int = 1,
    control_value: int = -1,
) -> type[ControlledGate]:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class _CustomGate(ControlledGate):
        latex_str = r"{\rm CU}"
        _target_gate_class = target_gate
        num_qubits = n_ctrl_qubits + target_gate.num_qubits
        num_ctrl_qubits = n_ctrl_qubits

        @property
        def control_value(self) -> int:
            if control_value == -1:
                return 2**n_ctrl_qubits - 1
            return control_value

    return _CustomGate


class SingleQubitGate(Gate):
    """Abstract one-qubit gate."""

    num_qubits: int = 1


class ParametrizedSingleQubitGate(ParametrizedGate):
    num_qubits: int = 1

    def validate_params(self):
        if len(self.arg_value) != self.num_param:
            raise ValueError(f"Requires {self.num_param} parameters, got {len(self.arg_value)}")


class TwoQubitGate(Gate):
    """Abstract two-qubit gate."""
    num_qubits: int = 2

class ParametrizedTwoQubitGate(ParametrizedGate):
    num_qubits: int = 2

    def validate_params(self):
        if len(self.arg_value) != self.num_param:
            raise ValueError(f"Requires {self.num_param} parameters, got {len(self.arg_value)}")
