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
class _ReadOnlyGateMetaClass(ABCMeta):
    _read_only = ["num_qubits", "num_ctrl_qubits", "num_params"]

    def __setattr__(cls, name, value):
        for attribute in cls._read_only:
            if name == attribute and hasattr(cls, attribute):
                raise AttributeError(f"{attribute} is read-only!")
            super().__setattr__(name, value)


class Gate(ABC, metaclass=_ReadOnlyGateMetaClass):
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
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls): # Skip the below check for an abstract class
            return

        # If name attribute is not set in subclass, set it to the name of the subclass
        # e.g. class H(Gate):
        #         pass
        #
        #      print(H.name) -> 'H'
        #
        # e.g. class H(Gate):
        #         name = "Hadamard"
        #         pass
        #
        #      print(H.name) -> 'Hadamard'
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

        # Same as above for attribute latex_str (used in circuit draw)
        if "latex_str" not in cls.__dict__:
            cls.latex_str = cls.__name__

        num_qubits = getattr(cls, "num_qubits", None)
        if (type(num_qubits) is not int) or (num_qubits < 0):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_qubits' must be a postive integer, "
                f"got {type(num_qubits)} with value {num_qubits}."
            )

    @property
    @abstractmethod
    def num_qubits(self) -> Qobj:
        pass

    def get_compact_qobj(self) -> Qobj:
        """
        Get the compact :class:`qutip.Qobj` representation of the gate
        operator, ignoring the controls and targets.

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

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        raise NotImplementedError

    def __str__(self):
        return f"Gate({self.name}"

    def __repr__(self):
        return str(self)


# Make this an abstract class
class ControlledGate(Gate):
    num_qubits = 1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        num_ctrl_qubits = getattr(cls, "num_ctrl_qubits", None)
        if (type(num_ctrl_qubits) is not int) or (num_ctrl_qubits < 0):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_ctrl_qubits' must be a postive integer, "
                f"got {type(num_ctrl_qubits)} with value {num_ctrl_qubits}."
            )

        if cls.num_ctrl_qubits >= cls.num_qubits:
            raise ValueError(f"{cls.__name__}:'num_ctrl_qubits' be strictly less than the 'num_qubits'")


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
    @abstractmethod
    def num_ctrl_qubits(self) -> int:
        pass

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
            Gate({self.name}, num_qubits={self.num_qubits}
            control_value={self.control_value},
        """

    def get_qobj(self):
        return controlled_gate(
            U=self.target_gate.get_qobj(),
            control_value=self.control_value,
        )


class ParametrizedGate(Gate):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        num_params = getattr(cls, "num_params", None)
        if (type(num_params) is not int) or (num_params < 0):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_params' must be a postive integer, "
                f"got {type(num_params)} with value {num_params}."
            )

    def __init__(self, arg_value: float, arg_label: str = None):
        if type(arg_value) is float or type(arg_value) is np.float64:
            arg_value = [arg_value]

        # if len(arg_value) != self.num_params:
        #     raise ValueError(f"Requires {self.num_params} parameters, got {len(arg_value)}")

        self.arg_label = arg_label
        self.arg_value = arg_value
        # self.validate_params()

    @property
    @abstractmethod
    def num_params(self) -> Qobj:
        pass


    @abstractmethod
    def validate_params(self):
        pass

    def __str__(self):
        return f"""
            Gate({self.name}, arg_value={self.arg_value},
            arg_label={self.arg_label},
        """


class ControlledParamGate(ParametrizedGate, ControlledGate, ABC):
    num_params = 1
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
    num_params: int = 1

    def validate_params(self):
        if len(self.arg_value) != self.num_params:
            raise ValueError(f"Requires {self.num_params} parameters, got {len(self.arg_value)}")


class TwoQubitGate(Gate):
    """Abstract two-qubit gate."""
    num_qubits: int = 2

class ParametrizedTwoQubitGate(ParametrizedGate):
    num_qubits: int = 2
    num_params: int = 1

    def validate_params(self):
        if len(self.arg_value) != self.num_params:
            raise ValueError(f"Requires {self.num_params} parameters, got {len(self.arg_value)}")
