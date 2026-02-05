from collections.abc import Iterable
from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj
from qutip_qip.operations import controlled_gate, expand_operator


"""
.. testsetup::

   import numpy as np
   np.set_printoptions(5)
"""


class Gate(ABC):
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

    latex_str = r"U"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.latex_str = cls.__name__

    def __init__(
        self,
        name: str = None,
    ):
        """
        Create a gate with specified parameters.
        """
        self.name = name if name is not None else self.__class__.__name__

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        pass

    @property
    def num_ctrl_qubits(self) -> int:
        return 0

    @abstractmethod
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
        pass

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
        # This method isn't being used in the codebase
        return expand_operator(
            self.get_compact_qobj(),
            dims=dims,
            targets=qubits,
        )

    def __str__(self):
        return f"""
            Gate({self.name},
        """

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)


class ControlledGate(Gate):
    def __init__(
        self,
        target_gate=None,
        control_value=None,
    ):
        if target_gate is None:
            if self._target_gate_class is not None:
                target_gate = self._target_gate_class
            else:
                raise ValueError(
                    "target_gate must be provided either as argument or class attribute."
                )

        super().__init__()
        self.target_gate = target_gate
        if control_value is None:
            self._control_value = 2**self.num_ctrl_qubits - 1
        else:
            self._validate_control_value(control_value)
            self._control_value = control_value
        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        self.latex_str = target_gate.latex_str

    @property
    def control_value(self) -> int:
        return self._control_value

    @property
    def num_qubits(self) -> int:
        return self.target_gate.num_qubits + self.num_ctrl_qubits

    @property
    @abstractmethod
    def num_ctrl_qubits(self) -> int:
        raise NotImplementedError

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

    def get_compact_qobj(self):
        return controlled_gate(
            U=self.target_gate.get_compact_qobj(),
            control_value=self.control_value,
        )


class ParametrizedGate(Gate):
    def __init__(
        self,
        arg_value: float,
        arg_label: str = None,
    ):
        super().__init__()
        self.arg_label = arg_label
        self.arg_value = arg_value

    def __str__(self):
        return f"""
            Gate({self.name}, arg_value={self.arg_value},
            arg_label={self.arg_label},
        """


class ControlledParamGate(ParametrizedGate, ControlledGate):
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

        ControlledGate.__init__(
            self,
            target_gate=target_gate(arg_value=arg_value, arg_label=arg_label),
            control_value=control_value,
        )
        self.arg_label = arg_label
        self.arg_value = arg_value

    def __str__(self):
        return f"""
            Gate({self.name},
            arg_value={self.arg_value}, arg_label={self.arg_label},
            control_value={self.control_value},
        """


def custom_gate_factory(name: str, U: Qobj) -> Gate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class CustomGate(Gate):
        latex_str = r"U"

        def __init__(
            self,
        ):
            super().__init__(
                name=name,
            )
            self._U = U

        @staticmethod
        def get_compact_qobj():
            return U

        @property
        def num_qubits(self) -> int:
            return int(np.log2(U.shape[0]))

    return CustomGate


def controlled_gate_factory(
    target_gate: Gate,
    num_ctrl_qubits: int = 1,
    control_value: int = -1,
) -> ControlledGate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class _CustomGate(ControlledGate):
        latex_str = r"{\rm CU}"
        _target_gate_class = target_gate

        @property
        def control_value(self) -> int:
            if control_value == -1:
                return 2**num_ctrl_qubits - 1
            return control_value

        @property
        def num_ctrl_qubits(self) -> int:
            return num_ctrl_qubits

        @property
        def num_qubits(self) -> int:
            return target_gate.num_qubits + self.num_ctrl_qubits

    return _CustomGate


class SingleQubitGate(Gate):
    """Abstract one-qubit gate."""

    @property
    def num_qubits(self) -> int:
        return 1


class ParametrizedSingleQubitGate(ParametrizedGate):
    @property
    def num_qubits(self) -> int:
        return 1


class TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    @property
    def num_qubits(self) -> int:
        return 2


class ParametrizedTwoQubitGate(ParametrizedGate):
    @property
    def num_qubits(self) -> int:
        return 2
