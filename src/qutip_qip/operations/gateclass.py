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
    targets : list or int
        The target qubits fo the gate.
    controls : list or int
        The controlling qubits of the gate.
    arg_value : object
        Argument value of the gate. It will be saved as an attributes and
        can be accessed when generating the `:obj:qutip.Qobj`.
    classical_controls : int or list of int, optional
        Indices of classical bits to control the unitary operator.
    control_value : int, optional
        The decimal value of controlling bits for executing
        the unitary operator on the target qubits.
        E.g. if the gate should be executed when the zero-th bit is 1,
        ``controll_value=1``;
        If the gate should be executed when the two bits are 1 and 0,
        ``controll_value=2``.
    classical_control_value : int, optional
        The decimal value of controlling classical bits for executing
        the unitary operator on the target qubits.
        E.g. if the gate should be executed when the zero-th bit is 1,
        ``controll_value=1``;
        If the gate should be executed when the two bits are 1 and 0,
        ``controll_value=2``.
        The default is ``2**len(classical_controls)-1``
        (i.e. all classical controls are 1).
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
    style : dict, optional
        A dictionary of style options for the gate.
        The options are passed to the `matplotlib` plotter.
        The default is None.
    """

    latex_str = r"U"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.latex_str = cls.__name__

    def __init__(
        self,
        name: str = None,
        targets: int | list[int] = None,
        classical_controls: int | list[int] | None = None,
        classical_control_value: int | None = None,
        style: dict | None = None,
    ):
        """
        Create a gate with specified parameters.
        """
        self.name = name if name is not None else self.__class__.__name__
        self.style = style

        if not isinstance(targets, Iterable) and targets is not None:
            self.targets = [targets]
        else:
            self.targets = targets

        if (
            not isinstance(classical_controls, Iterable)
            and classical_controls is not None
        ):
            self.classical_controls = [classical_controls]
        else:
            self.classical_controls = classical_controls

        if (
            self.classical_controls is not None
            and classical_control_value is None
        ):
            self.classical_control_value = (
                2 ** len(self.classical_controls) - 1
            )
        else:
            self.classical_control_value = classical_control_value

    @property
    @abstractmethod
    def qubit_count(self) -> int:
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

    def get_qobj(self, num_qubits=None, dims=None):
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

        all_targets = self.get_all_qubits()
        if num_qubits is None:
            num_qubits = max(all_targets) + 1

        return expand_operator(
            self.get_compact_qobj(),
            dims=dims,
            targets=all_targets,
        )

    def __str__(self):
        return f"""
            Gate({self.name}, targets={self.targets},
            classical controls={self.classical_controls},
            classical_control_value={self.classical_control_value})
        """

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)

class ControlledGate(Gate):
    def __init__(
        self,
        controls,
        targets,
        target_gate=None,
        control_value=1,
        classical_controls=None,
        classical_control_value=None,
        style=None,
    ):
        if target_gate is None:
            if self._target_gate_class is not None:
                target_gate = self._target_gate_class
            else:
                raise ValueError(
                    "target_gate must be provided either as argument or class attribute."
                )

        super().__init__(
            targets=targets,
            classical_controls=classical_controls,
            classical_control_value=classical_control_value,
            style=style,
        )
        self.target_gate = target_gate
        self.controls = (
            [controls] if not isinstance(controls, list) else controls
        )
        self._num_ctrl_qubits = len(self.controls)
        if control_value is None:
            self.control_value = 2 ** len(self.controls) - 1
        else:
            self.control_value = control_value
        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        self.latex_str = target_gate.latex_str

    @property
    def qubit_count(self) -> int:
        return self.target_gate.qubit_count + self.num_ctrl_qubits

    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits

    def get_all_qubits(self):
        return self.controls + self.targets

    def __str__(self):
        return f"""
            Gate({self.name}, targets={self.targets},
            controls={self.controls}, control_value={self.control_value},
            classical controls={self.classical_controls},
            classical_control_value={self.classical_control_value})
        """

    def get_compact_qobj(self):
        return controlled_gate(
            U=self.target_gate.get_compact_qobj(),
            controls=list(range(len(self.controls))),
            targets=list(
                range(
                    len(self.controls), len(self.targets) + len(self.controls)
                )
            ),
            control_value=self.control_value,
        )


class ParametrizedGate(Gate):
    def __init__(
        self,
        arg_value: float,
        arg_label: str = None,
        targets=None,
        classical_controls=None,
        classical_control_value=None,
        style=None,
    ):
        super().__init__(
            targets=targets,
            classical_controls=classical_controls,
            classical_control_value=classical_control_value,
            style=style,
        )
        self.arg_label = arg_label
        self.arg_value = arg_value

    def __str__(self):
        return f"""
            Gate({self.name}, targets={self.targets}, arg_value={self.arg_value},
            classical controls={self.classical_controls}, arg_label={self.arg_label},
            classical_control_value={self.classical_control_value})
        """


class ControlledParamGate(ParametrizedGate, ControlledGate):
    def __init__(
        self,
        controls,
        arg_value,
        arg_label=None,
        targets=None,
        target_gate=None,
        control_value=1,
        classical_controls=None,
        classical_control_value=None,
        style=None,
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
            controls=controls,
            targets=targets,
            target_gate=target_gate(
                targets=targets, arg_value=arg_value, arg_label=arg_label
            ),
            control_value=control_value,
            classical_controls=classical_controls,
            classical_control_value=classical_control_value,
            style=style,
        )
        self.arg_label = arg_label
        self.arg_value = arg_value

    def __str__(self):
        return f"""
            Gate({self.name}, targets={self.targets},
            arg_value={self.arg_value}, arg_label={self.arg_label},
            controls={self.controls}, control_value={self.control_value},
            classical controls={self.classical_controls},
            classical_control_value={self.classical_control_value})
        """

def custom_gate_factory(name: str, U: Qobj) -> Gate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class CustomGate(Gate):
        latex_str = r"U"

        def __init__(
            self,
            targets,
            classical_controls=None,
            classical_control_value=None,
            style=None,
        ):
            super().__init__(
                name=name,
                targets=targets,
                classical_controls=classical_controls,
                classical_control_value=classical_control_value,
                style=style,
            )
            self._U = U

        @staticmethod
        def get_compact_qobj():
            return U

        @property
        def qubit_count(self) -> int:
            return int(np.log2(U.shape[0]))

    return CustomGate


def controlled_gate_factory(
    target_gate: Gate,
    num_ctrl_qubits: int = 1,
) -> ControlledGate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class _CustomGate(ControlledGate):
        latex_str = r"{\rm CU}"
        _target_gate_class = target_gate

        @property
        def num_ctrl_qubits(self) -> int:
            return num_ctrl_qubits

        @property
        def qubit_count(self) -> int:
            return target_gate.qubit_count + self.num_ctrl_qubits

    return _CustomGate


class SingleQubitGate(Gate):
    """Abstract one-qubit gate."""

    @property
    def qubit_count(self) -> int:
        return 1


class ParametrizedSingleQubitGate(ParametrizedGate):
    @property
    def qubit_count(self) -> int:
        return 1

    def _verify_parameters(self):
        if self.targets is None or len(self.targets) != 1:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires one target"
            )


class TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    @property
    def qubit_count(self) -> int:
        return 2


class ParametrizedTwoQubitGate(ParametrizedGate):
    @property
    def qubit_count(self) -> int:
        return 2
