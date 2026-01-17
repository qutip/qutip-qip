import numbers
from collections.abc import Iterable
from abc import ABC, abstractmethod

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
        controls: int | list[int] = None,
        arg_value=None,
        control_value: int | None = None,
        classical_controls: int | list[int] | None = None,
        classical_control_value: int | None = None,
        arg_label: str | None = None,
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

        if not isinstance(controls, Iterable) and controls is not None:
            self.controls = [controls]
        else:
            self.controls = controls

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
        self.control_value = control_value
        self.arg_value = arg_value
        self.arg_label = arg_label

        for ind_list in [self.targets, self.controls, self.classical_controls]:
            if ind_list is None:
                continue
            all_integer = all(
                [isinstance(ind, numbers.Integral) for ind in ind_list]
            )
            if not all_integer:
                raise ValueError("Index of a qubit must be an integer")

    def get_all_qubits(self):
        """
        Return a list of all qubits that the gate operator
        acts on.
        The list concatenates the two lists representing
        the controls and the targets qubits while retains the order.

        Returns
        -------
        targets_list : list of int
            A list of all qubits, including controls and targets.
        """
        if self.controls is not None:
            return self.controls + self.targets
        if self.targets is not None:
            return self.targets
        else:
            # Special case: the global phase gate
            return []

    def __str__(self):
        str_name = (
            "Gate(%s, targets=%s, controls=%s,"
            " classical controls=%s, control_value=%s, classical_control_value=%s)"
        ) % (
            self.name,
            self.targets,
            self.controls,
            self.classical_controls,
            self.control_value,
            self.classical_control_value,
        )
        return str_name

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)

    def _to_qasm(self, qasm_out):
        """
        Pipe output of gate signature and application to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_gate = qasm_out.qasm_name(self.name)

        if not qasm_gate:
            error_str = "{} gate's qasm defn is not specified".format(
                self.name
            )
            raise NotImplementedError(error_str)

        if self.classical_controls:
            err_msg = "Exporting controlled gates is not implemented yet."
            raise NotImplementedError(err_msg)
        else:
            qasm_out.output(
                qasm_out._qasm_str(
                    qasm_gate, self.controls, self.targets, self.arg_value
                )
            )

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


class MultiControlledGate(Gate):
    def __init__(
        self,
        target_gate,
        controls,
        targets,
        control_value,
        style=None,
        **kwargs,
    ):
        super().__init__(
            controls=controls,
            targets=targets,
            control_value=control_value,
            style=style,
            **kwargs
        )
        self.target_gate = target_gate
        self.controls = (
            [controls] if not isinstance(controls, list) else controls
        )
        if control_value is None:
             self.control_value = 2**len(self.controls) - 1
        else:
             self.control_value = control_value
        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        self.latex_str = target_gate.latex_str

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


class ControlledGate(MultiControlledGate):
    def __init__(
        self,
        target_gate,
        controls,
        targets,
        control_value=1,
        style=None,
        **kwargs
    ):
        super().__init__(
            target_gate=target_gate,
            controls=controls,
            targets=targets,
            control_value=control_value,
            style=style,
            **kwargs
        )
    

class ParametrizedGate(Gate):
    def __init__(
        self,
        targets,
        arg_value,
        arg_label=None,
        style=None,
        **kwargs,
    ):
        super().__init__(
            targets=targets,
            arg_value=arg_value,
            arg_label=arg_label,
            style=style,
            **kwargs,
        )

    @abstractmethod
    def get_compact_qobj(self):
        pass


class CustomGate(Gate):
    """
    Custom gate that wraps an arbitrary quantum operator.
    """

    latex_str = r"U"

    def __init__(self, targets, U, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self._U = U

    def get_compact_qobj(self):
        return self._U


class SingleQubitGate(Gate):
    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        if self.targets is None or len(self.targets) != 1:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires one target"
            )
        if self.controls:
            raise ValueError(
                f"Gate {self.__class__.__name__} cannot have a control"
            )


class ParametrizedSingleQubitGate(ParametrizedGate):
    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        if self.targets is None or len(self.targets) != 1:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires one target"
            )


class TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        if len(self.get_all_qubits()) != 2:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires two targets"
            )


class ParametrizedTwoQubitGate(ParametrizedGate):
    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        if len(self.get_all_qubits()) != 2:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires two qubits"
            )
