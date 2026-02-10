from abc import ABC, ABCMeta, abstractmethod
import inspect

import numpy as np
from qutip import Qobj
from qutip_qip.operations import controlled_gate


class _ReadOnlyGateMetaClass(ABCMeta):
    """
    The purpose of this meta class is to enforce read-only constraints on specific class attributes.
    
    This meta class prevents critical attributes from being overwritten 
    after definition, while still allowing them to be set during inheritance.

    For example:
         class X(Gate):
            num_qubits = 1   # Allowed (during class creation)

    But:
        X.num_qubits = 2     # Raises AttributeError (prevention of overwrite)

    This is required since num_qubits etc. are class attributes (shared by all object instances).
    """

    _read_only = ["num_qubits", "num_ctrl_qubits", "num_params", "target_gate", "self_inverse", "is_clifford"]
    _read_only_set = set(_read_only)

    def __setattr__(cls, name: str, value: any) -> None:
        for attribute in cls._read_only_set:
            if name == attribute and hasattr(cls, attribute):
                raise AttributeError(f"{attribute} is read-only!")
            super().__setattr__(name, value)


class Gate(ABC, metaclass=_ReadOnlyGateMetaClass):
    r"""
    Abstract base class for a quantum gate.

    Concrete gate classes or gate implementations should be defined as subclasses 
    of this class.

    Attributes
    ----------
    name : str
        The name of the gate. If not manually set, this defaults to the 
        class name. This is a class attribute; modifying it affects all 
        instances.

    num_qubits : int
        The number of qubits the gate acts upon. This is a mandatory 
        class attribute for subclasses.

    self_inverse: bool
        Indicates if the gate is its own inverse (e.g., $U = U^{-1}$).

    is_clifford: bool
        Indicates if the gate belongs to the Clifford group, which maps 
        Pauli operators to Pauli operators. Default value is False

    latex_str : str
        The LaTeX string representation of the gate (used for circuit drawing).
        Defaults to the class name if not provided.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically runs when a new subclass is defined via inheritance.

        This method sets the ``name`` and ``latex_str`` attributes 
        if they are not defined in the subclass. It also validates that 
        ``num_qubits`` is a non-negative integer.
        """

        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls): # Skip the below check for an abstract class
            return

        # If name attribute in subclass is not defined, set it to the name of the subclass
        # e.g. class H(Gate):
        #         pass
        
        #      print(H.name) -> 'H'
        
        # e.g. class H(Gate):
        #         name = "Hadamard"
        #         pass
        
        #      print(H.name) -> 'Hadamard'

        if "name" not in cls.__dict__:
            cls.name = cls.__name__

        # Same as above for attribute latex_str (used in circuit draw)
        if "latex_str" not in cls.__dict__:
            cls.latex_str = cls.__name__

        # Assert num_qubits is a non-negative integer
        num_qubits = getattr(cls, "num_qubits", None)
        if (type(num_qubits) is not int) or (num_qubits < 0):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_qubits' must be a non-negative integer, "
                f"got {type(num_qubits)} with value {num_qubits}."
            )

    @property
    @abstractmethod
    def num_qubits(self) -> Qobj:
        pass

    @staticmethod
    @abstractmethod
    def get_qobj() -> Qobj:
        """
        Get the :class:`qutip.Qobj` representation of the gate operator.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        pass

    @property
    def is_clifford(self) -> bool:
        return False

    @property
    @abstractmethod
    def self_inverse(self) -> bool:
        pass

    def inverse(self):
        """
        Return the inverse of the gate.

        If ``self_inverse`` is True, returns ``self``. Otherwise, 
        returns the specific inverse gate class.

        Returns
        -------
        Gate
            A Gate instance representing $G^{-1}$.
        """
        if self.self_inverse:
            return self
        # Implement this via gate factory?

    @staticmethod
    def is_controlled_gate() -> bool:
        """
        Check if the gate is a controlled gate.

        Returns
        -------
        bool
        """
        return False

    @staticmethod
    def is_parametric_gate() -> bool:
        """
        Check if the gate accepts variable parameters (e.g., rotation angles).

        Returns
        -------
        bool
            True if the gate is parametric (e.g., RX, RY, RZ), False otherwise.
        """
        return False

    def __str__(self) -> str:
        return f"Gate({self.name})"

    def __repr__(self) -> str:
        return f"Gate({self.name}, num_qubits={self.num_qubits}, qobj={self.get_qobj()})"


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
        if type(arg_value) is float or type(arg_value) is np.float64:
            arg_value = [arg_value]

        if len(arg_value) != self.num_params:
            raise ValueError(f"Requires {self.num_params} parameters, got {len(arg_value)} parameters")

        self.validate_params(arg_value)
        self.arg_value = arg_value
        self.arg_label = arg_label

    @property
    @abstractmethod
    def num_params(self) -> Qobj:
        pass

    @abstractmethod
    def validate_params(self, arg_value):
        """
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


class ControlledGate(Gate):
    r"""
    Abstract base class for controlled quantum gates.

    A controlled gate applies a target unitary operation only when the control 
    qubits are in a specific state. 

    Parameters
    ----------
    control_value : int, optional
        The decimal value of the control state required to execute the 
        unitary operator on the target qubits.
        
        Examples:
        * If the gate should execute when the 0-th qubit is $|1\rangle$, 
            set ``control_value=1``.
        * If the gate should execute when two control qubits are $|10\rangle$ 
            (binary 10), set ``control_value=2``.
        
        Defaults to all-ones (e.g., $2^N - 1$) if not provided.

    Attributes
    ----------
    num_ctrl_qubits : int
        The number of qubits acting as controls.

    target_gate : Gate
        The gate to be applied to the target qubits.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Validates the subclass definition.

        Ensures that:
        1. `num_ctrl_qubits` is a positive integer.
        2. `num_ctrl_qubits` is less than the total `num_qubits`.
        3. The sum of `num_ctrl_qubits` and `target.num_qubits` equals the total `num_qubits`.
        """

        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        # Assert num_ctrl_qubits is a positive integer
        num_ctrl_qubits = getattr(cls, "num_ctrl_qubits", None)
        if (type(num_ctrl_qubits) is not int) or (num_ctrl_qubits < 1):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_ctrl_qubits' must be a postive integer, "
                f"got {type(num_ctrl_qubits)} with value {num_ctrl_qubits}."
            )

        if cls.num_ctrl_qubits >= cls.num_qubits:
            raise ValueError(f"{cls.__name__}: 'num_ctrl_qubits' must be less than the 'num_qubits'")

        # Assert num_ctrl_qubits + target_gate.num_qubits = num_qubits
        if cls.num_ctrl_qubits + cls.target_gate.num_qubits != cls.num_qubits:
            raise AttributeError(f"'num_ctrls_qubits' {cls.num_ctrl_qubits} + 'target_gate qubits' {cls.target_gate.num_qubits} must be equal to 'num_qubits' {cls.num_qubits}")

        # Default value for control_value
        cls._control_value = 2**cls.num_ctrl_qubits - 1

    def __init__(self, control_value: int | None = None) -> None:
        if control_value is not None:
            self._validate_control_value(control_value)
            self._control_value = control_value

        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        self.latex_str = self.target_gate.latex_str

    @property
    @abstractmethod
    def num_ctrl_qubits(self) -> int:
        pass

    @property
    @abstractmethod
    def target_gate(self) -> int:
        pass

    @property
    def self_inverse(self) -> int:
        return self.target_gate.self_inverse

    @property
    def control_value(self) -> int:
        return self._control_value

    def _validate_control_value(self, control_value: int) -> None:
        """
        Internal validation for the control value.

        Raises
        ------
        TypeError
            If control_value is not an integer.
        ValueError
            If control_value is negative or exceeds the maximum value 
            possible for the number of control qubits ($2^N - 1$).
        """

        if type(control_value) is not int:
            raise TypeError(f"Control value must be an int, got {control_value}")

        if control_value < 0:
            raise ValueError(f"Control value can't be negative, got {control_value}")

        if control_value > 2**self.num_ctrl_qubits - 1:
            raise ValueError(f"Control value can't be greater than 2^num_ctrl_qubits - 1, got {control_value}")

    def get_qobj(self) -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        return controlled_gate(
            U=self.target_gate.get_qobj(),
            control_value=self.control_value,
        )

    @staticmethod
    def is_controlled_gate() -> bool:
        return True

    def is_parametric_gate(self) -> bool:
        return self.target_gate.is_parametric_gate()

    def __str__(self) -> str:
        return f"Gate({self.name}, target_gate={self.target_gate}, num_ctrl_qubits={self.num_ctrl_qubits}, control_value={self.control_value})"


class ControlledParametricGate(ParametricGate, ControlledGate, ABC):
    r"""
    Abstract base class for controlled parametric quantum gates.

    This class combines the functionality of :class:`ParametricGate` and 
    :class:`ControlledGate`. It represents gates that have both variable 
    parameters (like rotation angles) and control qubits.

    Common examples include the Controlled-Phase shift ($CPhase(\phi)$) or 
    Controlled-Rotation gates ($CR_x(\theta)$, $CR_y(\theta)$). 

    Parameters
    ----------
    arg_value : float or list of float
        The parameter value(s) for the target gate (e.g., the rotation angle $\theta$).

    arg_label : str, optional
        The LaTeX string label for the parameter, used in circuit draw.

    control_value : int, optional
        The decimal value of the control state required to execute the gate. 
        Defaults to all-ones (e.g., $2^N - 1$) if not provided.
    """

    def __init__(
        self,
        arg_value: any,
        arg_label: str | None = None,
        control_value: int | None = None
    ) -> None:

        if type(arg_value) is float:
            arg_value = [arg_value]

        ControlledGate.__init__(self, control_value=control_value)
        ParametricGate.__init__(self, arg_value=arg_value, arg_label=arg_label)

    def get_qobj(self) -> Qobj:
        return controlled_gate(
            U=self.target_gate(self.arg_value).get_qobj(),
            control_value=self.control_value,
        )

    def __str__(self) -> str:
        return f"""
            Gate({self.name}, target_gate{self.target_gate}
            arg_value={self.arg_value}, arg_label={self.arg_label},
            control_value={self.control_value}),
        """


def custom_gate_factory(gate_name: str, U: Qobj) -> Gate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    inverse = (U == U.dag())

    class CustomGate(Gate):
        latex_str = r"U"
        name = gate_name
        num_qubits = int(np.log2(U.shape[0]))
        self_inverse = inverse

        def __init__(self):
            self._U = U

        @staticmethod
        def get_qobj():
            return U

    return CustomGate


def controlled_gate_factory(
    gate: Gate,
    n_ctrl_qubits: int = 1,
    control_value: int = -1,
) -> ControlledGate:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    class _CustomGate(ControlledGate):
        latex_str = rf"C{gate.name}"
        target_gate = gate
        num_qubits = n_ctrl_qubits + target_gate.num_qubits
        num_ctrl_qubits = n_ctrl_qubits

        @property
        def control_value(self) -> int:
            if control_value == -1:
                return 2**n_ctrl_qubits - 1
            return control_value

    return _CustomGate


class AngleParametricGate(ParametricGate):
    def validate_params(self, arg_value):
        for arg in arg_value:
            try:
                float(arg)
            except TypeError:
                raise ValueError(f"Invalid arg {arg} in arg_value")

    @property
    def self_inverse(self) -> int:
        return False
