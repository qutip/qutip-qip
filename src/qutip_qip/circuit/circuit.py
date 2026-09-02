"""
Quantum circuit representation and simulation.
"""

import warnings
import inspect
from typing import Iterable, Type, Self
from qutip import qeye, Qobj, basis, tensor
import numpy as np

from qutip_qip.circuit import CircuitSimulator
from qutip_qip.circuit._decompose import (
    _resolve_to_universal,
    _resolve_2q_basis,
)
from qutip_qip.circuit.instruction import (
    CircuitInstruction,
    ConditionalBranchInstruction,
    GateInstruction,
    LabelInstruction,
    MeasurementInstruction,
)
from qutip_qip.operations.conditional import Conditional, Label, ClassicalControlCheck
from qutip_qip.operations import (
    BloqBuilder,
    Gate,
    Measurement,
    Op,
    OpInstruction,
    expand_operator,
    get_unitary_gate,
)
from qutip_qip.operations.measurement import Mz
from qutip_qip.operations import AngleParametricGate
from qutip_qip.operations import gates as std
from qutip_qip.typing import Int, IntSequence
from qutip_qip.utils import check_limit, convert_type_input_to_sequence

try:
    from IPython.display import Image as DisplayImage, SVG as DisplaySVG
except ImportError:
    # If IPython doesn't exist, then we set the nice display hooks to be simple
    # pass-throughs.
    def DisplayImage(data, *args, **kwargs):
        return data

    def DisplaySVG(data, *args, **kwargs):
        return data


class QubitCircuit:
    """
    Representation of a quantum program/algorithm, maintaining a sequence
    of gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system.
    input_states : list
        A list of string such as `0`,'+', "A", "Y". Only used for latex.
    dims : list
        A list of integer for the dimension of each composite system.
        e.g [2,2,2,2,2] for 5 qubits system. If None, qubits system
        will be the default option.
    num_cbits : int
        Number of classical bits in the system.
    """

    def __init__(
        self,
        num_qubits=None,
        input_states=None,
        output_states=None,
        reverse_states=True,
        dims=None,
        num_cbits=0,
        user_gates=None,
        N=None,
    ):
        # number of qubits in the register
        if N is not None:
            warnings.warn(
                "The 'N' parameter is deprecated. Please use 'num_qubits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            num_qubits = N

        self.dims = dims if dims is not None else [2] * num_qubits
        self._builder = BloqBuilder(num_qubits, num_cbits, qreg_dim=self.dims)

        self.reverse_states = reverse_states
        self._instructions: list[CircuitInstruction] = []
        self._built_count = -1

        if input_states:
            self.input_states = input_states
        else:
            self.input_states = [None for i in range(self.num_qubits + num_cbits)]

        if output_states:
            self.output_states = output_states
        else:
            self.output_states = [None for i in range(self.num_qubits + num_cbits)]

        if user_gates is not None:
            self.user_gates = user_gates

    @property
    def num_qubits(self) -> int:
        """
        Number of qubits in the circuit.
        """
        return self._builder.num_qreg

    @property
    def num_cbits(self) -> int:
        """
        Number of cbits in the circuit.
        """
        return self._builder.num_creg

    @property
    def N(self) -> int:
        """
        Number of qubits in the circuit.
        """
        warnings.warn(
            "The 'N' parameter is deprecated. Please use 'num_qubits' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_qubits

    def __repr__(self) -> str:
        return ""

    @property
    def global_phase(self):
        return self._builder.global_phase

    def add_global_phase(self, phase: float):
        self._builder.add_global_phase(phase)

    @property
    def gates(self) -> list[CircuitInstruction]:
        warnings.warn(
            "QubitCircuit.gates has been replaced with QubitCircuit.instructions",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.instructions

    @gates.setter
    def gates(self, value: any) -> None:
        raise AttributeError(
            "QubitCircuit.gates has been replaced with QubitCircuit.instructions",
        )

    @property
    def user_gates(self) -> dict[str, Type[Gate]]:
        return getattr(self, "_user_gates", {})

    @user_gates.setter
    def user_gates(self, value) -> None:
        warnings.warn(
            "Using circuit.user_gates with string gate names "
            "is deprecated. Please define custom gates as "
            "Gate classes instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.__dict__["user_gates"] = value

        gate_classes = {}
        for gate_name, user_gate_callable in value.items():
            parameters = tuple(
                inspect.signature(user_gate_callable).parameters.values()
            )
            num_params = len(parameters)

            if num_params == 0:
                # Non parametric gate
                unitary = user_gate_callable()
                gate_classes[gate_name] = get_unitary_gate(gate_name, unitary)
                continue

            # Parametric gate.
            # We don't know the dimension of the unitary, need a try-run.
            sample_arg_value = tuple(
                (
                    parameter.default
                    if parameter.default is not inspect.Parameter.empty
                    else 0
                )
                for parameter in parameters
            )
            sample_unitary = user_gate_callable(*sample_arg_value)
            _num_qubits = int(np.log2(sample_unitary.shape[0]))
            _num_params = num_params

            # Without it, if you write get_qobj directly in the loop and reference
            # user_gate_callable, all generated classes can end up using the last callable
            # from the loop (late binding).
            def _make_get_qobj(_user_gate_callable):
                def get_qobj(self, dtype="dense"):
                    return _user_gate_callable(*self.arg_value).to(dtype)

                return get_qobj

            class _OldUserParametricGate(AngleParametricGate):
                __slots__ = ()
                namespace = None
                name = gate_name
                num_qubits = _num_qubits
                num_params = _num_params

                get_qobj = _make_get_qobj(user_gate_callable)

            gate_classes[gate_name] = _OldUserParametricGate

        self._user_gates = gate_classes

    @property
    def ops(self) -> tuple[OpInstruction, ...]:
        return self._builder.instructions

    @property
    def instructions(self) -> list[CircuitInstruction]:
        if self._built_count != len(self._builder._op_instructions):
            self.build()
        return self._instructions

    def add_state(
        self,
        state: str,
        targets: IntSequence,
        state_type: str = "input",  # FIXME Add an enum type hinting?
    ):
        """
        Add an input or output state to the circuit. By default all the input
        and output states will be initialized to `None`. A particular state can
        be added by specifying the state and the qubit where it has to be added
        along with the type as input or output.

        Parameters
        ----------
        state: str
            The state that has to be added. It can be any string such as `0`,
            '+', "A", "Y"
        targets: list
            A list of qubit positions where the given state has to be added.
        state_type: str
            One of either "input" or "output". This specifies whether the state
            to be added is an input or output.
            default: "input"

        """

        if state_type == "input":
            for i in targets:
                self.input_states[i] = state

        if state_type == "output":
            for i in targets:
                self.output_states[i] = state

    def if_test(
        self: Self,
        cbits: int | IntSequence,
        value: int,
        check: ClassicalControlCheck = "EQ",
    ):
        return self._builder.if_test(cbits, value, check)

    def add_op(
        self: Self,
        op: Op,
        qreg: int | IntSequence = (),
        creg: int | IntSequence = (),
        style: dict | None = None,
    ):
        self._builder.add_op(op, qreg, creg, style)

    def build(self: Self):
        """
        Converts _ops list to a frozen _instructions tuple.
        """
        self._instructions = []
        bloq = self._builder.build()

        for op_instruction in bloq.instructions:
            op = op_instruction.op

            if isinstance(op, Gate) or (isinstance(op, type) and issubclass(op, Gate)):
                self._instructions.append(
                    GateInstruction(
                        operation=op,
                        qubits=op_instruction.qreg,
                        cbits=op_instruction.creg,
                    )
                )

            # TODO In future Measurement should always just be a class
            elif isinstance(op, Measurement) or (
                isinstance(op, type) and issubclass(op, Measurement)
            ):
                self._instructions.append(
                    MeasurementInstruction(
                        operation=op,
                        qubits=op_instruction.qreg,
                        cbits=op_instruction.creg,
                    )
                )

            elif isinstance(op, Conditional):
                self._instructions.append(
                    ConditionalBranchInstruction(
                        operation=op, cbits=op_instruction.creg
                    )
                )

            elif isinstance(op, Label):
                self._instructions.append(LabelInstruction(operation=op))

            # TODO handle non-gate/non-measurement op

        self._instructions = tuple(self._instructions)
        self._built_count = len(self._builder._op_instructions)

    def add_measurement(
        self,
        measurement: Type[Measurement],
        targets: int | IntSequence,
        classical_store: int | IntSequence,
        index: None = None,
    ):
        """
        Adds a measurement with specified parameters to the circuit.

        Parameters
        ----------
        measurement : :class:`.Measurement`
            `Measurement` subclasses
        targets : int or Sequence of int
            Gate targets
        classical_store : int or Sequence of int
            Classical register where result of measurement is stored.
        index : list
            Positions to add the gate.
        """
        if index is not None:
            warnings.warn(
                "The 'index' argument to 'add_measurement' is no longer supported and will "
                "be ignored. Passing 'index' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(measurement, str):
            warnings.warn(
                "Passing a string name to add_measurement has been removed. "
                "Defaulting to measurement in the Z- basis.",
                DeprecationWarning,
                stacklevel=2,
            )
            measurement = Mz

        if targets is None or classical_store is None:
            raise ValueError("'targets' and 'classical_store' must not be None")

        targets = convert_type_input_to_sequence(int, "targets", targets)
        classical_store = convert_type_input_to_sequence(
            int, "classical_store", classical_store
        )

        check_limit("targets", targets, 0, self.num_qubits - 1)
        check_limit("classical_store", classical_store, 0, self.num_cbits - 1)

        self.add_op(
            op=measurement,
            qreg=tuple(targets),
            creg=tuple(classical_store),
        )

    def add_gate(
        self,
        gate: Gate | Type[Gate] | str,
        targets: int | IntSequence = (),
        controls: int | IntSequence = (),
        classical_controls: int | IntSequence = (),
        classical_control_value: int | None = None,
        style: dict = None,
        arg_value: None = None,
        arg_label: None = None,
        control_value: None = None,
        index: None = None,
    ):
        """
        Adds a gate with specified parameters to the circuit.

        Parameters
        ----------
        gate: :class:`~.operations.Gate` or :obj:`~.operations.Gate` or str
            Gate name. If gate is an instance of :class:`~.operations.Gate`,
            parameters are unpacked and added.
        targets: int or list, optional
            Index for the target qubits.
        controls: int or list, optional
            Indices for the (quantum) control qubits.
        arg_value: Any, optional
            Arguments for the gate. It will be used when generating the
            unitary matrix. For predefined gates, they are used when
            calling the ``get_compact_qobj`` methods of a gate.
        arg_label: string, optional
            Label for gate representation.
        classical_controls : int or list of int, optional
            Indices of classical bits to control the gate.
        control_value : optional
            Value of classical bits to control on, the classical controls are
            interpreted as an integer with the lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        style:
            For circuit draw
        """
        # Deprecation warnings
        if index is not None:
            warnings.warn(
                "The 'index' argument to 'add_measurement' is no longer supported and will "
                "be ignored. Gates are appended to the end of the circuit. "
                "Passing 'index' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if arg_value is not None or arg_label is not None:
            warnings.warn(
                "Define 'arg_value', 'arg_label' in your Gate object e.g. RX(np.pi)"
                ", 'arg_value', 'arg_label' arguments will be removed from 'add_gate' method in the future version.\n"
                "Please refer to the tutorial for the recommended way to define them:\n"
                "https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v5/quantum-circuits/quantum-gates.ipynb",
                DeprecationWarning,
                stacklevel=2,
            )

        if control_value is not None:
            warnings.warn(
                "'control_value' is no longer a valid argument and has been deprecated and will be removed in the future version.\n"
                "from qutip_qip.operations import get_controlled_gate\n"
                "from qutip_qip.operations.gates import X\n"
                "Example: gate = get_controlled_gate(X, num_ctrl_qubits=1, control_value=0)\n"
                "Please refer to the tutorial for the recommended way to define them:\n"
                "https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v5/quantum-circuits/quantum-gates.ipynb",
                DeprecationWarning,
                stacklevel=2,
            )

        if type(gate) is std.GLOBALPHASE:
            self.add_global_phase(gate.arg_value[0])
            return

        # This conditional block can be remove if the gate input is only
        # restricted to Gate subclasses or object instead of strings in the future.
        if not isinstance(gate, Gate):
            if type(gate) is str:
                if gate in std.GATE_CLASS_MAP:
                    warnings.warn(
                        "Passing Gate as a string input has been deprecated and will be removed in future versions.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    gate_class = std.GATE_CLASS_MAP[gate]

                elif gate in self.user_gates:
                    gate_class = self.user_gates[gate]
                else:
                    raise ValueError(
                        f"Gate '{gate}' is not a recognized standard gate or user-defined gate."
                    )

            elif issubclass(gate, Gate):
                gate_class = gate

            else:
                raise TypeError(
                    "gate must be of Gate type or object or a string ",
                    f"got {type(gate)} instead.",
                )

            if gate_class.is_parametric:
                gate = gate_class(arg_value, arg_label=arg_label)
            else:
                gate = gate_class

        # Check for gates
        if inspect.isabstract(gate):
            raise TypeError("gate must not be an abstract class")
        elif not (isinstance(gate, Gate) or issubclass(gate, Gate)):
            raise TypeError(f"gate must be of type Gate, got {gate}")
        elif gate.is_parametric and (not isinstance(gate, Gate)):
            raise TypeError(
                "You must pass an instantiated object for a Parametrized Gate"
            )
        elif (not gate.is_parametric) and (not issubclass(gate, Gate)):
            raise TypeError("You must pass a Gate type for a non-parametrized gate")

        # Handling case for integer input
        targets = convert_type_input_to_sequence(int, "targets", targets)
        controls = convert_type_input_to_sequence(int, "controls", controls)
        classical_controls = convert_type_input_to_sequence(
            int, "classical_controls", classical_controls
        )

        # Checks each element within the limit
        check_limit("targets", targets, 0, self.num_qubits - 1)
        check_limit("controls", controls, 0, self.num_qubits - 1)
        check_limit("classical_controls", classical_controls, 0, self.num_cbits - 1)

        # Check len(controls) == gate.num_ctrl_qubits
        if gate.is_controlled and len(controls) != gate.num_ctrl_qubits:
            raise ValueError(
                f"{gate.name} takes {gate.num_ctrl_qubits} qubits, but {len(controls)} were provided."
            )

        if len(controls) + len(targets) != gate.num_qubits:
            raise ValueError(
                f"{gate.name} takes {gate.num_qubits} qubits, but {len(controls) + len(targets)} were provided."
            )

        # Check for classical control
        default_classical_ctrl_val = 2 ** (len(classical_controls)) - 1

        if classical_control_value is None:
            if len(classical_controls) > 0:
                classical_control_value = default_classical_ctrl_val

        elif not isinstance(classical_control_value, Int):
            raise TypeError(
                f"classical_control_value must be an integer or None, got {classical_control_value}"
            )

        elif (
            classical_control_value < 0
            or classical_control_value > default_classical_ctrl_val
        ):
            raise ValueError(
                f"{classical_control_value} must be with [0, {default_classical_ctrl_val}]"
            )

        qubits = []
        qubits.extend(controls)
        qubits.extend(targets)

        if len(classical_controls) > 0:
            with self._builder.if_test(classical_controls, classical_control_value):
                self.add_op(op=gate, qreg=tuple(qubits), style=style)
        else:
            self.add_op(op=gate, qreg=tuple(qubits), style=style)

    def add_circuit(self, qc, start=0):  # TODO Instead of start have a qubit mapping?
        """
        Adds a block of a qubit circuit to the main circuit.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            The circuit block to be added to the main circuit.
        start : int
            The qubit on which the first gate is applied.
        """
        warnings.warn(
            "QubitCircuit.add_circuit has been deprecated, instead use Ops",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.num_qubits - start < qc.num_qubits:
            raise NotImplementedError("Targets exceed number of qubits.")

        for circuit_op in qc.ops:
            self.add_op(
                op=circuit_op.op,
                qreg=[start + t for t in circuit_op.qreg],
                creg=circuit_op.creg,
                style=circuit_op.style,
            )

    def adjacent_gates(*args, **kwargs):
        raise AttributeError(
            "adjacent_gates method has been removed. "
            "Use qutip_qip.transpiler.to_chain_structure method instead"
        )

    def remove_gate_or_measurement(
        self, index=None, end=None, name=None, remove="first"
    ):
        """
        Remove a gate from a specific index or between two indexes or the
        first, last or all instances of a particular gate.

        Parameters
        ----------
        index : int
            Location of gate or measurement to be removed.
        name : string
            Gate or Measurement name to be removed.
        remove : string
            If first or all gates/measurements are to be removed.
        """
        raise AttributeError("remove_gate_or_measurement method has been removed. ")

    def reverse_circuit(self):
        """
        Reverse an entire circuit of unitary gates.

        Returns
        -------
        qubit_circuit : :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of resolved gates for the
            qubit circuit in the reverse order.
        """

        warnings.warn(
            "QubitCircuit.reverse_circuit has been deprecated and will be future in future versions",
            DeprecationWarning,
            stacklevel=2,
        )

        temp = QubitCircuit(
            self.num_qubits,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
            input_states=self.input_states,
            output_states=self.output_states,
        )

        for circ_op in reversed(self.ops):
            temp.add_op(
                op=circ_op.op,
                qreg=circ_op.qreg,
                creg=circ_op.creg,
                style=circ_op.style,
            )

        return temp

    def run(
        self,
        state=None,
        cbits=None,
        measure_results=None,
    ):
        """
        Calculate the result of one instance of circuit run.

        Parameters
        ----------
        state : ket or oper, optional
                state vector or density matrix input. If not provided,
                defaults to the all-zero ket state \\|00...0> for the
                number of qubits in the circuit.
        cbits : List of ints, optional
                initialization of the classical bits.
        measure_results : tuple of ints, optional
            optional specification of each measurement result to enable
            post-selection. If specified, the measurement results are
            set to the tuple of bits (sequentially) instead of being
            chosen at random.

        Returns
        -------
        final_state : Qobj
                output state of the circuit run.
        """
        if state is None:
            state = tensor([basis(d, 0) for d in self.dims])
        if state.isket:
            mode = "state_vector_simulator"
        elif state.isoper:
            mode = "density_matrix_simulator"
        else:
            raise TypeError("State is not a ket or a density matrix.")

        sim = CircuitSimulator(self, mode)
        return sim.run(state, cbits, measure_results).get_final_states(0)

    def run_statistics(self, state, cbits=None):
        """
        Calculate all the possible outputs of a circuit
        (varied by measurement gates).

        Parameters
        ----------
        state : ket or oper
                state vector or density matrix input.
        cbits : List of ints, optional
                initialization of the classical bits.

        Returns
        -------
        result: CircuitResult
            Return a CircuitResult object containing
            output states and and their probabilities.
        """
        if state.isket:
            mode = "state_vector_simulator"
        elif state.isoper:
            mode = "density_matrix_simulator"
        else:
            raise TypeError("State is not a ket or a density matrix.")
        sim = CircuitSimulator(self, mode)
        return sim.run_statistics(state, cbits)

    def resolve_gates(self, basis=["CX", "RX", "RY", "RZ"]):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right in the specified
        basis.
        Calls '_resolve_to_universal' for each gate, this function maps
        each 'GATENAME' with its corresponding '_gate_basis_2q'
        Subsequently calls _resolve_2q_basis for each basis, this function maps
        each '2QGATENAME' with its corresponding '_basis_'

        Parameters
        ----------
        basis : list.
            Basis of the resolved circuit.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of resolved gates
            for the qubit circuit in the desired basis.
        """

        num_measurements = len(
            list(filter(lambda x: x.is_measurement_instruction(), self.instructions))
        )
        if num_measurements > 0:
            raise NotImplementedError("adjacent_gates must be called before \
                measurements are added to the circuit")

        basis_1q_valid = ["RX", "RY", "RZ", "IDENTITY"]
        basis_2q_valid = [
            "CX",
            "CSIGN",
            "CZ",
            "ISWAP",
            "SQRTSWAP",
            "SQRTISWAP",
        ]
        basis_1q = []
        basis_2q = []

        if isinstance(basis, Iterable):
            for gate in basis:
                if gate in basis_2q_valid:
                    basis_2q.append(gate)
                elif gate in basis_1q_valid:
                    basis_1q.append(gate)

            if len(basis_1q) == 1:
                raise ValueError("Not sufficient single-qubit gates in basis")
            if len(basis_1q) == 0:
                basis_1q = ["RX", "RY", "RZ"]

        else:  # only one 2q gate is given as basis
            basis_1q = ["RX", "RY", "RZ"]
            if basis in basis_2q_valid:
                basis_2q = [basis]
            else:
                raise ValueError(f"{basis} is not a valid two-qubit basis gate")

        match = False
        qc_temp = QubitCircuit(
            self.num_qubits,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
        )
        temp_resolved = QubitCircuit(self.num_qubits)

        for circ_instruction in self.instructions:
            gate = circ_instruction.operation
            targets = circ_instruction.targets
            controls = circ_instruction.controls

            if gate in (std.X, std.Y, std.Z):
                temp_resolved.add_global_phase(phase=np.pi / 2)

                if gate == std.X:
                    temp_resolved.add_gate(std.RX(np.pi), targets=targets)
                elif gate == std.Y:
                    temp_resolved.add_gate(std.RY(np.pi), targets=targets)
                else:
                    temp_resolved.add_gate(std.RZ(np.pi), targets=targets)

            else:
                try:
                    _resolve_to_universal(
                        circ_instruction, temp_resolved, basis_1q, basis_2q
                    )
                except KeyError:
                    if gate.name in basis:
                        temp_resolved.add_gate(
                            gate,
                            targets=targets,
                            controls=controls,
                            classical_controls=circ_instruction.cbits,
                            classical_control_value=circ_instruction.cbits_ctrl_value,
                        )
                    else:
                        exception = f"Gate {gate.name} cannot be resolved."
                        raise NotImplementedError(exception)

        qc_temp.add_global_phase(temp_resolved.global_phase)

        for basis_unit in ["CSIGN", "CZ", "ISWAP", "SQRTSWAP", "SQRTISWAP"]:
            if basis_unit in basis_2q:
                match = True
                _resolve_2q_basis(basis_unit, qc_temp, temp_resolved)
                break
        if not match:
            qc_temp._builder = temp_resolved._builder

        if len(basis_1q) != 2:
            return qc_temp

        instructions = qc_temp.instructions
        phase = qc_temp.global_phase
        half_pi = np.pi / 2

        qc_temp._builder = BloqBuilder(qc_temp.num_qubits, qc_temp.num_cbits)
        qc_temp.add_global_phase(phase)

        for circ_instruction in instructions:
            gate = circ_instruction.operation
            targets = circ_instruction.targets
            controls = circ_instruction.controls

            if type(gate) is std.RX and "RX" not in basis_1q:
                qc_temp.add_gate(
                    std.RY(-half_pi, arg_label=r"-\pi/2"),
                    targets=targets,
                )
                qc_temp.add_gate(
                    std.RZ(gate.arg_value[0], arg_label=gate.arg_label),
                    targets=targets,
                )
                qc_temp.add_gate(
                    std.RY(-half_pi, arg_label=r"\pi/2"),
                    targets=targets,
                )

            elif type(gate) is std.RY and "RY" not in basis_1q:
                qc_temp.add_gate(
                    std.RZ(-half_pi, arg_label=r"-\pi/2"),
                    targets=targets,
                )
                qc_temp.add_gate(
                    std.RX(gate.arg_value[0], arg_label=gate.arg_label),
                    targets=targets,
                )
                qc_temp.add_gate(
                    std.RZ(half_pi, arg_label=r"\pi/2"),
                    targets=targets,
                )

            elif type(gate) is std.RZ and "RZ" not in basis_1q:
                qc_temp.add_gate(
                    std.RX(-half_pi, arg_label=r"-\pi/2"),
                    targets=targets,
                )
                qc_temp.add_gate(
                    std.RY(gate.arg_value[0], arg_label=gate.arg_label),
                    targets=targets,
                )
                qc_temp.add_gate(
                    std.RX(half_pi, arg_label=r"\pi/2"),
                    targets=targets,
                )
            else:
                qc_temp.add_gate(
                    gate,
                    targets=targets,
                    controls=controls,
                    classical_controls=circ_instruction.cbits,
                    classical_control_value=circ_instruction.cbits_ctrl_value,
                )

        return qc_temp

    def propagators(self, expand=True, ignore_measurement=False):
        """
        Propagator matrix calculator returning the individual
        steps as unitary matrices operating from left to right.

        Parameters
        ----------
        expand : bool, optional
            Whether to expand the unitary matrices for the individual
            steps to the full Hilbert space for num_qubits.
            Defaults to ``True``.
            If ``False``, the unitary matrices will not be expanded and the
            list of unitaries will need to be combined with the list of
            gates in order to determine which qubits the unitaries should
            act on.
        ignore_measurement: bool, optional
            Whether :class:`.Measurement` operators should be ignored.
            If set False, it will raise an error
            when the circuit has measurement.

        Returns
        -------
        U_list : list
            Return list of unitary matrices for the qubit circuit.
            The global phase of circuit is the last element of ``U_list``.

        Notes
        -----
        If ``expand=False``, the global phase gate only returns a number.
        Also, classical controls are be ignored.
        """
        U_list = []

        gates = [
            (circ_instruction.operation, circ_instruction.qubits)
            for circ_instruction in self.instructions
            if circ_instruction.is_gate_instruction()
        ]
        if len(gates) < len(self.instructions) and not ignore_measurement:
            raise TypeError(
                "Cannot compute the propagator of a measurement operator."
                "Please set ignore_measurement=True."
            )

        # For Gate Instructions
        for gate, qubits in gates:
            qobj = gate.get_qobj()
            if expand:
                qobj = expand_operator(qobj, dims=self.dims, targets=qubits)
            U_list.append(qobj)

        # For Circuit's Global Phase
        qobj = Qobj([self.global_phase])
        if expand:
            qobj = std.GLOBALPHASE(self.global_phase).get_expanded_qobj(
                num_qubits=self.num_qubits
            )

        U_list.append(qobj)
        return U_list

    def compute_unitary(self):
        """Evaluates the matrix of all the gates in a quantum circuit.

        Returns
        -------
        circuit_unitary : :class:`qutip.Qobj`
            Product of all gate arrays in the quantum circuit.
        """
        sim = CircuitSimulator(self)
        result = sim.run(qeye(self.dims))
        circuit_unitary = result.get_final_states()[0]
        return circuit_unitary

    def draw(
        self,
        renderer="matplotlib",
        file_type="png",
        dpi=None,
        file_path="circuit",
        save=False,
        **kwargs,
    ):
        """
        Export circuit object as an image file in a supported format.

        Parameters
        ----------
        renderer : str, optional
            The renderer to use for the circuit. Options are 'latex', 'matplotlib', or 'text'.
            Default is 'matplotlib'.

        file_type : str, optional
            The type of the image file to export. Supported types are 'svg' and 'png'.
            Default is 'png'.

        dpi : int, optional
            The image density in dots per inch (dpi). Applicable for PNG, not for SVG.
            Default is None (set to 100 internally for PNG).

        file_path : str, optional
            The path to save the image file. Default is the current working directory.

        save : bool, optional
            If True, the image will be saved to the specified path.
            Default is False.

        kwargs : dict
            Additional keyword arguments passed to the renderer.
            Passed to StyleConfig Dataclass under base_renderer.py, pls refer to the file for more details.

        See Also
        --------
        :class:`~qutip_qip.circuit.base_renderer.StyleConfig`
            Configuration class for detailed style options.
        """

        if renderer == "latex":
            from qutip_qip.circuit.draw import TeXRenderer

            if file_type == "png" and dpi is None:
                dpi = 100

            latex = TeXRenderer(self)
            image_data = latex.raw_img(file_type, dpi)

            if save:
                mode = "w" if file_type == "svg" else "wb"
                with open(f"{file_path}.{file_type}", mode) as file:
                    file.write(image_data)

            return (
                DisplaySVG(data=image_data)
                if file_type == "svg"
                else DisplayImage(data=image_data)
            )

        elif renderer == "matplotlib":
            from qutip_qip.circuit.draw import MatRenderer

            if dpi is not None:
                kwargs["dpi"] = dpi

            mat = MatRenderer(self, **kwargs)
            mat.canvas_plot()
            if save:
                mat.save(file_path)

        elif renderer == "text":
            from qutip_qip.circuit.draw import TextRenderer

            text = TextRenderer(self, **kwargs)
            text.layout()
            if save:
                text.save(file_path)

        else:
            raise ValueError(
                f"Unknown renderer '{renderer}' not supported. Please choose from 'latex', 'matplotlib', 'text'."
            )

    def _to_qasm(self, qasm_out):
        """
        Pipe output of circuit object to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_out.output(f"qreg q[{self.num_qubits}];")
        if self.num_cbits:
            qasm_out.output(f"creg c[{self.num_cbits}];")
        qasm_out.output(n=1)

        for circ_instruction in self.instructions:
            if circ_instruction.is_gate_instruction() and not qasm_out.is_defined(
                circ_instruction.operation.name
            ):
                qasm_out._qasm_defns(circ_instruction.operation)

        for circ_instruction in self.instructions:
            circ_instruction.to_qasm(qasm_out)
