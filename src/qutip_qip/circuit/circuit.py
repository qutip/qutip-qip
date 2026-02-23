"""
Quantum circuit representation and simulation.
"""

import warnings
from typing import Iterable
from qutip import qeye, Qobj
import numpy as np

from ._decompose import _resolve_to_universal, _resolve_2q_basis
from qutip_qip.circuit import (
    CircuitSimulator,
    CircuitInstruction,
    GateInstruction,
    MeasurementInstruction,
)
from qutip_qip.circuit.utils import _check_iterable, _check_limit_
from qutip_qip.operations import Gate, Measurement, expand_operator
from qutip_qip.operations.std import RX, RY,RZ, GLOBALPHASE, GATE_CLASS_MAP
from qutip_qip.typing import IntList

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
        self._num_qubits = num_qubits
        if N is not None:
            warnings.warn(
                "The 'N' parameter is deprecated. Please use "
                "'num_qubits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._num_qubits = N

        self.reverse_states = reverse_states
        self.num_cbits: int = num_cbits
        self._global_phase: float = 0.0
        self._instructions: list[CircuitInstruction] = []
        self.dims = dims if dims is not None else [2] * self.num_qubits

        if input_states:
            self.input_states = input_states
        else:
            self.input_states = [
                None for i in range(self.num_qubits + num_cbits)
            ]

        if output_states:
            self.output_states = output_states
        else:
            self.output_states = [
                None for i in range(self.num_qubits + num_cbits)
            ]

        if user_gates is not None:
            raise ValueError(
                "`user_gates` has been removed from qutip-qip from version 0.5.0"
                "To define custom gates refer to this example in documentation <link>"
            )

    @property
    def global_phase(self):
        return self._global_phase

    def add_global_phase(self, phase: float):
        self._global_phase += phase
        self._global_phase %= 2 * np.pi

    @property
    def gates(self) -> list[CircuitInstruction]:
        warnings.warn(
            "QubitCircuit.gates has been replaced with QubitCircuit.instructions",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._instructions

    gates.setter

    def gates(self) -> None:
        warnings.warn(
            "QubitCircuit.gates has been replaced with QubitCircuit.instructions",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    def instructions(self) -> list[CircuitInstruction]:
        return self._instructions

    @property
    def num_qubits(self) -> int:
        """
        Number of qubits in the circuit.
        """
        return self._num_qubits

    @property
    def N(self) -> int:
        """
        Number of qubits in the circuit.
        """
        warnings.warn(
            "The 'N' parameter is deprecated. Please use "
            "'num_qubits' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_qubits

    def __repr__(self) -> str:
        return ""

    def _repr_png_(self) -> None:
        """
        Provide PNG representation for Jupyter Notebook.
        """
        try:
            self.draw(renderer="matplotlib")
        except ImportError:
            self.draw("text")

    def add_state(
        self,
        state: str,
        targets: IntList,
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

    def add_measurement(
        self,
        measurement: str | Measurement,
        targets: int | IntList,
        classical_store: int,
        index: None = None,
    ):
        """
        Adds a measurement with specified parameters to the circuit.

        Parameters
        ----------
        measurement: string
            Measurement name. If name is an instance of `Measurement`,
            parameters are unpacked and added.
        targets: list
            Gate targets
        classical_store : int
            Classical register where result of measurement is stored.
        index : list
            Positions to add the gate.
        """
        if index is not None:
            raise ValueError("argument index is no longer supported")

        if isinstance(measurement, Measurement):
            name = measurement.name
            targets = measurement.targets
            classical_store = measurement.classical_store

        else:
            name = measurement

        meas = Measurement(
            name, targets=targets, classical_store=classical_store
        )

        if type(targets) is int:
            targets = [targets]

        if type(classical_store) is int:
            classical_store = [classical_store]

        self._instructions.append(
            MeasurementInstruction(
                operation=meas,
                qubits=tuple(targets),
                cbits=tuple(classical_store),
            )
        )

    def add_gate(
        self,
        gate: Gate | str,
        targets: Iterable[int] = [],
        controls: Iterable[int] = [],
        arg_value: any = None,
        arg_label: str | None = None,
        control_value: int | None = None,
        classical_controls: Iterable[int] = [],
        classical_control_value: int | None = None,
        style: dict = None,
        index: None = None,
    ):
        """
        Adds a gate with specified parameters to the circuit.

        Parameters
        ----------
        gate: string or :class:`~.operations.Gate`
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
        control_value : int, optional
            Value of classical bits to control on, the classical controls are
            interpreted as an integer with the lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        style:
            For circuit draw
        """
        if index is not None:
            raise ValueError("argument index is no longer supported")

        if arg_value is not None or arg_label is not None:
            warnings.warn(
                "Define 'arg_value', 'arg_label' in your Gate object e.g. RX(arg_value=np.pi)"
                ", 'arg_value', 'arg_label' arguments will be removed from 'add_gate' method in the future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if control_value is not None:
            warnings.warn(
                "Define 'control_value', in your Gate object e.g. CX(control_value=0)"
                ", 'control_value' argument will be removed from 'add_gate' method in the future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(gate, GLOBALPHASE):
            self.add_global_phase(gate.arg_value[0])
            return

        # Handling case for int input (TODO use try except)
        targets = [targets] if type(targets) is int else targets
        controls = [controls] if type(controls) is int else controls
        classical_controls = (
            [classical_controls]
            if type(classical_controls) is int
            else classical_controls
        )

        # This will raise an error if not an iterable type (e.g. list, tuple, etc.)
        _check_iterable("targets", targets)
        _check_iterable("controls", controls)
        _check_iterable("classical_controls", classical_controls)

        # Checks each element is of given type (e.g. int) and within the limit
        _check_limit_("targets", targets, self.num_qubits - 1, int)
        _check_limit_("controls", controls, self.num_qubits - 1, int)
        _check_limit_(
            "classical_controls", classical_controls, self.num_cbits - 1, int
        )

        # Check len(controls) == gate.num_ctrl_qubits

        # Default value for classical control
        if len(classical_controls) > 0 and classical_control_value is None:
            classical_control_value = 2 ** (len(classical_controls)) - 1

        # This can be remove if the gate input is only restricted to Gate or its object instead of strings
        if not isinstance(gate, Gate):
            if type(gate) is str and gate in GATE_CLASS_MAP:
                warnings.warn(
                    "Passing Gate as a string input has been deprecated and will be removed in future versions.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                gate_class = GATE_CLASS_MAP[gate]

            elif issubclass(gate, Gate):
                gate_class = gate
            else:
                raise ValueError(
                    "Can only pass standard gate name as strings"
                    "or Gate class or its object instantiation"
                )

            if (
                gate_class.is_controlled_gate()
                and gate_class.is_parametric_gate()
            ):
                gate = gate_class(
                    control_value=control_value,
                    arg_value=arg_value,
                    arg_label=arg_label,
                )

            elif gate_class.is_parametric_gate():
                gate = gate_class(arg_value=arg_value, arg_label=arg_label)

            elif gate_class.is_controlled_gate():
                gate = gate_class(control_value=control_value)
            else:
                gate = gate_class

        qubits = []
        if controls is not None:
            qubits.extend(controls)
        qubits.extend(targets)

        cbits = tuple()
        if classical_controls is not None:
            cbits = tuple(classical_controls)

        self._instructions.append(
            GateInstruction(
                operation=gate,
                qubits=tuple(qubits),
                cbits=cbits,
                cbits_ctrl_value=classical_control_value,
                style=style,
            )
        )

    def add_circuit(
        self, qc, start=0
    ):  # TODO Instead of start have a qubit mapping?
        """
        Adds a block of a qubit circuit to the main circuit.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            The circuit block to be added to the main circuit.
        start : int
            The qubit on which the first gate is applied.
        """
        if self.num_qubits - start < qc.num_qubits:
            raise NotImplementedError("Targets exceed number of qubits.")

        for circuit_op in qc.instructions:
            if circuit_op.is_gate_instruction():
                self.add_gate(
                    circuit_op.operation,
                    targets=[start + t for t in circuit_op.targets],
                    controls=[start + c for c in circuit_op.controls],
                    classical_controls=circuit_op.cbits,
                    classical_control_value=circuit_op.cbits_ctrl_value,
                    style=circuit_op.style,
                )

            elif circuit_op.is_measurement_instruction():
                self.add_measurement(
                    circuit_op.operation.name,
                    targets=[target + start for target in circuit_op.qubits],
                    classical_store=list(circuit_op.cbits),
                )

            else:
                raise TypeError(f"The circuit to be added contains unknown \
                    operator {circuit_op[0]}")

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
        if index is not None:
            if index > len(self.instructions):
                raise ValueError("Index exceeds number \
                    of gates + measurements.")

            if end is not None and end <= len(self.instructions):
                for i in range(end - index):
                    self._instructions.pop(index + i)

            elif end is not None and end > self.num_qubits:
                raise ValueError("End target exceeds number \
                    of gates + measurements.")

            else:
                self._instructions.pop(index)

        elif name is not None and remove == "first":
            for circuit_op in self.instructions:
                if name == circuit_op.operation.name:
                    self._instructions.remove(circuit_op)
                    break

        elif name is not None and remove == "last":
            for i in reversed(range(len(self.instructions))):
                if name == self.instructions[i].operation.name:
                    self._instructions.pop(i)
                    break

        elif name is not None and remove == "all":
            for i in reversed(range(len(self.instructions))):
                if name == self.instructions[i].operation.name:
                    self._instructions.pop(i)

        else:
            self._instructions.pop()

    def reverse_circuit(self):
        """
        Reverse an entire circuit of unitary gates.

        Returns
        -------
        qubit_circuit : :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of resolved gates for the
            qubit circuit in the reverse order.

        """
        temp = QubitCircuit(
            self.num_qubits,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
            input_states=self.input_states,
            output_states=self.output_states,
        )

        for circ_instruction in reversed(self.instructions):
            if circ_instruction.is_gate_instruction():
                temp.add_gate(
                    gate=circ_instruction.operation,
                    targets=circ_instruction.targets,
                    controls=circ_instruction.controls,
                    classical_controls=circ_instruction.cbits,
                    classical_control_value=circ_instruction.cbits_ctrl_value,
                    style=circ_instruction.style,
                )

            elif circ_instruction.is_measurement_instruction():
                temp.add_measurement(
                    measurement=circ_instruction.operation,
                    targets=circ_instruction.qubits,
                    classical_store=circ_instruction.cbits[0],
                )

        return temp

    def run(
        self,
        state,
        cbits=None,
        measure_results=None,
        precompute_unitary=False,
    ):
        """
        Calculate the result of one instance of circuit run.

        Parameters
        ----------
        state : ket or oper
                state vector or density matrix input.
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
        if state.isket:
            mode = "state_vector_simulator"
        elif state.isoper:
            mode = "density_matrix_simulator"
        else:
            raise TypeError("State is not a ket or a density matrix.")
        sim = CircuitSimulator(
            self,
            mode,
            precompute_unitary,
        )
        return sim.run(state, cbits, measure_results).get_final_states(0)

    def run_statistics(self, state, cbits=None, precompute_unitary=False):
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
        sim = CircuitSimulator(self, mode, precompute_unitary)
        return sim.run_statistics(state, cbits)

    def resolve_gates(self, basis=["CNOT", "CX", "RX", "RY", "RZ"]):
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
            list(
                filter(
                    lambda x: x.is_measurement_instruction(), self.instructions
                )
            )
        )
        if num_measurements > 0:
            raise NotImplementedError("adjacent_gates must be called before \
                measurements are added to the circuit")

        basis_1q_valid = ["RX", "RY", "RZ", "IDLE"]
        basis_2q_valid = [
            "CNOT",
            "CX",
            "CSIGN",
            "CZ",
            "ISWAP",
            "SQRTSWAP",
            "SQRTISWAP",
        ]
        basis_1q = []
        basis_2q = []

        if isinstance(basis, list):
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
                raise ValueError(
                    f"{basis} is not a valid two-qubit basis gate"
                )

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

            if gate.name in ("X", "Y", "Z"):
                temp_resolved.add_global_phase(phase=np.pi / 2)

                if gate.name == "X":
                    temp_resolved.add_gate(RX(np.pi), targets=targets)
                elif gate.name == "Y":
                    temp_resolved.add_gate(RY(np.pi), targets=targets)
                else:
                    temp_resolved.add_gate(RZ(np.pi), targets=targets)

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
                            style=circ_instruction.style,
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
            qc_temp._instructions = temp_resolved.instructions

        if len(basis_1q) != 2:
            return qc_temp

        instructions = qc_temp.instructions
        qc_temp._instructions = []
        half_pi = np.pi / 2

        for circ_instruction in instructions:
            gate = circ_instruction.operation
            targets = circ_instruction.targets
            controls = circ_instruction.controls

            if gate.name == "RX" and "RX" not in basis_1q:
                qc_temp.add_gate(
                    RY(arg_value=-half_pi, arg_label=r"-\pi/2"),
                    targets=targets,
                )
                qc_temp.add_gate(
                    RZ(arg_value=gate.arg_value, arg_label=gate.arg_label),
                    targets=targets,
                )
                qc_temp.add_gate(
                    RY(arg_value=-half_pi, arg_label=r"\pi/2"),
                    targets=targets,
                )

            elif gate.name == "RY" and "RY" not in basis_1q:
                qc_temp.add_gate(
                    RZ(arg_value=-half_pi, arg_label=r"-\pi/2"),
                    targets=targets,
                )
                qc_temp.add_gate(
                    RX(arg_value=gate.arg_value, arg_label=gate.arg_label),
                    targets=targets,
                )
                qc_temp.add_gate(
                    RZ(arg_value=half_pi, arg_label=r"\pi/2"),
                    targets=targets,
                )

            elif gate.name == "RZ" and "RZ" not in basis_1q:
                qc_temp.add_gate(
                    RX(arg_value=-half_pi, arg_label=r"-\pi/2"),
                    targets=targets,
                )
                qc_temp.add_gate(
                    RY(arg_value=gate.arg_value, arg_label=gate.arg_label),
                    targets=targets,
                )
                qc_temp.add_gate(
                    RX(arg_value=half_pi, arg_label=r"\pi/2"),
                    targets=targets,
                )
            else:
                qc_temp.add_gate(
                    gate,
                    targets=targets,
                    controls=controls,
                    classical_controls=circ_instruction.cbits,
                    classical_control_value=circ_instruction.cbits_ctrl_value,
                    style=circ_instruction.style,
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
            qobj = GLOBALPHASE(self.global_phase).get_qobj(
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

        qasm_out.output("qreg q[{}];".format(self.num_qubits))
        if self.num_cbits:
            qasm_out.output("creg c[{}];".format(self.num_cbits))
        qasm_out.output(n=1)

        for circ_instruction in self.instructions:
            if (
                circ_instruction.is_gate_instruction()
                and not qasm_out.is_defined(circ_instruction.operation.name)
            ):
                qasm_out._qasm_defns(circ_instruction.operation)

        for circ_instruction in self.instructions:
            circ_instruction.to_qasm(qasm_out)
