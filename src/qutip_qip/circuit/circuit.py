"""
Quantum circuit representation and simulation.
"""

from __future__ import annotations
import inspect
from typing import Optional, Union, Tuple, List, TYPE_CHECKING
from collections.abc import Iterable

import numpy as np
from copy import deepcopy

from .texrenderer import TeXRenderer
from ._decompose import _resolve_to_universal, _resolve_2q_basis
from ..operations import (
    Gate,
    Measurement,
    expand_operator,
    GATE_CLASS_MAP,
)
from .circuitsimulator import (
    CircuitSimulator,
    CircuitResult,
)
from qutip import Qobj, qeye

if TYPE_CHECKING:
    from IPython.core.display import Image
    from ..qasm import QasmOutput

try:
    from IPython.display import Image as DisplayImage, SVG as DisplaySVG
except ImportError:
    # If IPython doesn't exist, then we set the nice display hooks to be simple
    # pass-throughs.
    def DisplayImage(data, *args, **kwargs):
        return data

    def DisplaySVG(data, *args, **kwargs):
        return data


__all__ = [
    "QubitCircuit",
    "CircuitResult",
]


class QubitCircuit:
    """
    Representation of a quantum program/algorithm, maintaining a sequence
    of gates.

    Parameters
    ----------
    N : int
        Number of qubits in the system.
    user_gates : dict
        Define a dictionary of the custom gates. See examples for detail.
    input_states : list
        A list of string such as `0`,'+', "A", "Y". Only used for latex.
    dims : list
        A list of integer for the dimension of each composite system.
        e.g [2,2,2,2,2] for 5 qubits system. If None, qubits system
        will be the default option.
    num_cbits : int
        Number of classical bits in the system.

    Examples
    --------
    >>> from qutip_qip.circuit import QubitCircuit
    >>> def user_gate():
    ...     mat = np.array([[1.,   0],
    ...                     [0., 1.j]])
    ...     return Qobj(mat, dims=[[2], [2]])
    >>> qubit_circuit = QubitCircuit(2, user_gates={"T":user_gate})
    >>> qubit_circuit.add_gate("T", targets=[0])
    """

    def __init__(
        self,
        N: int,
        input_states: Optional[Union[List[str], List[Optional[str]]]] = None,
        output_states: Optional[List[Optional[str]]] = None,
        reverse_states: bool = True,
        user_gates: None = None,
        dims: Optional[List[int]] = None,
        num_cbits: int = 0,
    ):
        # number of qubits in the register
        self.N = N
        self.reverse_states = reverse_states
        self.gates = []
        self.dims = dims if dims is not None else [2] * N
        self.num_cbits = num_cbits

        if input_states:
            self.input_states = input_states
        else:
            self.input_states = [None for i in range(N + num_cbits)]

        if output_states:
            self.output_states = output_states
        else:
            self.output_states = [None for i in range(N + num_cbits)]

        if user_gates is None:
            self.user_gates = {}
        else:
            if isinstance(user_gates, dict):
                self.user_gates = user_gates
            else:
                raise ValueError(
                    "`user_gate` takes a python dictionary of the form"
                    "{{str: gate_function}}, not {}".format(user_gates)
                )

    def __repr__(self) -> str:
        return ""

    def _repr_png_(self):
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
        targets: Optional[List[int]] = None,
        state_type: str = "input",
    ):
        """
        Add an input or ouput state to the circuit. By default all the input
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
        measurement: Union[str, Measurement],
        targets: Optional[Union[List[int], int]] = None,
        index: None = None,
        classical_store: Optional[Union[List[int], int]] = None,
    ):
        """
        Adds a measurement with specified parameters to the circuit.

        Parameters
        ----------
        measurement: string
            Measurement name. If name is an instance of `Measuremnent`,
            parameters are unpacked and added.
        targets: list
            Gate targets
        index : list
            Positions to add the gate.
        classical_store : int
            Classical register where result of measurement is stored.
        """

        if isinstance(measurement, Measurement):
            name = measurement.name
            targets = measurement.targets
            classical_store = measurement.classical_store

        else:
            name = measurement

        if index is None:
            self.gates.append(
                Measurement(
                    name, targets=targets, classical_store=classical_store
                )
            )

        else:
            for position in index:
                self.gates.insert(
                    position,
                    Measurement(
                        name, targets=targets, classical_store=classical_store
                    ),
                )

    def add_gate(
        self,
        gate: Any,
        targets: Optional[Union[List[int], int]] = None,
        controls: Optional[Union[List[int], int]] = None,
        arg_value: Optional[Any] = None,
        arg_label: Optional[str] = None,
        index: Optional[List[int]] = None,
        classical_controls: Optional[List[int]] = None,
        control_value: None = None,
        classical_control_value: Optional[int] = None,
        style: None = None,
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
        index : list, optional
            Positions to add the gate. Each index in the supplied list refers
            to a position in the original list of gates.
        classical_controls : int or list of int, optional
            Indices of classical bits to control the gate.
        control_value : int, optional
            Value of classical bits to control on, the classical controls are
            interpreted as an integer with the lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        """
        if not isinstance(gate, Gate):
            if gate in GATE_CLASS_MAP:
                gate_class = GATE_CLASS_MAP[gate]
            else:
                gate_class = Gate
            gate = gate_class(
                name=gate,
                targets=targets,
                controls=controls,
                arg_value=arg_value,
                arg_label=arg_label,
                classical_controls=classical_controls,
                control_value=control_value,
                classical_control_value=classical_control_value,
                style=style,
            )

        if index is None:
            self.gates.append(gate)

        else:
            # NOTE: Every insertion shifts the indices in the original list of
            #       gates by an additional position to the right.
            shifted_inds = np.sort(index) + np.arange(len(index))
            for position in shifted_inds:
                self.gates.insert(position, gate)

    def add_gates(
        self,
        gates: Union[
            Tuple[Gate, Gate, Gate, Gate],
            Tuple[Gate, Gate, Gate, Gate, Gate, Gate, Gate, Gate],
        ],
    ):
        """
        Adds a sequence of gates to the circuit in a positive order, i.e.
        the first gate in the sequence will be applied first to the state.

        Parameters
        ----------
        gates: Iterable (e.g., list)
            The sequence of gates to be added.
        """
        for g in gates:
            self.add_gate(g)

    def add_1q_gate(
        self,
        name: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        qubits: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Adds a single qubit gate with specified parameters on a variable
        number of qubits in the circuit. By default, it applies the given gate
        to all the qubits in the register.

        Parameters
        ----------
        name : string
            Gate name or the :class:`~.operations.Gate` object.
        start : int
            Starting location of qubits.
        end : int
            Last qubit for the gate.
        qubits : list
            Specific qubits for applying gates.
        kwargs : dict
            Keyword arguments for the gate, except for `targets`.
            See :class:`~.QubitCircuit.add_gate`.
        """
        if qubits is None:
            if start is None or end is None:
                raise ValueError(
                    "Both start and end must be specified if target qubits"
                    " are not provided."
                )
            qubits = range(start, end + 1)
        if not isinstance(qubits, Iterable):
            qubits = [qubits]
        for q in qubits:
            self.add_gate(name, targets=q, **kwargs)

    def add_circuit(
        self,
        qc: "QubitCircuit",
        start: int = 0,
        overwrite_user_gates: bool = False,
    ):
        """
        Adds a block of a qubit circuit to the main circuit.
        Globalphase gates are not added.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            The circuit block to be added to the main circuit.
        start : int
            The qubit on which the first gate is applied.
        """
        if self.N - start < qc.N:
            raise NotImplementedError("Targets exceed number of qubits.")

        # Inherit the user gates
        for user_gate in qc.user_gates:
            if user_gate in self.user_gates and not overwrite_user_gates:
                continue
            self.user_gates[user_gate] = qc.user_gates[user_gate]

        for circuit_op in qc.gates:
            if isinstance(circuit_op, Gate):
                if circuit_op.targets is not None:
                    tar = [target + start for target in circuit_op.targets]
                else:
                    tar = None
                if circuit_op.controls is not None:
                    ctrl = [control + start for control in circuit_op.controls]
                else:
                    ctrl = None

                self.add_gate(
                    circuit_op.name,
                    targets=tar,
                    controls=ctrl,
                    arg_value=circuit_op.arg_value,
                )
            elif isinstance(circuit_op, Measurement):
                self.add_measurement(
                    circuit_op.name,
                    targets=[target + start for target in circuit_op.targets],
                    classical_store=circuit_op.classical_store,
                )
            else:
                raise TypeError(
                    "The circuit to be added contains unknown \
                    operator {}".format(
                        circuit_op
                    )
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
        if index is not None:
            if index > len(self.gates):
                raise ValueError(
                    "Index exceeds number \
                                    of gates + measurements."
                )
            if end is not None and end <= len(self.gates):
                for i in range(end - index):
                    self.gates.pop(index + i)
            elif end is not None and end > self.N:
                raise ValueError(
                    "End target exceeds number \
                                    of gates + measurements."
                )
            else:
                self.gates.pop(index)

        elif name is not None and remove == "first":
            for circuit_op in self.gates:
                if name == circuit_op.name:
                    self.gates.remove(circuit_op)
                    break

        elif name is not None and remove == "last":
            for i in reversed(range(len(self.gates))):
                if name == self.gates[i].name:
                    self.gates.pop(i)
                    break

        elif name is not None and remove == "all":
            for i in reversed(range(len(self.gates))):
                if name == self.gates[i].name:
                    self.gates.pop(i)

        else:
            self.gates.pop()

    def reverse_circuit(self) -> "QubitCircuit":
        """
        Reverse an entire circuit of unitary gates.

        Returns
        -------
        qubit_circuit : :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of resolved gates for the
            qubit circuit in the reverse order.

        """
        temp = QubitCircuit(
            self.N,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
            input_states=self.input_states,
            output_states=self.output_states,
        )

        for circuit_op in reversed(self.gates):
            if isinstance(circuit_op, Gate):
                temp.add_gate(circuit_op)
            else:
                temp.add_measurement(circuit_op)

        return temp

    def run(
        self,
        state: Qobj,
        cbits: Optional[List[int]] = None,
        U_list: None = None,
        measure_results: Optional[List[int]] = None,
        precompute_unitary: bool = False,
    ) -> Qobj:
        """
        Calculate the result of one instance of circuit run.

        Parameters
        ----------
        state : ket or oper
                state vector or density matrix input.
        cbits : List of ints, optional
                initialization of the classical bits.
        U_list: list of Qobj, optional
            list of predefined unitaries corresponding to circuit.
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

    def run_statistics(
        self,
        state: Qobj,
        U_list: None = None,
        cbits: None = None,
        precompute_unitary: bool = False,
    ) -> CircuitResult:
        """
        Calculate all the possible outputs of a circuit
        (varied by measurement gates).

        Parameters
        ----------
        state : ket or oper
                state vector or density matrix input.
        cbits : List of ints, optional
                initialization of the classical bits.
        U_list: list of Qobj, optional
            list of predefined unitaries corresponding to circuit.

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

    def resolve_gates(
        self, basis: Union[str, List[str]] = ["CNOT", "RX", "RY", "RZ"]
    ) -> "QubitCircuit":
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
        qc_temp = QubitCircuit(
            self.N,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
        )
        temp_resolved = []

        basis_1q_valid = ["RX", "RY", "RZ", "IDLE"]
        basis_2q_valid = ["CNOT", "CSIGN", "ISWAP", "SQRTSWAP", "SQRTISWAP"]

        num_measurements = len(
            list(filter(lambda x: isinstance(x, Measurement), self.gates))
        )
        if num_measurements > 0:
            raise NotImplementedError(
                "adjacent_gates must be called before \
            measurements are added to the circuit"
            )

        if isinstance(basis, list):
            basis_1q = []
            basis_2q = []
            for gate in basis:
                if gate in basis_2q_valid:
                    basis_2q.append(gate)
                elif gate in basis_1q_valid:
                    basis_1q.append(gate)
                else:
                    pass
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
                    "%s is not a valid two-qubit basis gate" % basis
                )

        for gate in self.gates:
            if gate.name in ("X", "Y", "Z"):
                qc_temp.gates.append(Gate("GLOBALPHASE", arg_value=np.pi / 2))
                gate = Gate(
                    "R" + gate.name, targets=gate.targets, arg_value=np.pi
                )
            try:
                _resolve_to_universal(gate, temp_resolved, basis_1q, basis_2q)
            except KeyError:
                if gate.name in basis:
                    temp_resolved.append(gate)
                else:
                    exception = f"Gate {gate.name} cannot be resolved."
                    raise NotImplementedError(exception)

        match = False
        for basis_unit in ["CSIGN", "ISWAP", "SQRTSWAP", "SQRTISWAP"]:
            if basis_unit in basis_2q:
                match = True
                _resolve_2q_basis(basis_unit, qc_temp, temp_resolved)
                break
        if not match:
            qc_temp.gates = temp_resolved

        if len(basis_1q) == 2:
            temp_resolved = qc_temp.gates
            qc_temp.gates = []
            half_pi = np.pi / 2
            for gate in temp_resolved:
                if gate.name == "RX" and "RX" not in basis_1q:
                    qc_temp.gates.append(
                        Gate(
                            "RY",
                            gate.targets,
                            None,
                            arg_value=-half_pi,
                            arg_label=r"-\pi/2",
                        )
                    )
                    qc_temp.gates.append(
                        Gate(
                            "RZ",
                            gate.targets,
                            None,
                            gate.arg_value,
                            gate.arg_label,
                        )
                    )
                    qc_temp.gates.append(
                        Gate(
                            "RY",
                            gate.targets,
                            None,
                            arg_value=half_pi,
                            arg_label=r"\pi/2",
                        )
                    )
                elif gate.name == "RY" and "RY" not in basis_1q:
                    qc_temp.gates.append(
                        Gate(
                            "RZ",
                            gate.targets,
                            None,
                            arg_value=-half_pi,
                            arg_label=r"-\pi/2",
                        )
                    )
                    qc_temp.gates.append(
                        Gate(
                            "RX",
                            gate.targets,
                            None,
                            gate.arg_value,
                            gate.arg_label,
                        )
                    )
                    qc_temp.gates.append(
                        Gate(
                            "RZ",
                            gate.targets,
                            None,
                            arg_value=half_pi,
                            arg_label=r"\pi/2",
                        )
                    )
                elif gate.name == "RZ" and "RZ" not in basis_1q:
                    qc_temp.gates.append(
                        Gate(
                            "RX",
                            gate.targets,
                            None,
                            arg_value=-half_pi,
                            arg_label=r"-\pi/2",
                        )
                    )
                    qc_temp.gates.append(
                        Gate(
                            "RY",
                            gate.targets,
                            None,
                            gate.arg_value,
                            gate.arg_label,
                        )
                    )
                    qc_temp.gates.append(
                        Gate(
                            "RX",
                            gate.targets,
                            None,
                            arg_value=half_pi,
                            arg_label=r"\pi/2",
                        )
                    )
                else:
                    qc_temp.gates.append(gate)

        qc_temp.gates = deepcopy(qc_temp.gates)

        return qc_temp

    def adjacent_gates(self) -> "QubitCircuit":
        """
        Method to resolve two qubit gates with non-adjacent control/s or
        target/s in terms of gates with adjacent interactions.

        Returns
        -------
        qubit_circuit: :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of the gates
            for the qubit circuit with the resolved non-adjacent gates.

        """
        temp = QubitCircuit(
            self.N,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
        )
        swap_gates = [
            "SWAP",
            "ISWAP",
            "SQRTISWAP",
            "SQRTSWAP",
            "BERKELEY",
            "SWAPalpha",
        ]
        num_measurements = len(
            list(filter(lambda x: isinstance(x, Measurement), self.gates))
        )
        if num_measurements > 0:
            raise NotImplementedError(
                "adjacent_gates must be called before \
            measurements are added to the circuit"
            )

        for gate in self.gates:
            if gate.name == "CNOT" or gate.name == "CSIGN":
                start = min([gate.targets[0], gate.controls[0]])
                end = max([gate.targets[0], gate.controls[0]])
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        # Apply required gate if control, target are adjacent
                        # to each other, provided |control-target| is even.
                        if end == gate.controls[0]:
                            temp.gates.append(
                                Gate(gate.name, targets=[i], controls=[i + 1])
                            )
                        else:
                            temp.gates.append(
                                Gate(gate.name, targets=[i + 1], controls=[i])
                            )
                    elif (
                        start + end - i - i == 2 and (end - start + 1) % 2 == 1
                    ):
                        # Apply a swap between i and its adjacent gate, then
                        # the required gate if and then another swap if control
                        # and target have one qubit between them, provided
                        # |control-target| is odd.
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        if end == gate.controls[0]:
                            temp.gates.append(
                                Gate(
                                    gate.name,
                                    targets=[i + 1],
                                    controls=[i + 2],
                                )
                            )
                        else:
                            temp.gates.append(
                                Gate(
                                    gate.name,
                                    targets=[i + 2],
                                    controls=[i + 1],
                                )
                            )
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        i += 1
                    else:
                        # Swap the target/s and/or control with their adjacent
                        # qubit to bring them closer.
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        temp.gates.append(
                            Gate(
                                "SWAP",
                                targets=[start + end - i - 1, start + end - i],
                            )
                        )
                    i += 1

            elif gate.name in swap_gates:
                start = min([gate.targets[0], gate.targets[1]])
                end = max([gate.targets[0], gate.targets[1]])
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        temp.gates.append(Gate(gate.name, targets=[i, i + 1]))
                    elif (start + end - i - i) == 2 and (
                        end - start + 1
                    ) % 2 == 1:
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        temp.gates.append(
                            Gate(gate.name, targets=[i + 1, i + 2])
                        )
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        i += 1
                    else:
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        temp.gates.append(
                            Gate(
                                "SWAP",
                                targets=[start + end - i - 1, start + end - i],
                            )
                        )
                    i += 1

            else:
                raise NotImplementedError(
                    "`adjacent_gates` is not defined for "
                    "gate {}.".format(gate.name)
                )

        temp.gates = deepcopy(temp.gates)

        return temp

    def propagators(
        self, expand: bool = True, ignore_measurement: bool = False
    ) -> List[Qobj]:
        """
        Propagator matrix calculator returning the individual
        steps as unitary matrices operating from left to right.

        Parameters
        ----------
        expand : bool, optional
            Whether to expand the unitary matrices for the individual
            steps to the full Hilbert space for N qubits.
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

        Notes
        -----
        If ``expand=False``, the global phase gate only returns a number.
        Also, classical controls are be ignored.
        """
        U_list = []

        gates = [g for g in self.gates if not isinstance(g, Measurement)]
        if len(gates) < len(self.gates) and not ignore_measurement:
            raise TypeError(
                "Cannot compute the propagator of a measurement operator."
                "Please set ignore_measurement=True."
            )
        for gate in gates:
            if gate.name == "GLOBALPHASE":
                qobj = gate.get_qobj(self.N)
            else:
                qobj = self._get_gate_unitary(gate)
                if expand:
                    all_targets = gate.get_all_qubits()
                    qobj = expand_operator(
                        qobj, dims=self.dims, targets=all_targets
                    )
            U_list.append(qobj)
        return U_list

    def _get_gate_unitary(self, gate: Gate) -> Qobj:
        if gate.name in self.user_gates:
            if gate.controls is not None:
                raise ValueError(
                    "A user defined gate {} takes only  "
                    "`targets` variable.".format(gate.name)
                )
            func_or_oper = self.user_gates[gate.name]
            if inspect.isfunction(func_or_oper):
                func = func_or_oper
                para_num = len(inspect.getfullargspec(func)[0])
                if para_num == 0:
                    qobj = func()
                elif para_num == 1:
                    qobj = func(gate.arg_value)
                else:
                    raise ValueError(
                        "gate function takes at most one parameters."
                    )
            elif isinstance(func_or_oper, Qobj):
                qobj = func_or_oper
            else:
                raise ValueError("gate is neither function nor operator")
        else:
            qobj = gate.get_compact_qobj()
        return qobj

    def compute_unitary(self) -> Qobj:
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

        # This slightly convoluted dance with the conversion formats is because

    def draw(
        self,
        renderer: str = "matplotlib",
        file_type: str = "png",
        dpi: Optional[Union[float, int]] = None,
        file_path: str = "circuit",
        save: bool = False,
        **kwargs,
    ) -> Image:
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
            from .mat_renderer import MatRenderer

            if dpi is not None:
                kwargs["dpi"] = dpi

            mat = MatRenderer(self, **kwargs)
            mat.canvas_plot()
            if save:
                mat.save(file_path)

        elif renderer == "text":
            from .text_renderer import TextRenderer

            text = TextRenderer(self, **kwargs)
            text.layout()
            if save:
                text.save(file_path)

        else:
            raise ValueError(
                f"Unknown renderer '{renderer}' not supported. Please choose from 'latex', 'matplotlib', 'text'."
            )

    def _to_qasm(self, qasm_out: QasmOutput):
        """
        Pipe output of circuit object to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_out.output("qreg q[{}];".format(self.N))
        if self.num_cbits:
            qasm_out.output("creg c[{}];".format(self.num_cbits))
        qasm_out.output(n=1)

        for op in self.gates:
            if (not isinstance(op, Measurement)) and not qasm_out.is_defined(
                op.name
            ):
                qasm_out._qasm_defns(op)

        for op in self.gates:
            op._to_qasm(qasm_out)
