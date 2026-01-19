"""
Quantum circuit representation and simulation.
"""

import numpy as np

from ._decompose import _resolve_to_universal, _resolve_2q_basis
from qutip_qip.operations import (
    Gate,
    ControlledGate,
    ParametrizedGate,
    ControlledParamGate,
    GLOBALPHASE,
    RX,
    RY,
    RZ,
    Measurement,
    expand_operator,
    GATE_CLASS_MAP,
)
from qutip_qip.circuit import CircuitSimulator
from qutip import qeye


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
    N : int
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
        N,
        input_states=None,
        output_states=None,
        reverse_states=True,
        dims=None,
        num_cbits=0,
        user_gates=None,
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

        if user_gates is not None:
            raise ValueError(
                "`user_gates` has been removed from qutip-qip from version 0.5.0"
                "To define custom gates refer to this example in documentation <link>"
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

    def add_state(self, state, targets=None, state_type="input"):
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
        self, measurement, targets=None, index=None, classical_store=None
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
        gate,
        targets=None,
        controls=None,
        arg_value=None,
        arg_label=None,
        index=None,
        classical_controls=None,
        control_value=None,
        classical_control_value=None,
        style=None,
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
            if isinstance(gate, type) and issubclass(gate, Gate):
                gate_class = gate
            elif gate in GATE_CLASS_MAP:
                gate_class = GATE_CLASS_MAP[gate]
            else:
                raise ValueError(
                    "Can only pass standard gate name as strings"
                    "or Gate class or its object instantiation"
                )

            if issubclass(gate_class, ControlledParamGate):
                gate = gate_class(
                    targets=targets,
                    controls=controls,
                    control_value=control_value,
                    arg_value=arg_value,
                    arg_label=arg_label,
                    classical_controls=classical_controls,
                    classical_control_value=classical_control_value,
                    style=style,
                )

            elif gate_class == GLOBALPHASE:
                gate = gate_class(
                    arg_value=arg_value,
                    arg_label=arg_label,
                    classical_controls=classical_controls,
                    classical_control_value=classical_control_value,
                    style=style,
                )

            elif issubclass(gate_class, ParametrizedGate):
                gate = gate_class(
                    targets=targets,
                    arg_value=arg_value,
                    arg_label=arg_label,
                    classical_controls=classical_controls,
                    classical_control_value=classical_control_value,
                    style=style,
                )

            elif issubclass(gate_class, ControlledGate):
                gate = gate_class(
                    targets=targets,
                    controls=controls,
                    control_value=control_value,
                    classical_controls=classical_controls,
                    classical_control_value=classical_control_value,
                    style=style,
                )

            else:
                gate = gate_class(
                    targets=targets,
                    classical_controls=classical_controls,
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

    def add_circuit(self, qc, start=0):
        """
        Adds a block of a qubit circuit to the main circuit.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            The circuit block to be added to the main circuit.
        start : int
            The qubit on which the first gate is applied.
        """
        if self.N - start < qc.N:
            raise NotImplementedError("Targets exceed number of qubits.")

        for circuit_op in qc.gates:
            if isinstance(circuit_op, Gate):
                if circuit_op.targets is not None:
                    tar = [target + start for target in circuit_op.targets]
                else:
                    tar = None
                if isinstance(circuit_op, ControlledGate):
                    ctrl = [control + start for control in circuit_op.controls]
                else:
                    ctrl = None

                arg = None
                if isinstance(circuit_op, ParametrizedGate):
                    circuit_op.arg_value

                self.add_gate(
                    circuit_op.name,
                    targets=tar,
                    controls=ctrl,
                    arg_value=arg,
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
            self.N,
            reverse_states=self.reverse_states,
            num_cbits=self.num_cbits,
            input_states=self.input_states,
            output_states=self.output_states,
        )

        for circuit_op in reversed(self.gates):
            if isinstance(circuit_op, Gate):
                temp.add_gate(circuit_op)  # TODO add other arguments like target
            else:
                temp.add_measurement(circuit_op)

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

    def run_statistics(
        self, state, cbits=None, precompute_unitary=False
    ):
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

    def resolve_gates(self, basis=["CNOT", "RX", "RY", "RZ"]):
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

        basis_1q = []
        basis_2q = []
        if isinstance(basis, list):
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

        match = False
        temp_resolved = QubitCircuit(self.N)
        for gate in self.gates:
            if gate.name in ("X", "Y", "Z"):
                temp_resolved.add_gate("GLOBALPHASE", arg_value=np.pi / 2)

                if gate.name == "X":
                    gate = RX(targets=gate.targets, arg_value=np.pi)
                elif gate.name == "Y":
                    gate = RY(targets=gate.targets, arg_value=np.pi)
                else:
                    gate = RZ(targets=gate.targets, arg_value=np.pi)
                temp_resolved.add_gate(gate)

            else:
                try:
                    _resolve_to_universal(gate, temp_resolved, basis_1q, basis_2q)
                except KeyError:
                    if gate.name in basis:
                        temp_resolved.add_gate(gate)
                    else:
                        exception = f"Gate {gate.name} cannot be resolved."
                        raise NotImplementedError(exception)

        for basis_unit in ["CSIGN", "ISWAP", "SQRTSWAP", "SQRTISWAP"]:
            if basis_unit in basis_2q:
                match = True
                _resolve_2q_basis(basis_unit, qc_temp, temp_resolved)
                break
        if not match:
            qc_temp.gates = temp_resolved.gates

        if len(basis_1q) == 2:
            temp_resolved.gates = qc_temp.gates
            qc_temp.gates = []
            half_pi = np.pi / 2

            for gate in temp_resolved.gates:
                if gate.name == "RX" and "RX" not in basis_1q:
                    qc_temp.add_gate(
                        "RY",
                        targets=gate.targets,
                        arg_value=-half_pi,
                        arg_label=r"-\pi/2",
                    )
                    qc_temp.add_gate(
                        "RZ",
                        targets=gate.targets,
                        arg_value=gate.arg_value,
                        arg_label=gate.arg_label,
                    )
                    qc_temp.add_gate(
                        "RY",
                        targets=gate.targets,
                        arg_value=-half_pi,
                        arg_label=r"\pi/2",
                    )

                elif gate.name == "RY" and "RY" not in basis_1q:
                    qc_temp.add_gate(
                        "RZ",
                        targets=gate.targets,
                        arg_value=-half_pi,
                        arg_label=r"-\pi/2",
                    )
                    qc_temp.add_gate(
                        "RX",
                        targets=gate.targets,
                        arg_value=gate.arg_value,
                        arg_label=gate.arg_label,
                    )
                    qc_temp.add_gate(
                        "RZ",
                        targets=gate.targets,
                        arg_value=half_pi,
                        arg_label=r"\pi/2",
                    )

                elif gate.name == "RZ" and "RZ" not in basis_1q:
                    qc_temp.add_gate(
                        "RX",
                        targets=gate.targets,
                        arg_value=-half_pi,
                        arg_label=r"-\pi/2",
                    )
                    qc_temp.add_gate(
                        "RY",
                        targets=gate.targets,
                        arg_value=gate.arg_value,
                        arg_label=gate.arg_label,
                    )
                    qc_temp.add_gate(
                        "RX",
                        targets=gate.targets,
                        arg_value=half_pi,
                        arg_label=r"\pi/2",
                    )
                else:
                    qc_temp.add_gate(gate)

        return qc_temp

    def propagators(self, expand=True, ignore_measurement=False):
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
                qobj = gate.get_compact_qobj()
                if expand:
                    all_targets = gate.get_all_qubits()
                    qobj = expand_operator(
                        qobj, dims=self.dims, targets=all_targets
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
