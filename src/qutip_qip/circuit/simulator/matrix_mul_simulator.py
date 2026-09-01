from itertools import product
from operator import mul
from functools import reduce
from typing import Type
import string
import numpy as np

from qutip import ket2dm, Qobj, einsum
from qutip.settings import settings
from qutip.measurement import measurement_statistics
from qutip_qip.circuit.simulator import CircuitResult
from qutip_qip.operations import expand_operator, Gate, Measurement
from qutip_qip.typing import IntSequence


def _decimal_to_binary(decimal, length):
    binary = [int(s) for s in "{0:#b}".format(decimal)[2:]]
    return [0] * (length - len(binary)) + binary


def _check_classical_control_value(classical_controls, classical_control_value, cbits):
    """Check if the gate should be executed, depending on the current value of classical bits."""
    matched = np.empty(len(classical_controls), dtype=bool)
    cbits_conditions = _decimal_to_binary(
        classical_control_value,
        len(classical_controls),
    )
    for i in range(len(classical_controls)):
        cbit_index = classical_controls[i]
        control_value = cbits_conditions[i]
        matched[i] = cbits[cbit_index] == control_value
    return all(matched)


class CircuitSimulator:
    """
    Operator based circuit simulator.
    """

    def __init__(self, qc, mode: str = "state_vector_simulator") -> None:
        """
        Simulate state evolution for Quantum Circuits.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            Quantum Circuit to be simulated.

        mode: string, optional
            Specify if input state (and therefore computation) is in
            state-vector mode or in density matrix mode.
            In state_vector_simulator mode, the input must be a ket
            and with each measurement, one of the collapsed
            states is the new state (when using run()).
            In density_matrix_simulator mode, the input can be a ket or a
            density matrix and after measurement, the new state is the
            mixed ensemble state obtained after the measurement.
            If in density_matrix_simulator mode and given
            a state vector input, the output must be assumed to
            be a density matrix.
        """

        self._qc = qc
        self.dims = qc.dims
        self.mode = mode

    @property
    def qc(self):
        return self._qc

    def initialize(self, state=None, cbits=None, measure_results=None):
        """
        Reset Simulator state variables to start a new run.

        Parameters
        ----------
        state: ket or oper
            ket or density matrix

        cbits: list of int, optional
            initial value of classical bits

        measure_results : tuple of ints, optional
            optional specification of each measurement result to enable
            post-selection. If specified, the measurement results are
            set to the tuple of bits (sequentially) instead of being
            chosen at random.
        """
        # Initializing the unitary operators.
        if cbits and len(cbits) == self.qc.num_cbits:
            self.cbits = cbits
        elif self.qc.num_cbits > 0:
            self.cbits = [0] * self.qc.num_cbits
        else:
            self.cbits = None

        # Parameters that will be updated during the simulation.
        # self._state keeps track of the current state of the evolution.
        # It is not guaranteed to be a Qobj and could be reshaped.
        # Use self.state to return the Qobj representation.
        if state is not None:
            if self.mode == "density_matrix_simulator" and state.isket:
                self._state = ket2dm(state)
            else:
                state = np.exp(1j * self.qc.global_phase) * state
                self._state = state

            # Reconstruct CuState with multipartite hilbert_dims for CuOperator mode mapping
            if type(self._state.data).__name__ == "CuState":
                from qutip_cuquantum.state import CuState as CuStateClass

                hilbert_dims = tuple(state.dims[0])
                self._state = Qobj(
                    CuStateClass(self._state.data.to_cupy(), hilbert_dims=hilbert_dims),
                    dims=self._state.dims,
                )
        else:
            # Just computing the full unitary, no state
            self._state = None

        self._state_dims = state.dims.copy()  # Record the dimension of the state.
        self._probability = 1
        self._op_index = 0
        self._measure_results = measure_results
        self._measure_ind = 0
        if self.mode == "state_vector_simulator":
            self._tensor_dims = self._state_dims[0].copy()
            if state.type == "oper":
                # apply the gate to a unitary, add an ancillary axis.
                self._state_mat_shape = [reduce(mul, self._state_dims[0], 1)] * 2
                self._tensor_dims += [reduce(mul, self._state_dims[0], 1)]
            else:
                self._state_mat_shape = [
                    reduce(mul, self._state_dims[0], 1),
                    1,
                ]
            self._tensor_dims = tuple(self._tensor_dims)
            self._state_mat_shape = tuple(self._state_mat_shape)

    @property
    def state(self):
        """
        The current state of the simulator as a `qutip.Qobj`

        Returns:
            `qutip.Qobj`: The current state of the simulator.
        """
        if not isinstance(self._state, Qobj) and self._state is not None:
            self._state = self._state.reshape(self._state_mat_shape)
            return Qobj(self._state, dims=self._state_dims)
        else:
            return self._state

    def run(self, state, cbits=None, measure_results=None):
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
        result: CircuitResult
            Return a CircuitResult object containing
            output state and probability.
        """
        self.initialize(state, cbits, measure_results)

        for _ in range(len(self._qc.instructions)):
            self.step()
            if self._state is None:
                # TODO This only happens if there is predefined post-selection on the measurement results and the measurement results is exactly 0. This needs to be improved.
                break

        return CircuitResult(self.state, self._probability, self.cbits)

    def run_statistics(self, state, cbits=None):
        """
        Calculate all the possible outputs of a circuit
        (varied by measurement gates).

        Parameters
        ----------
        state : ket
                state to be observed on specified by density matrix.
        cbits : List of ints, optional
                initialization of the classical bits.

        Returns
        -------
        result: CircuitResult
            Return a CircuitResult object containing
            output states and and their probabilities.
        """

        probabilities = []
        states = []
        cbits_results = []

        num_measurements = len(
            list(
                filter(
                    lambda x: x.is_measurement_instruction(),
                    self.qc.instructions,
                )
            )
        )

        for results in product("01", repeat=num_measurements):
            run_result = self.run(state, cbits=cbits, measure_results=results)
            final_state = run_result.get_final_states(0)
            probability = run_result.get_probabilities(0)
            states.append(final_state)
            probabilities.append(probability)
            cbits_results.append(self.cbits)

        return CircuitResult(states, probabilities, cbits_results)

    def step(self):
        """
        Return state after one step of circuit evolution
        (gate or measurement).

        Returns
        -------
        state : ket or oper
            state after one evolution step.
        """

        circ_instruction = self.qc.instructions[self._op_index]
        current_state = self._state

        if self.qc.instructions[self._op_index].is_measurement_instruction():
            targets = circ_instruction.qubits
            classical_store = circ_instruction.cbits
            if self.mode in ("state_vector_simulator", "density_matrix_simulator"):
                state = self._apply_measurement_einsum(
                    circ_instruction.operation, targets, classical_store
                )
            else:
                state = self._apply_measurement(
                    circ_instruction.operation, targets, classical_store
                )

        elif self.qc.instructions[self._op_index].is_gate_instruction():
            gate = circ_instruction.operation
            qubits = circ_instruction.qubits
            classical_controls = circ_instruction.cbits
            classical_control_value = circ_instruction.cbits_ctrl_value

            if len(classical_controls) > 0:
                apply_gate = _check_classical_control_value(
                    classical_controls, classical_control_value, self.cbits
                )
            else:
                apply_gate = True

            if not apply_gate:
                self._op_index += 1
                return
            if self.mode == "state_vector_simulator":
                state = self._evolve_state_einsum(gate, qubits, current_state)
            elif self.mode == "density_matrix_simulator":
                state = self._evolve_state_einsum_dm(gate, qubits, current_state)
            else:
                state = self._evolve_state(gate, qubits, current_state)

        else:
            raise ValueError(
                f"Invalid operation {self.qc.instructions[self._op_index]}"
            )

        self._state = state
        self._op_index += 1

    def _generate_einsum_eq(
        self,
        targets: int | IntSequence,
        num_qubits: int,
        num_cols: int = 1,
        is_oper: bool = False,
    ) -> str:
        """
        Generates the einsum string for tensor contraction supporting up to 52 qubits.
        Uses standard ASCII letters (a-z, A-Z) to map input and output indices.

        Parameters
        ----------
        targets : int or list of int
            The target qubits the gate acts on.
        num_qubits : int
            The total number of qubits (tensor dimensions) in the state.
        num_cols : int, optional
            The number of column dimensions in the state representation.
        is_oper : bool, optional
            Whether the state is an operator (e.g. unitary matrix) or ket vector.

        Returns
        -------
        eq : str
            The einsum equation string (e.g., "ab,cde->cde").
        """
        if isinstance(targets, int):
            targets = [targets]

        chars = string.ascii_letters

        row_in = list(chars[:num_qubits])
        k = len(targets)
        gate_out = list(chars[num_qubits : num_qubits + k])
        gate_in = [row_in[t] for t in targets]

        row_out = row_in.copy()
        for i, t in enumerate(targets):
            row_out[t] = gate_out[i]

        sub_gate = "".join(gate_out + gate_in)

        if is_oper and num_cols == 1:
            num_cols = num_qubits

        col_in = list(chars[num_qubits + k : num_qubits + k + num_cols])
        sub_state = "".join(row_in + col_in)
        sub_out = "".join(row_out + col_in)

        return f"{sub_gate},{sub_state}->{sub_out}"

    def _generate_dm_einsum_eq(
        self, targets: int | IntSequence, num_qubits: int
    ) -> str:
        r"""
        Generates the einsum string for density matrix tensor contraction U rho U^\dagger.

        Parameters
        ----------
        targets : int or list of int
            The target qubits the gate acts on.
        num_qubits : int
            The total number of qubits in the state.

        Returns
        -------
        eq : str
            The einsum equation string (e.g., "ghac,abcdef,dfij->gbhiej").
        """
        if isinstance(targets, int):
            targets = [targets]

        chars = string.ascii_letters

        row_in = list(chars[:num_qubits])
        col_in = list(chars[num_qubits : 2 * num_qubits])

        offset = 2 * num_qubits
        k = len(targets)
        gate_out_rows = list(chars[offset : offset + k])
        gate_out_cols = list(chars[offset + k : offset + 2 * k])

        gate_in_rows = [row_in[t] for t in targets]
        gate_in_cols = [col_in[t] for t in targets]

        row_out = row_in.copy()
        col_out = col_in.copy()
        for i, t in enumerate(targets):
            row_out[t] = gate_out_rows[i]
            col_out[t] = gate_out_cols[i]

        sub_U = "".join(gate_out_rows + gate_in_rows)
        sub_rho = "".join(row_in + col_in)
        sub_Udag = "".join(gate_in_cols + gate_out_cols)
        sub_out = "".join(row_out + col_out)

        return f"{sub_U},{sub_rho},{sub_Udag}->{sub_out}"

    def _evolve_state(
        self, operation: Gate | Qobj, targets_indices: int | IntSequence, state: Qobj
    ) -> Qobj:
        """
         Applies a unitary gate to the quantum state using operator expansion.

         Parameters
         ----------
        operation : :class:`.Gate` or :class:`qutip.Qobj`
             The quantum gate or operator to be applied.
         targets_indices : int or sequence of int
             The indices of the target qubits.
         state : :class:`qutip.Qobj`
             The current quantum state (ket or density matrix).

         Returns
         -------
         state : :class:`qutip.Qobj`
             The updated quantum state.
        """
        state_dtype = type(state.data).__name__
        gate_dtype = "CuOperator" if state_dtype == "CuState" else state_dtype

        # Construct CuOperator with explicit hilbert_dims and target mode mapping
        # so cuQuantum knows which qubit sites to act on (calling .to('CuOperator')
        # on raw gate Qobjs loses multipartite mode information).
        gate_qobj = (
            operation.get_qobj() if hasattr(operation, "get_qobj") else operation
        )
        if gate_dtype == "CuOperator":
            from qutip_cuquantum.operator import CuOperator as CuOperatorClass

            U = Qobj(
                CuOperatorClass(
                    gate_qobj.data,
                    hilbert_dims=tuple(gate_qobj.dims[0]),
                    mode=tuple(range(len(gate_qobj.dims[0]))),
                ),
                dims=gate_qobj.dims,
            )
        else:
            U = gate_qobj.to(gate_dtype)

        U = expand_operator(
            U,
            dims=self.dims,
            targets=targets_indices,
            dtype=gate_dtype,
        )
        if self.mode == "state_vector_simulator":
            state = U * state
        elif self.mode == "density_matrix_simulator":
            state = U * state * U.dag()
        else:
            raise NotImplementedError(f"mode {self.mode} is not available.")
        return state

    def _evolve_state_einsum(
        self, operation: Gate, targets_indices: int | IntSequence, state: Qobj
    ) -> Qobj:
        """
        Applies a gate to the state using tensor contraction (einsum).

        Parameters
        ----------
        operation : :class:`.Gate`
            The quantum gate to be applied.
        targets_indices : int or sequence of int
            The indices of the target qubits.
        state : :class:`qutip.Qobj`
            The current quantum state vector or operator.

        Returns
        -------
        state : :class:`qutip.Qobj`
            The updated quantum state.
        """
        state_dtype = type(state.data).__name__
        # There is no einsum specialisation registered for CuState/CuOperator, so einsum
        # falls back to CPU NumPy. We route to matrix mul (_evolve_state) to stay on GPU.
        if state_dtype == "CuState":
            return self._evolve_state(operation, targets_indices, state)

        gate_dtype = state_dtype

        gate_qobj = operation.get_qobj().to(gate_dtype)

        num_dims = len(self._state_dims[0])
        num_cols = len(state.dims[1])
        is_oper = state.isoper
        eq = self._generate_einsum_eq(
            targets_indices, num_dims, num_cols=num_cols, is_oper=is_oper
        )

        return einsum(eq, gate_qobj, state)

    def _evolve_state_einsum_dm(
        self, operation: Gate, targets_indices: int | IntSequence, state: Qobj
    ) -> Qobj:
        """
        Applies a gate to the density matrix state using tensor contraction (einsum).

        Parameters
        ----------
        operation : :class:`.Gate`
            The quantum gate to be applied.
        targets_indices : int or sequence of int
            The indices of the target qubits.
        state : :class:`qutip.Qobj`
            The current quantum density matrix state.

        Returns
        -------
        state : :class:`qutip.Qobj`
            The updated quantum density matrix state.
        """
        state_dtype = type(state.data).__name__
        # There is no einsum specialisation registered for CuState/CuOperator, so einsum
        # falls back to CPU NumPy. We route to matrix mul (_evolve_state) to stay on GPU.
        if state_dtype == "CuState":
            return self._evolve_state(operation, targets_indices, state)

        gate_dtype = state_dtype

        gate_qobj = operation.get_qobj().to(gate_dtype)

        num_dims = len(self._state_dims[0])
        eq = self._generate_dm_einsum_eq(targets_indices, num_dims)

        return einsum(eq, gate_qobj, state, gate_qobj.dag())

    def _apply_measurement(
        self,
        operation: Type[Measurement],
        qubits: tuple[int, ...],
        cbits: tuple[int, ...],
    ) -> Qobj:
        """
        Applies measurement gate specified by operation to current state.

        Parameters
        ----------
        operation: :class:`.Measurement` or Type[:class:`.Measurement`]
            Measurement gate in a circuit object or its class.
        qubits : tuple of int
            The indices of the qubits to be measured.
        cbits : tuple of int
            The indices of the classical registers where the measurement
            results will be stored.

        Returns
        -------
        state : qutip.Qobj
            The collapsed state after the measurement.
        """
        current_state = self.state
        n = self.qc.num_qubits
        if isinstance(operation, type) and issubclass(operation, Measurement):
            operation = operation()
        raw_ops = operation.get_measurement_ops()
        measurement_ops = [
            expand_operator(oper=op, dims=[2] * n, targets=qubits) for op in raw_ops
        ]

        states, probabilities = measurement_statistics(current_state, measurement_ops)

        if self.mode == "state_vector_simulator":
            if self._measure_results:
                i = int(self._measure_results[self._measure_ind])
                self._measure_ind += 1
            else:
                probabilities = [p / sum(probabilities) for p in probabilities]
                outcome_indices = np.arange(len(probabilities))
                i = np.random.choice(outcome_indices, p=probabilities)
            self._probability *= probabilities[i]
            state = states[i]
            if cbits:
                cbit_index = cbits[0]
                self.cbits[cbit_index] = i

        elif self.mode == "density_matrix_simulator":
            states = list(filter(lambda x: x is not None, states))
            probabilities = list(filter(lambda x: x != 0, probabilities))
            state = sum(p * s for s, p in zip(states, probabilities))

        else:
            raise NotImplementedError(f"mode {self.mode} is not available.")

        return state

    def _apply_measurement_einsum(
        self,
        operation: Measurement | Type[Measurement],
        qubits: tuple[int, ...],
        cbits: tuple[int, ...],
    ) -> Qobj:
        """
        Applies measurement gate specified by operation using tensor contraction (einsum).

        Parameters
        ----------
        operation : :class:`.Measurement` or Type[:class:`.Measurement`]
            Measurement gate in a circuit object or its class.
        qubits : tuple of int
            The indices of the qubits to be measured.
        cbits : tuple of int
            The indices of the classical registers where the measurement
            results will be stored.

        Returns
        -------
        state : :class:`qutip.Qobj`
            The collapsed state after the measurement.
        """
        current_state = self.state
        num_qubits = self.qc.num_qubits
        if isinstance(operation, type) and issubclass(operation, Measurement):
            operation = operation()

        state_dtype = type(current_state.data).__name__
        raw_ops = operation.get_measurement_ops()
        # Ops for CuState are converted to CuOperator in _evolve_state()
        if state_dtype != "CuState":
            raw_ops = [op.to(state_dtype) for op in raw_ops]

        num_cols = len(current_state.dims[1])

        states = []
        probabilities = []
        tol = settings.core["atol"]

        if self.mode == "state_vector_simulator":
            eq = self._generate_einsum_eq(
                qubits, num_qubits, num_cols=num_cols, is_oper=current_state.isoper
            )

            for op in raw_ops:
                if state_dtype == "CuState":
                    unnorm_state = self._evolve_state(op, qubits, current_state)
                else:
                    unnorm_state = einsum(eq, op, current_state)
                p = float(np.real(unnorm_state.overlap(unnorm_state)))

                if p >= tol:
                    states.append(unnorm_state / np.sqrt(p))
                    probabilities.append(p)
                else:
                    states.append(None)
                    probabilities.append(0.0)

            if self._measure_results:
                i = int(self._measure_results[self._measure_ind])
                self._measure_ind += 1
            else:
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    norm_probs = [p / prob_sum for p in probabilities]
                else:
                    norm_probs = [1.0 / len(probabilities)] * len(probabilities)
                i = np.random.choice(len(probabilities), p=norm_probs)

            self._probability *= probabilities[i]
            state = states[i]
            if cbits:
                cbit_index = cbits[0]
                self.cbits[cbit_index] = i

        elif self.mode == "density_matrix_simulator":
            eq = self._generate_dm_einsum_eq(qubits, num_qubits)
            unnorm_states = []

            for op in raw_ops:
                if state_dtype == "CuState":
                    unnorm_rho = self._evolve_state(op, qubits, current_state)
                else:
                    unnorm_rho = einsum(eq, op, current_state, op.dag())
                p = float(np.real(unnorm_rho.tr()))

                if p >= tol:
                    states.append(unnorm_rho / p)
                    probabilities.append(p)
                    unnorm_states.append(unnorm_rho)
                else:
                    states.append(None)
                    probabilities.append(0.0)

            if self._measure_results:
                i = int(self._measure_results[self._measure_ind])
                self._measure_ind += 1
                self._probability *= probabilities[i]
                state = states[i]
                if cbits:
                    self.cbits[cbits[0]] = i
            else:
                state = sum(unnorm_states) if unnorm_states else current_state

        else:
            raise NotImplementedError(f"mode {self.mode} is not available.")

        return state
