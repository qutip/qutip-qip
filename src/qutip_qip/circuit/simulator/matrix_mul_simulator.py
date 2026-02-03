from itertools import product
from operator import mul
from functools import reduce
import numpy as np

from qutip import ket2dm, Qobj
from qutip_qip.circuit.simulator import CircuitResult
from qutip_qip.operations import expand_operator
import warnings


def _decimal_to_binary(decimal, length):
    binary = [int(s) for s in "{0:#b}".format(decimal)[2:]]
    return [0] * (length - len(binary)) + binary


def _check_classical_control_value(
    classical_controls, classical_control_value, cbits
):
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

    def __init__(
        self,
        qc,
        mode: str = "state_vector_simulator",
        precompute_unitary: bool = False,
    ):
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
        if precompute_unitary:
            warnings.warn(
                "Precomputing the full unitary is no longer supported. Switching to normal simulation mode."
            )

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
        else:
            # Just computing the full unitary, no state
            self._state = None
        self._state_dims = (
            state.dims.copy()
        )  # Record the dimension of the state.
        self._probability = 1
        self._op_index = 0
        self._measure_results = measure_results
        self._measure_ind = 0
        if self.mode == "state_vector_simulator":
            self._tensor_dims = self._state_dims[0].copy()
            if state.type == "oper":
                # apply the gate to a unitary, add an ancillary axis.
                self._state_mat_shape = [
                    reduce(mul, self._state_dims[0], 1)
                ] * 2
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

        op = self.qc.instructions[self._op_index].operation
        current_state = self._state

        if self.qc.instructions[self._op_index].is_measurement_instruction():
            state = self._apply_measurement(op, current_state)

        elif self.qc.instructions[self._op_index].is_gate_instruction():
            qubits = self.qc.instructions[self._op_index].qubits
            classical_controls = self.qc.instructions[self._op_index].cbits
            classical_control_value = self.qc.instructions[
                self._op_index
            ].control_value

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
                state = self._evolve_state_einsum(op, qubits, current_state)
            else:
                state = self._evolve_state(op, qubits, current_state)

        self._state = state
        self._op_index += 1

    def _evolve_state(self, operation, targets_indices, state):
        """
        Applies unitary to state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        """
        U = operation.get_compact_qobj()
        U = expand_operator(
            U,
            dims=self.dims,
            targets=targets_indices,
        )
        if self.mode == "state_vector_simulator":
            state = U * state
        elif self.mode == "density_matrix_simulator":
            state = U * state * U.dag()
        else:
            raise NotImplementedError(f"mode {self.mode} is not available.")
        return state

    def _evolve_state_einsum(self, gate, targets_indices, state):
        # Prepare the state tensor.
        if isinstance(state, Qobj):
            # If it is a Qobj, transform it to the array representation.
            state = state.full()
            # Transform the gate and state array to the corresponding
            # tensor form.
            state = state.reshape(self._tensor_dims)

        # Prepare the gate tensor.
        gate = gate.get_compact_qobj()
        gate_array = gate.full().reshape(gate.dims[0] + gate.dims[1])

        # Compute the tensor indices and call einsum.
        num_site = len(state.shape)
        ancillary_indices = range(num_site, num_site + len(targets_indices))
        index_list = range(num_site)
        new_index_list = list(index_list)
        for j, k in enumerate(targets_indices):
            new_index_list[k] = j + num_site

        state = np.einsum(
            gate_array,
            list(ancillary_indices) + list(targets_indices),
            state,
            index_list,
            new_index_list,
        )
        return state

    def _apply_measurement(self, operation, state):
        """
        Applies measurement gate specified by operation to current state.

        Parameters
        ----------
        operation: :class:`.Measurement`
            Measurement gate in a circuit object.
        """
        states, probabilities = operation.measurement_comp_basis(self.state)

        if self.mode == "state_vector_simulator":
            if self._measure_results:
                i = int(self._measure_results[self._measure_ind])
                self._measure_ind += 1
            else:
                probabilities = [p / sum(probabilities) for p in probabilities]
                i = np.random.choice([0, 1], p=probabilities)
            self._probability *= probabilities[i]
            state = states[i]
            if operation.classical_store is not None:
                self.cbits[operation.classical_store] = i

        elif self.mode == "density_matrix_simulator":
            states = list(filter(lambda x: x is not None, states))
            probabilities = list(filter(lambda x: x != 0, probabilities))
            state = sum(p * s for s, p in zip(states, probabilities))

        else:
            raise NotImplementedError(f"mode {self.mode} is not available.")

        return state
