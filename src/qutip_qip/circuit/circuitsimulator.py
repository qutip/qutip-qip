from itertools import product, chain
from operator import mul
from functools import reduce
import numpy as np

from ..operations import (
    Gate,
    Measurement,
    expand_operator,
)
from qutip import basis, ket2dm, Qobj, tensor
import warnings


__all__ = ["CircuitSimulator", "CircuitResult"]


def _flatten(lst):
    """
    Helper to flatten lists.
    """

    return [item for sublist in lst for item in sublist]


def _mult_sublists(tensor_list, overall_inds, U, inds):
    """
    Calculate the revised indices and tensor list by multiplying a new unitary
    U applied to inds.

    Parameters
    ----------
    tensor_list : list of Qobj
        List of gates (unitaries) acting on disjoint qubits.

    overall_inds : list of list of int
        List of qubit indices corresponding to each gate in tensor_list.

    U: Qobj
        Unitary to be multiplied with the the unitary specified by tensor_list.

    inds: list of int
        List of qubit indices corresponding to U.

    Returns
    -------
    tensor_list_revised: list of Qobj
        List of gates (unitaries) acting on disjoint qubits incorporating U.

    overall_inds_revised: list of list of int
        List of qubit indices corresponding to each gate in tensor_list_revised.

    Examples
    --------

    First, we get some imports out of the way,

    >>> from qutip_qip.operations.gates import _mult_sublists
    >>> from qutip_qip.operations.gates import x_gate, y_gate, toffoli, z_gate

    Suppose we have a unitary list of already processed gates,
    X, Y, Z applied on qubit indices 0, 1, 2 respectively and
    encounter a new TOFFOLI gate on qubit indices (0, 1, 3).

    >>> tensor_list = [x_gate(), y_gate(), z_gate()]
    >>> overall_inds = [[0], [1], [2]]
    >>> U = toffoli()
    >>> U_inds = [0, 1, 3]

    Then, we can use _mult_sublists to produce a new list of unitaries by
    multiplying TOFFOLI (and expanding) only on the qubit indices involving
    TOFFOLI gate (and any multiplied gates).

    >>> U_list, overall_inds = _mult_sublists(tensor_list, overall_inds, U, U_inds)
    >>> np.testing.assert_allclose(U_list[0]) == z_gate())
    >>> toffoli_xy = toffoli() * tensor(x_gate(), y_gate(), identity(2))
    >>> np.testing.assert_allclose(U_list[1]), toffoli_xy)
    >>> overall_inds = [[2], [0, 1, 3]]
    """

    tensor_sublist = []
    inds_sublist = []

    tensor_list_revised = []
    overall_inds_revised = []

    for sub_inds, sub_U in zip(overall_inds, tensor_list):
        if len(set(sub_inds).intersection(inds)) > 0:
            tensor_sublist.append(sub_U)
            inds_sublist.append(sub_inds)
        else:
            overall_inds_revised.append(sub_inds)
            tensor_list_revised.append(sub_U)

    inds_sublist = _flatten(inds_sublist)
    U_sublist = tensor(tensor_sublist)

    revised_inds = list(set(inds_sublist).union(set(inds)))
    N = len(revised_inds)

    sorted_positions = sorted(range(N), key=lambda key: revised_inds[key])
    ind_map = {ind: pos for ind, pos in zip(revised_inds, sorted_positions)}

    U_sublist = expand_operator(
        U_sublist, dims=[2] * N, targets=[ind_map[ind] for ind in inds_sublist]
    )
    U = expand_operator(
        U, dims=[2] * N, targets=[ind_map[ind] for ind in inds]
    )

    U_sublist = U * U_sublist
    inds_sublist = revised_inds

    overall_inds_revised.append(inds_sublist)
    tensor_list_revised.append(U_sublist)

    return tensor_list_revised, overall_inds_revised


def _expand_overall(tensor_list, overall_inds):
    """
    Tensor unitaries in tensor list and then use expand_operator to rearrange
    them appropriately according to the indices in overall_inds.
    """

    U_overall = tensor(tensor_list)
    overall_inds = _flatten(overall_inds)
    U_overall = expand_operator(
        U_overall, dims=[2] * len(overall_inds), targets=overall_inds
    )
    overall_inds = sorted(overall_inds)
    return U_overall, overall_inds


def _gate_sequence_product(U_list, ind_list):
    """
    Calculate the overall unitary matrix for a given list of unitary operations
    that are still of original dimension.

    Parameters
    ----------
    U_list : list of Qobj
        List of gates(unitaries) implementing the quantum circuit.

    ind_list : list of list of int
        List of qubit indices corresponding to each gate in tensor_list.

    Returns
    -------
    U_overall : qobj
        Unitary matrix corresponding to U_list.

    overall_inds : list of int
        List of qubit indices on which U_overall applies.

    Examples
    --------

    First, we get some imports out of the way,

    >>> from qutip_qip.operations.gates import _gate_sequence_product
    >>> from qutip_qip.operations.gates import x_gate, y_gate, toffoli, z_gate

    Suppose we have a circuit with gates X, Y, Z, TOFFOLI
    applied on qubit indices 0, 1, 2 and [0, 1, 3] respectively.

    >>> tensor_lst = [x_gate(), y_gate(), z_gate(), toffoli()]
    >>> overall_inds = [[0], [1], [2], [0, 1, 3]]

    Then, we can use _gate_sequence_product to produce a single unitary
    obtained by multiplying unitaries in the list using heuristic methods
    to reduce the size of matrices being multiplied.

    >>> U_list, overall_inds = _gate_sequence_product(tensor_lst, overall_inds)
    """
    num_qubits = len(set(chain(*ind_list)))
    sorted_inds = sorted(set(_flatten(ind_list)))
    ind_list = [[sorted_inds.index(ind) for ind in inds] for inds in ind_list]

    U_overall = 1
    overall_inds = []

    for i, (U, inds) in enumerate(zip(U_list, ind_list)):
        # when the tensor_list covers the full dimension of the circuit, we
        # expand the tensor_list to a unitary and call _gate_sequence_product
        # recursively on the rest of the U_list.
        if len(overall_inds) == 1 and len(overall_inds[0]) == num_qubits:
            U_overall, overall_inds = _expand_overall(
                tensor_list, overall_inds
            )
            U_left, rem_inds = _gate_sequence_product(U_list[i:], ind_list[i:])
            U_left = expand_operator(
                U_left, dims=[2] * num_qubits, targets=rem_inds
            )
            return U_left * U_overall, [
                sorted_inds[ind] for ind in overall_inds
            ]

        # special case for first unitary in the list
        if U_overall == 1:
            U_overall = U_overall * U
            overall_inds = [ind_list[0]]
            tensor_list = [U_overall]
            continue

        # case where the next unitary interacts on some subset of qubits
        # with the unitaries already in tensor_list.
        elif len(set(_flatten(overall_inds)).intersection(set(inds))) > 0:
            tensor_list, overall_inds = _mult_sublists(
                tensor_list, overall_inds, U, inds
            )

        # case where the next unitary does not interact with any unitary in
        # tensor_list
        else:
            overall_inds.append(inds)
            tensor_list.append(U)

    U_overall, overall_inds = _expand_overall(tensor_list, overall_inds)

    return U_overall, [sorted_inds[ind] for ind in overall_inds]


def _gate_sequence_product_with_expansion(U_list, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary
    operations, assuming that all operations have the same dimension.
    This is only for backward compatibility.

    Parameters
    ----------
    U_list : list
        List of gates(unitaries) implementing the quantum circuit.

    left_to_right : Boolean
        Check if multiplication is to be done from left to right.

    Returns
    -------
    U_overall : qobj
        Unitary matrix corresponding to U_list.
    """

    U_overall = 1
    for U in U_list:
        if left_to_right:
            U_overall = U * U_overall
        else:
            U_overall = U_overall * U

    return U_overall


class CircuitSimulator:
    """
    Operator based circuit simulator.
    """

    def __init__(
        self,
        qc,
        mode="state_vector_simulator",
        precompute_unitary=False,
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
        for _ in range(len(self._qc.gates)):
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
            list(filter(lambda x: isinstance(x, Measurement), self._qc.gates))
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

        def _decimal_to_binary(decimal, length):
            binary = [int(s) for s in "{0:#b}".format(decimal)[2:]]
            return [0] * (length - len(binary)) + binary

        def _check_classical_control_value(operation, cbits):
            """Check if the gate should be executed, depending on the current value of classical bits."""
            matched = np.empty(len(operation.classical_controls), dtype=bool)
            cbits_conditions = _decimal_to_binary(
                operation.classical_control_value,
                len(operation.classical_controls),
            )
            for i in range(len(operation.classical_controls)):
                cbit_index = operation.classical_controls[i]
                control_value = cbits_conditions[i]
                matched[i] = cbits[cbit_index] == control_value
            return all(matched)

        op = self._qc.gates[self._op_index]
        self._op_index += 1

        current_state = self._state
        if isinstance(op, Measurement):
            state = self._apply_measurement(op, current_state)
        elif isinstance(op, Gate):
            if op.classical_controls is not None:
                apply_gate = _check_classical_control_value(op, self.cbits)
            else:
                apply_gate = True
            if not apply_gate:
                return current_state
            if self.mode == "state_vector_simulator":
                state = self._evolve_state_einsum(op, current_state)
            else:
                state = self._evolve_state(op, current_state)

        self._state = state

    def _evolve_state(self, operation, state):
        """
        Applies unitary to state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        """
        if operation.name == "GLOBALPHASE":
            # This is just a complex number.
            U = np.exp(1.0j * operation.arg_value)
        else:
            # We need to use the circuit because the custom gates
            # are still saved in circuit instance.
            # This should be changed once that is deprecated.
            U = self.qc._get_gate_unitary(operation)
            U = expand_operator(
                U,
                dims=self.dims,
                targets=operation.get_all_qubits(),
            )
        if self.mode == "state_vector_simulator":
            state = U * state
        elif self.mode == "density_matrix_simulator":
            state = U * state * U.dag()
        else:
            raise NotImplementedError(
                "mode {} is not available.".format(self.mode)
            )
        return state

    def _evolve_state_einsum(self, gate, state):
        if gate.name == "GLOBALPHASE":
            # This is just a complex number.
            return np.exp(1.0j * gate.arg_value) * state
        # Prepare the state tensor.
        targets_indices = gate.get_all_qubits()
        if isinstance(state, Qobj):
            # If it is a Qobj, transform it to the array representation.
            state = state.full()
            # Transform the gate and state array to the corresponding
            # tensor form.
            state = state.reshape(self._tensor_dims)
        # Prepare the gate tensor.
        gate = self.qc._get_gate_unitary(gate)
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
            raise NotImplementedError(
                "mode {} is not available.".format(self.mode)
            )
        return state


class CircuitResult:
    """
    Result of a quantum circuit simulation.
    """

    def __init__(self, final_states, probabilities, cbits=None):
        """
        Store result of CircuitSimulator.

        Parameters
        ----------
        final_states: list of Qobj.
            List of output kets or density matrices.

        probabilities: list of float.
            List of probabilities of obtaining each output state.

        cbits: list of list of int, optional
            List of cbits for each output.
        """

        if isinstance(final_states, Qobj) or final_states is None:
            self.final_states = [final_states]
            self.probabilities = [probabilities]
            if cbits:
                self.cbits = [cbits]
        else:
            inds = list(
                filter(
                    lambda x: final_states[x] is not None,
                    range(len(final_states)),
                )
            )
            self.final_states = [final_states[i] for i in inds]
            self.probabilities = [probabilities[i] for i in inds]
            if cbits:
                self.cbits = [cbits[i] for i in inds]

    def get_final_states(self, index=None):
        """
        Return list of output states.

        Parameters
        ----------
        index: int
            Indicates i-th state to be returned.

        Returns
        -------
        final_states: Qobj or list of Qobj.
            List of output kets or density matrices.
        """

        if index is not None:
            return self.final_states[index]
        return self.final_states

    def get_probabilities(self, index=None):
        """
        Return list of probabilities corresponding to the output states.

        Parameters
        ----------
        index: int
            Indicates i-th probability to be returned.

        Returns
        -------
        probabilities: float or list of float
            Probabilities associated with each output state.
        """

        if index is not None:
            return self.probabilities[index]
        return self.probabilities

    def get_cbits(self, index=None):
        """
        Return list of classical bit outputs corresponding to the results.

        Parameters
        ----------
        index: int
            Indicates i-th output, probability pair to be returned.

        Returns
        -------
        cbits: list of int or list of list of int
            list of classical bit outputs
        """

        if index is not None:
            return self.cbits[index]
        return self.cbits
