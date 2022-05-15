from itertools import product
import os

import numpy as np

from . import circuit_latex as _latex
from ..operations import (
    Gate,
    Measurement,
    expand_operator,
    gate_sequence_product,
)
from qutip import basis, ket2dm, Qobj


__all__ = ["CircuitSimulator", "CircuitResult"]


class CircuitSimulator:
    """
    Operator based circuit simulator.
    """

    def __init__(
        self,
        qc,
        state=None,
        cbits=None,
        U_list=None,
        measure_results=None,
        mode="state_vector_simulator",
        precompute_unitary=False,
    ):
        """
        Simulate state evolution for Quantum Circuits.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            Quantum Circuit to be simulated.

        state: ket or oper
            ket or density matrix

        cbits: list of int, optional
            initial value of classical bits

        U_list: list of Qobj, optional
            list of predefined unitaries corresponding to circuit.

        measure_results : tuple of ints, optional
            optional specification of each measurement result to enable
            post-selection. If specified, the measurement results are
            set to the tuple of bits (sequentially) instead of being
            chosen at random.

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

        precompute_unitary: Boolean, optional
            Specify if computation is done by pre-computing and aggregating
            gate unitaries. Possibly a faster method in the case of
            large number of repeat runs with different state inputs.
        """

        self.qc = qc
        self.mode = mode
        self.precompute_unitary = precompute_unitary

        if U_list:
            self.U_list = U_list
        elif precompute_unitary:
            self.U_list = qc.propagators(expand=False, ignore_measurement=True)
        else:
            self.U_list = qc.propagators(ignore_measurement=True)

        self.ops = []
        self.inds_list = []

        if precompute_unitary:
            self._process_ops_precompute()
        else:
            self._process_ops()

        self.initialize(state, cbits, measure_results)

    def _process_ops(self):
        """
        Process list of gates (including measurements), and stores
        them in self.ops (as unitaries) for further computation.
        """

        U_list_index = 0

        for operation in self.qc.gates:
            if isinstance(operation, Measurement):
                self.ops.append(operation)
            elif isinstance(operation, Gate):
                if operation.classical_controls:
                    self.ops.append((operation, self.U_list[U_list_index]))
                else:
                    self.ops.append(self.U_list[U_list_index])
                U_list_index += 1

    def _process_ops_precompute(self):
        """
        Process list of gates (including measurements), aggregate
        gate unitaries (by multiplying) and store them in self.ops
        for further computation. The gate multiplication is carried out
        only for groups of matrices in between classically controlled gates
        and measurement gates.

        Examples
        --------

        If we have a circuit that looks like:

        ----|X|-----|Y|----|M0|-----|X|----

        then self.ops = [YX, M0, X]
        """

        prev_index = 0
        U_list_index = 0

        for gate in self.qc.gates:
            if isinstance(gate, Measurement):
                continue
            else:
                self.inds_list.append(gate.get_all_qubits())

        for operation in self.qc.gates:
            if isinstance(operation, Measurement):
                if U_list_index > prev_index:
                    self.ops.append(
                        self._compute_unitary(
                            self.U_list[prev_index:U_list_index],
                            self.inds_list[prev_index:U_list_index],
                        )
                    )
                    prev_index = U_list_index
                self.ops.append(operation)

            elif isinstance(operation, Gate):
                if operation.classical_controls:
                    if U_list_index > prev_index:
                        self.ops.append(
                            self._compute_unitary(
                                self.U_list[prev_index:U_list_index],
                                self.inds_list[prev_index:U_list_index],
                            )
                        )
                        prev_index = U_list_index
                    self.ops.append((operation, self.U_list[prev_index]))
                    prev_index += 1
                    U_list_index += 1
                else:
                    U_list_index += 1

        if U_list_index > prev_index:
            self.ops.append(
                self._compute_unitary(
                    self.U_list[prev_index:U_list_index],
                    self.inds_list[prev_index:U_list_index],
                )
            )
            prev_index = U_list_index + 1
            U_list_index = prev_index

    def initialize(self, state=None, cbits=None, measure_results=None):
        """
        Reset Simulator state variables to start a new run.

        Parameters
        ----------
        state: ket or oper
            ket or density matrix

        cbits: list of int, optional
            initial value of classical bits

        U_list: list of Qobj, optional
            list of predefined unitaries corresponding to circuit.

        measure_results : tuple of ints, optional
            optional specification of each measurement result to enable
            post-selection. If specified, the measurement results are
            set to the tuple of bits (sequentially) instead of being
            chosen at random.
        """

        if cbits and len(cbits) == self.qc.num_cbits:
            self.cbits = cbits
        elif self.qc.num_cbits > 0:
            self.cbits = [0] * self.qc.num_cbits
        else:
            self.cbits = None

        self.state = None

        if state is not None:
            if self.mode == "density_matrix_simulator" and state.isket:
                self.state = ket2dm(state)
            else:
                self.state = state

        self.probability = 1
        self.op_index = 0
        self.measure_results = measure_results
        self.measure_ind = 0

    def _compute_unitary(self, U_list, inds_list):
        """
        Compute unitary corresponding to a product of unitaries in U_list
        and expand it to size of circuit.

        Parameters
        ----------
        U_list: list of Qobj
            list of predefined unitaries.

        inds_list: list of list of int
            list of qubit indices corresponding to each unitary in U_list

        Returns
        -------
        U: Qobj
            resultant unitary
        """

        U_overall, overall_inds = gate_sequence_product(
            U_list, inds_list=inds_list, expand=True
        )

        if len(overall_inds) != self.qc.N:
            U_overall = expand_operator(
                U_overall, N=self.qc.N, targets=overall_inds
            )
        return U_overall

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
        for _ in range(len(self.ops)):
            if self.step() is None:
                break
        return CircuitResult(self.state, self.probability, self.cbits)

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
            list(filter(lambda x: isinstance(x, Measurement), self.qc.gates))
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

        op = self.ops[self.op_index]
        if isinstance(op, Measurement):
            self._apply_measurement(op)
        elif isinstance(op, tuple):
            operation, U = op
            apply_gate = all(
                [self.cbits[i] for i in operation.classical_controls]
            )
            if apply_gate:
                if self.precompute_unitary:
                    U = expand_operator(
                        U, self.qc.N, operation.get_all_qubits()
                    )
                self._evolve_state(U)
        else:
            self._evolve_state(op)

        self.op_index += 1
        return self.state

    def _evolve_state(self, U):
        """
        Applies unitary to state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        """

        if self.mode == "state_vector_simulator":
            self._evolve_ket(U)
        elif self.mode == "density_matrix_simulator":
            self._evolve_dm(U)
        else:
            raise NotImplementedError(
                "mode {} is not available.".format(self.mode)
            )

    def _evolve_ket(self, U):
        """
        Applies unitary to ket state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        """

        self.state = U * self.state

    def _evolve_dm(self, U):
        """
        Applies unitary to density matrix state.

        Parameters
        ----------
        U: Qobj
            unitary to be applied.
        """

        self.state = U * self.state * U.dag()

    def _apply_measurement(self, operation):
        """
        Applies measurement gate specified by operation to current state.

        Parameters
        ----------
        operation: :class:`.Measurement`
            Measurement gate in a circuit object.
        """

        states, probabilities = operation.measurement_comp_basis(self.state)

        if self.mode == "state_vector_simulator":
            if self.measure_results:
                i = int(self.measure_results[self.measure_ind])
                self.measure_ind += 1
            else:
                probabilities = [p / sum(probabilities) for p in probabilities]
                i = np.random.choice([0, 1], p=probabilities)
            self.probability *= probabilities[i]
            self.state = states[i]
            if operation.classical_store is not None:
                self.cbits[operation.classical_store] = i

        elif self.mode == "density_matrix_simulator":
            states = list(filter(lambda x: x is not None, states))
            probabilities = list(filter(lambda x: x != 0, probabilities))
            self.state = sum(p * s for s, p in zip(states, probabilities))
        else:
            raise NotImplementedError(
                "mode {} is not available.".format(self.mode)
            )


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
        ----------
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
