"""Variational Quantum Algorithms generation and optimization"""

import types
import random
import numpy as np
from qutip import basis, tensor, Qobj, qeye, expect
from qutip_qip.circuit import QubitCircuit
from scipy.optimize import minimize
from scipy.linalg import expm_frechet
from .operations import gate_sequence_product
from .compat import to_scalar


class VQA:
    """
    Optimizes free parameters to generate :obj:`.QubitCircuit` instances
    based on Variational Quantum Algorithms.
    Accepts :obj:`.VQABlock` elements instead of :obj:`~.operations.Gate` elements,
    which allows for easy parameterization of user-defined circuit elements.
    Includes methods for parameter optimization and generators of
    :obj:`.QubitCircuit` instances.

    Parameters
    ----------
    num_qubits: int
        number of qubits used by the algorithm
    num_layers: int, optional
        number of layers used by the algorihtm
    cost_method: str
        method used to compute the cost of an instance of the circuit
        constructed by fixing its free parameters. Can be one of `OBSERVABLE`,
        `BITSTRING` or `STATE`.

        #.  If `OBSERVABLE` is set, then the attribute
            ``VQA.cost_observable`` needs to be specified as a ``Qobj``.
            The cost of the circuit is the expectation value of this observable
            in the final state.
        #.  If `STATE` is set, then ``VQA.cost_func`` needs to be specified
            as a callable that takes in a quantum state, as a ``Qobj``, and
            returns a float.
        #.  If `BITSTRING` is set, then ``VQA.cost_func`` needs to be
            specified as a callable that takes in a bitstring and returns
            a float.
    """

    def __init__(self, num_qubits, num_layers=1, cost_method="OBSERVABLE"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.blocks = []
        self.user_gates = {}
        self._cost_methods = ["OBSERVABLE", "STATE", "BITSTRING"]
        self.cost_method = cost_method
        self.cost_func = None
        self.cost_observable = None

        if self.num_qubits < 1:
            raise ValueError("Expected 1 or more qubits")
        if not isinstance(self.num_qubits, int):
            raise TypeError("Expected an integer number of qubits")
        if self.num_layers < 1:
            raise ValueError("Expected 1 or more layer")
        if not isinstance(self.num_layers, int):
            raise TypeError("Expected an integer number of layers")
        if self.cost_method not in self._cost_methods:
            raise ValueError(
                f"Cost method {self.cost_method} not one of "
                f"{self._cost_methods}"
            )

    def get_block_series(self):
        """
        Ordered list of circuit blocks, including layer repetitions,
        from first applied to last applied.
        """
        blocks = [*self.blocks]
        for _ in range(1, self.num_layers):
            for block in list(filter(lambda b: not b.initial, self.blocks)):
                blocks.append(block)
        return blocks

    def add_block(self, block):
        """
        Append a :obj:`.VQABlock` instance to the circuit, and update the
        user_gates dictionary if necessary.

        Parameters
        ----------
        block: VQABlock
        """
        if not block.name:
            block.name = "U" + str(len(self.blocks))
        if block.name in list(map(lambda b: b.name, self.blocks)):
            raise ValueError("Duplicate Block name in blocks dict")
        self.blocks.append(block)
        self.user_gates[block.name] = lambda angles=None: block.get_unitary(
            angles
        )

    def get_free_parameters_num(self):
        """
        Compute the number of free parameters required
        to evaluate the circuit.

        Returns
        -------
        num_params: int
            Number of free circuit parameters
        """
        initial_blocks = list(filter(lambda b: b.initial, self.blocks))
        layer_blocks = list(filter(lambda b: not b.initial, self.blocks))

        n_initial_params = sum(
            list(map(lambda b: b.get_free_parameters_num(), initial_blocks))
        )

        n_layer_params = (
            sum(list(map(lambda b: b.get_free_parameters_num(), layer_blocks)))
            * self.num_layers
        )

        return n_initial_params + n_layer_params

    def construct_circuit(self, angles):
        """
        Construct a circuit by specifying values for each
        free parameter.

        Parameters
        ----------
        angles: list of float
            A list of dimension (n,) for n free parameters in the circuit

        Returns
        -------
        circ: :obj:`.QubitCircuit`
        """
        circ = QubitCircuit(self.num_qubits)
        circ.user_gates = self.user_gates
        i = 0
        for layer_num in range(self.num_layers):
            for block in self.blocks:
                if block.initial and layer_num > 0:
                    continue
                if block.is_native_gate:
                    circ.add_gate(block.operator, targets=block.targets)
                else:
                    n = block.get_free_parameters_num()
                    circ.add_gate(
                        block.name,
                        targets=list(range(self.num_qubits)),
                        arg_value=angles[i : i + n] if n > 0 else None,
                    )
                    i += n
        return circ

    def get_initial_state(self):
        """
        Returns
        -------
        initial_state: :obj:`qutip.Qobj`
            Initial circuit state
        """
        initial_state = basis(2, 0)
        for _ in range(self.num_qubits - 1):
            initial_state = tensor(initial_state, basis(2, 0))
        return initial_state

    def get_final_state(self, angles):
        """
        Evaluate the circuit by specifying each circuit parameter

        Parameters
        ----------
        angles: list of float
            A list of dimension (n,) for n free parameters in the circuit

        Returns
        -------
        final_state: :obj:`qutip.Qobj`.
            Final state of the circuit after evaluation
        """
        circ = self.construct_circuit(angles)
        initial_state = self.get_initial_state()
        final_state = circ.run(initial_state)
        return final_state

    def _sample_bitstring_from_state(self, state):
        """
        Use probability amplitudes from state after measurement
        in computational basis to sample a bitstring.

        Parameters
        ----------
        state: :obj:`qutip.Qobj`

        Returns
        -------
        bitstring: str
            Formatted binary string with length corresponding to
            dimension of input state

        E.g. the state 1/sqrt(2) * (|0> + |1>)
        would return 0 and 1 with equal probability.
        """
        num_qubits = int(np.log2(state.shape[0]))
        outcome_indices = list(range(2**num_qubits))
        probs = [abs(i.item()) ** 2 for i in state]
        outcome_index = np.random.choice(outcome_indices, p=probs)
        return format(outcome_index, f"0{num_qubits}b")

    def evaluate_parameters(self, angles):
        """
        Evaluate a cost for the circuit, based on the VQA cost
        method defined.

        Parameters
        ----------
        angles: list of float
            A list of dimension (n,) for n free parameters in the circuit

        Returns
        -------
        cost: float
        """
        final_state = self.get_final_state(angles)
        if self.cost_method == "BITSTRING":
            if self.cost_func is None:
                raise ValueError(
                    "To use BITSTRING as the cost method, please"
                    " specify the attribute cost_func"
                )
            else:
                return self.cost_func(
                    self._sample_bitstring_from_state(final_state)
                )
        elif self.cost_method == "STATE":
            if self.cost_func is None:
                raise ValueError(
                    "To use STATE as the cost method, please"
                    " specify the attribute cost_func"
                )
            else:
                return self.cost_func(final_state)
        elif self.cost_method == "OBSERVABLE":
            if self.cost_observable is None:
                raise ValueError(
                    "To use OBSERVABLE as the cost method, please"
                    " specify the attribute cost_observable"
                )
            else:
                return expect(self.cost_observable, final_state)
        else:
            raise ValueError(f"Unrecognised cost method: {self.cost_method}")

    def optimize_parameters(
        self,
        initial="random",
        method="COBYLA",
        use_jac=False,
        layer_by_layer=False,
        bounds=None,
        constraints=(),
    ):
        """
        Run VQA optimization loop

        Parameters
        ----------
        initial: str or list of floats, optional
            Initialization method for the free parameters.
            If a list of floats of dimensions (n,) for n free parameters
            in the circuit is given, then these are taken to be the initial
            conditions for the optimizer. Otherwise if a string is given:

            * (Default) "random" will randomize initial free parameters
              between 0 and 1.

            * "ones" will set each initial free parameter to a value of 1.

        method: str or callable, optional
            Method to give to ``scipy.optimize.minimize``
        use_jac: bool, optional
            Whether to compute the jacobian or not. If computed, it will be
            passed to the optimizer chosen by ``method``, regardless of if
            the method is gradient-based or not.
            Note that derivatives of unitaries generated by
            ``ParameterizedHamiltonian`` are calculated with the
            Frechet derivative of the exponential function,
            using ``scipy.linalg.expm_frechet``.
        layer_by_layer: bool, optional
            Grow the circuit from a single layer to ``VQA.num_layers``. At each
            step, hold the parameters found for previous layers fixed.
        bounds: sequence or `scipy.optimize.Bounds`, optional
            Bounds to be passed to the optimizer. Either

            #. Instance of `scipy.optimize.Bounds`
            #. Sequence of ``(min, max)`` tuples corresponding to each
               free parameter.
        constraints: list of `Constraint`
            See `scipy.optimize.minimize` documentation.

        Returns
        -------
        result: :obj:`.OptimizationResult`
            The optimized angles and final state.
        """

        n_free_params = self.get_free_parameters_num()
        # Set initial circuit parameters
        if isinstance(initial, str):
            if initial == "random":
                angles = [random.random() for i in range(n_free_params)]
            elif initial == "ones":
                angles = [1 for i in range(n_free_params)]
            else:
                raise ValueError("Invalid initial condition string")
        elif isinstance(initial, list) or isinstance(initial, np.ndarray):
            if len(initial) != n_free_params:
                raise ValueError(
                    f"Expected {n_free_params} initial parameters"
                    f"but got {len(initial)}."
                )
            angles = initial
        else:
            raise ValueError(
                "Initial conditions were neither a list of values"
                " nor a string specifying initialization."
            )

        # Run the scipy minimization function
        if layer_by_layer:
            max_layers = self.num_layers
            n_params = 0
            params = []
            for layer_num in range(1, max_layers + 1):
                print(f"Optimizing layer {layer_num}/{max_layers}")
                self.num_layers = layer_num
                n_tot = self.get_free_parameters_num()
                # subset initialization parameters
                init = angles[n_params:n_tot]

                def layer(a, p):
                    return self.evaluate_parameters(np.append(p, a))

                if use_jac:

                    def layer_jac(a, p):
                        return self.compute_jac(
                            np.append(p, a), list(range(n_params, n_tot))
                        )

                else:
                    layer_jac = None
                res = minimize(
                    layer,
                    init,
                    args=(params),
                    method=method,
                    jac=layer_jac,
                    bounds=bounds,
                    constraints=constraints,
                )
                params = np.append(params, res.x)
                n_params += n_tot - n_params
            angles = params
        else:
            res = minimize(
                self.evaluate_parameters,
                angles,
                method=method,
                jac=self.compute_jac if use_jac else None,
                bounds=bounds,
                constraints=constraints,
                options={"disp": False},
            )
            angles = res.x
        final_state = self.get_final_state(angles)
        result = OptimizationResult(res, final_state)
        return result

    def get_unitary_products(self, propagators):
        """
        Return two ordered lists of propagators in the circuit.
        Useful for modifying individual propagators and computing
        the product with these modifications. For example, to modify
        U_k in a product of N unitaries, one could take
        U_prods_back[N - 1 - k] * modified_U_k * U_prods[k]

        Returns
        -------
        U_prods: list of :obj:`qutip.Qobj`
            Ordered list of [identity, U_0, U_1, ... U_N]
        U_prods_back: list of :obj:`qutip.Qobj`
            Ordered list of [identity, U_N, U_{N-1}, ... U_0]
        """
        U_prods = [qeye([2 for _ in range(self.num_qubits)])]
        U_prods_back = [qeye([2 for _ in range(self.num_qubits)])]
        for i, _ in enumerate(propagators):
            U_prods.append(propagators[i] * U_prods[-1])
            U_prods_back.append(U_prods_back[-1] * propagators[-i - 1])
        return U_prods, U_prods_back

    def cost_derivative(self, U, dU):
        """
        Returns partial derivative of cost function (in observable
        mode) with respect to the parameter in the block's unitary.
        Assuming a block unitary of the form e^{-iH * theta}, this
        will return d/(d theta) of the cost function in terms of
        an observable.

        Parameters
        ----------
        U: :obj:`qutip.Qobj`
            Block unitary
        dU: :obj:`qutip.Qobj`
            Partial derivative of U with respect to its parameter

        Returns
        -------
        dCost: float
            Partial derivative of cost with respect to block's parameter
        """
        if self.cost_observable is None:
            raise NotImplementedError(
                "cost_derivative function only "
                "implemented for observable cost "
                "functions"
            )
        init = self.get_initial_state()
        obs = self.cost_observable
        dCost = (init.dag() * dU.dag()) * obs * (U * init) + (
            init.dag() * U.dag()
        ) * obs * (dU * init)
        dCost = to_scalar(dCost)
        return np.real(dCost)

    def compute_jac(self, angles, indices_to_compute=None):
        """
        Compute the jacobian for the circuit's cost function,
        assuming the cost function is in observable mode.

        Parameters
        ----------
        angles: list of float
            Circuit free parameters
        indicies_to_compute: list of int, optional
            Block indices for which to use in computing the jacobian.
            By default, this is every index (every block).

        Returns
        -------
        jac: (n,) numpy array of floats
        """
        if indices_to_compute is None:
            indices_to_compute = list(range(len(angles)))

        circ = self.construct_circuit(angles)
        propagators = circ.propagators()
        U = gate_sequence_product(propagators)
        U_prods, U_prods_back = self.get_unitary_products(propagators)
        # subtract one for the identity matrix
        n = len(U_prods) - 1

        def modify_unitary(k, U):
            return U_prods_back[n - 1 - k] * U * U_prods[k]

        jacobian = []
        i = 0
        for k, block in enumerate(self.get_block_series()):
            n_params = block.get_free_parameters_num()
            if n_params > 0:
                if i in indices_to_compute:
                    dBlock = block.get_unitary_derivative(
                        angles[i : i + n_params]
                    )
                    dU = modify_unitary(k, dBlock)
                    jacobian.append(self.cost_derivative(U, dU))
                i += n_params
        return np.array(jacobian)

    def export_image(self, filename="circuit.png"):
        """
        Export an image of the circuit.

        Parameters
        ----------
        filename: str, optional
            The name of the exported file
        """
        circ = self.construct_circuit(
            [1 for _ in range(self.get_free_parameters_num())]
        )
        f = open(filename, "wb+")
        f.write(circ.png.data)
        f.close()
        print(f"Image saved to ./{filename}")


class ParameterizedHamiltonian:
    """
    Hamiltonian with 0 or more parameterized terms.
    In general, computes a unitary as
    :math:`U = e^{H_0 + p_1 H_1 + P_2 H_2 + \\dots}`

    Parameters
    ----------
    parameterized_terms: list of :obj:`qutip.Qobj`
        Hamiltonian terms which each require a unique parameter
    constant_term: :obj:`qutip.Qobj`
        Hamiltonian term which does not require parameters.
    """

    def __init__(self, parameterized_terms=[], constant_term=None):
        self.p_terms = parameterized_terms
        self.c_term = constant_term
        self.num_parameters = len(parameterized_terms)
        if len(self.p_terms) == 0 and self.c_term is None:
            raise ValueError(
                "Parameterized Hamiltonian " "initialised with no terms given"
            )

    def get_hamiltonian(self, params):
        if not len(params) == self.num_parameters:
            raise ValueError(
                f"params should be of length {self.num_parameters}"
                f"but was {len(params)}"
            )

        # Match each p_term with a parameter
        H_tot = sum(param * H for param, H in zip(self.p_terms, params)) + (
            self.c_term if self.c_term else 0
        )
        return H_tot


class VQABlock:
    """
    Component of a :obj:`.VQA`. Can return a unitary, and take
    derivatives of its own unitary. Forms a :obj:`~.operations.Gate` in the
    :obj:`.QubitCircuit` generated by the :obj:`.VQA`.

    Parameters
    ----------
    operator: :obj:`qutip.Qobj` or Callable or str
        If given as a :obj:`qutip.Qobj`, assumed to be a Hamiltonian with
        a single global parameter.
        If given as a Callable, assumed to take in a parameter, and return
        a unitary operator.
        If given as a str, assumed to reference a native QuTiP gate from
        ``qutip_qip.operations``
    is_unitary: bool, optional
        Specifies that the operator  was already in Unitary form,
        and does not need to be exponentiated, or take a parameter.
    name: str, optional
        Name of the block. This will be used in the custom
        ``user_gates`` dict of the circuit. If not provided,
        a name will be generated as "U"+str(len(VQA.blocks)).
    targets: list of int, optional
        The qubits targetted by the gate. By default, applied
        to all qubits.
    initial: bool, optional
        Whether or not to repeat this block in layers. For example,
        this should be True if this block is only used for
        circuit initialization.
    """

    def __init__(
        self,
        operator,
        is_unitary=False,
        name=None,
        targets=None,
        initial=False,
    ):
        self.operator = operator
        self.is_unitary = is_unitary
        self.name = name
        self.targets = targets
        self.initial = initial
        self.is_native_gate = False
        self.num_parameters = 0

        if isinstance(operator, Qobj):
            if not self.is_unitary:
                self.num_parameters = 1
        elif isinstance(operator, str):
            self.is_native_gate = True
            if targets is None:
                raise ValueError("Targets must be specified for native gates")
        elif isinstance(operator, ParameterizedHamiltonian):
            self.num_parameters = operator.num_parameters
        elif isinstance(operator, types.FunctionType):
            self.num_parameters = 1
        else:
            raise ValueError(
                "operator should be either: Qobj | function which"
                " returns Qobj | ParameterizedHamiltonian"
                " instance | string referring to gate."
            )

    def get_free_parameters_num(self):
        return self.num_parameters

    def get_unitary(self, angles=None):
        """
        Return the block unitary.

        Parameters
        ----------
        angles: list of float, optional
            Block free parameters. Required if the block has free parameters.
        """
        # Qobj unitary or zero-parmeters function returning Qobj unitary
        if angles is None:
            if self.is_unitary:
                return self.operator
            raise ValueError("No angles were given and block was not unitary")

        if len(angles) != self.get_free_parameters_num():
            raise ValueError(
                f"Expected {self.get_free_parameters_num()} angles"
                f" but got {len(angles)}."
            )

        # Case where the operator is a string referring to an existing gate.
        if self.is_native_gate:
            raise TypeError("Can't compute unitary of native gate")
        # Function returning Qobj unitary
        if isinstance(self.operator, types.FunctionType):
            # In the future, this could be generalized to multiple angles
            unitary = self.operator(angles[0])
            if not isinstance(unitary, Qobj):
                raise TypeError("Provided function does not return Qobj")
            return unitary
        # ParameterizedHamiltonian instance
        if isinstance(self.operator, ParameterizedHamiltonian):
            return (-1j * self.operator.get_hamiltonian(angles)).expm()

        # If there's no other specification, treat operator as Hamiltonian
        if len(angles) != 1:
            raise ValueError(
                "Expected one angle for singly-parameterized Hamiltonian."
            )

        return (-1j * angles[0] * self.operator).expm()

    def get_unitary_derivative(self, angles, term_index=0):
        """
        Compute the derivative of the block's unitary with respect to its
        free parameter, assuming it is of the form :math:`e^{-i \\theta H}`
        for a free parameter theta. If the block's operator is a
        :obj:`ParameterizedHamiltonian`, use the Frechet derivative of the
        exponential function.

        Parameters
        ----------
        angle: list of float
            free parameters to take derivatives with respect to
        term_index: int, optional
            Index of Parameterized Hamiltonian term that specifies the matrix
            direction in which to take the derivative.

        Returns
        -------
        derivative: float
        """
        if self.is_unitary or self.is_native_gate:
            raise ValueError(
                "Can only take derivative of block specified "
                "by Hamiltonians or ParameterizedHamiltonian instances."
            )
        if isinstance(self.operator, ParameterizedHamiltonian):
            arg = -1j * self.operator.get_hamiltonian(angles)
            direction = -1j * self.operator.p_terms[term_index]
            return Qobj(
                expm_frechet(arg.full(), direction.full(), compute_expm=False),
                dims=direction.dims,
            )
        if len(angles) != 1:
            raise ValueError(
                "Expected a single angle for non-"
                "ParameterizedHamiltonian instance."
            )
        return self.get_unitary(angles) * -1j * self.operator


class OptimizationResult:
    """
    Class for results of :obj:`.VQA` optimization loop.

    Parameters
    ----------
    res: scipy results instance
    final_state: :obj:`qutip.Qobj`
        Final state of the circuit after optimization.
    """

    def __init__(self, res, final_state):
        self.res = res
        self.angles = res.x
        self.min_cost = res.fun
        self.nfev = res.nfev
        self.final_state = final_state

    def _highest_prob_bitstring(self, state):
        """
        Return the bitstring associated with the
        highest probability amplitude measurement state.
        """
        num_qubits = int(np.log2(state.shape[0]))
        index = np.argmax(abs(state.full()))
        return format(index, f"0{num_qubits}b")

    def get_top_bitstring(self):
        """
        Return the bitstring associated with the highest probability
        measurement outcome

        Returns
        -------
        bitstring: str
            bitstring in the form :math:`|x_0x_1...x_n>` where each
            :math:`x_i` is 0 or 1 and n is the number of qubits of the system.
        """
        return "|" + self._highest_prob_bitstring(self.final_state) + ">"

    def __str__(self):
        return (
            "Optimization Result:\n"
            + f"\tMinimum cost: {self.min_cost}\n"
            + f"\tNumber of function evaluations: {self.nfev}\n"
            + f"\tParameters found: {self.angles}"
        )

    def _label_to_sets(self, S, bitstring):
        """
        Convert bitstring to string representation of
        two sets containing elements of the problem instance.

        Parameters
        ----------
        S: list of float
            Problem instance
        bitstring: str

        Returns
        -------
        sets: str
            String representation the two sets.
        """
        s1 = []
        s2 = []
        for i, c in enumerate(bitstring.strip("|").strip(">")):
            if c == "0":
                s1.append(S[i])
            else:
                s2.append(S[i])
        return (str(s1) + " " + str(s2)).replace("[", "{").replace("]", "}")

    def plot(self, S=None, label_sets=False, top_ten=False, display=True):
        """
        Plot probability amplitudes of each measurement
        outcome of a state.

        Parameters
        ----------
        S: list of float, optional
            Problem instance
        min_cost: str, optional
            The minimum cost found by optimization
        label_sets: bool, optional
            Replace bitstring labels with sets referring to the inferred
            output of the combinatorial optimization problem. For example
            a bitstring :math:`|010>` would produce a set with the first and
            last elements of S, and one with the second element of S.
        top_ten: bool, optional
            Only plot the ten highest-probability states.
        display: bool, optional
            Display the plot with the pyplot plot.show() method
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("could not import matplotlib.pyplot")
            quit()
        state = self.final_state
        min_cost = self.min_cost

        num_qubits = int(np.log2(state.shape[0]))
        probs = [abs(i.item()) ** 2 for i in state]
        bitstrings = [
            "|" + format(i, f"0{num_qubits}b") + ">"
            for i in range(2**num_qubits)
        ]
        if top_ten and len(probs) > 10:
            threshold = sorted(probs)[-11]
            top_probs = []
            top_bitstrings = []
            for i, prob in enumerate(probs):
                if prob > threshold:
                    top_probs.append(prob)
                    top_bitstrings.append(bitstrings[i])
            bitstrings = top_bitstrings
            probs = top_probs
        if label_sets:
            labels = [
                self._label_to_sets(S, bitstring) for bitstring in bitstrings
            ]
        fig, ax = plt.subplots()
        ax.bar(
            list(range(len(bitstrings))),
            probs,
            tick_label=labels if label_sets else bitstrings,
            width=0.8,
        )
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_xlabel("Measurement outcome")
        ax.set_ylabel("Probability")
        ax.set_title(
            "Measurement Outcomes after Optimisation. "
            f"Cost: {round(min_cost, 2)}"
        )
        fig.tight_layout()
        if display:
            fig.show()
