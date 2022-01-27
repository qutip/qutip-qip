import numpy as np
from qutip import basis, tensor, Qobj, qeye
from qutip.qip.circuit import QubitCircuit, Gate, Measurement
from qutip_qip.operations import *
from scipy.optimize import minimize
from scipy.linalg import expm_frechet
import matplotlib.pyplot as plt
import random

def sample_bitstring_from_state(state):
    """
    Uses probability amplitudes from state in computational
    basis to sample a bitstring.
    E.g. the state 1/sqrt(2) * (|0> + |1>)
    would return 0 and 1 with equal probability.
    """
    n_qubits = int(np.log2(state.shape[0]))
    outcome_indices = [i for i in range(2**n_qubits)]
    probs = [abs(i.item())**2 for i in state]
    outcome_index = np.random.choice(outcome_indices, p=probs)
    return format(outcome_index, f'0{n_qubits}b')
def highest_prob_bitstring(state):
    """
    Returns the bitstring associated with the
    highest probability amplitude state (computational basis).
    """
    n_qubits = int(np.log2(state.shape[0]))
    index = np.argmax(abs(state))
    return format(index, f'0{n_qubits}b')

class VQA:
    def __init__(self, n_qubits, n_layers=1, cost_method="BITSTRING"):
        # defaults for now
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.blocks = []
        self.user_gates = {}
        self._cost_methods = ["OBSERVABLE", "STATE", "BITSTRING"]
        self.cost_method = cost_method
        self.cost_func = None
        self.cost_observable = None
    def get_block_series(self):
        """
        Ordered list of blocks, including layer repetitions,
        from first applied to last.
        """
        blocks = [*self.blocks]
        for layer_num in range(1, self.n_layers):
            for block in list(filter(lambda b: not b.initial, self.blocks)):
                blocks.append(block)
        return blocks
    def add_block(self, block):
        if not block.name:
            block.name = "U" + str(len(self.blocks))
        if block.name in list(map(lambda b: b.name, self.blocks)):
            raise ValueError("Duplicate Block name in self.blocks")
        self.blocks.append(block)
        self.user_gates[block.name] = lambda angle=None: block.get_unitary(angle)
    def get_free_parameters(self):
        """
        Computes the number of free parameters required
        to evaluate the circuit.

        Returns
        -------
        num_params : int
            number of free parameters
        """
        initial_blocks = list(filter(
            lambda b: b.initial, self.blocks
            ))
        layer_blocks = list(filter(
            lambda b: not b.initial, self.blocks
            ))

        n_initial_params = sum(list(map(
            lambda b: b.get_free_parameters(), initial_blocks
            )))

        n_layer_params = sum(list(map(
            lambda b: b.get_free_parameters(), layer_blocks
            ))) * self.n_layers

        return n_initial_params + n_layer_params 
 
    def construct_circuit(self, angles):
        circ = QubitCircuit(self.n_qubits)
        circ.user_gates = self.user_gates
        i = 0
        for layer_num in range(self.n_layers):
            for block in self.blocks:
                if block.initial and layer_num > 0:
                    continue
                if block.is_native_gate:
                    circ.add_gate(block.operator, targets=block.targets)
                elif block.is_unitary:
                    circ.add_gate(block.name, targets=[k for k in range(self.n_qubits)])
                else:
                    circ.add_gate(block.name, arg_value=angles[i], targets=[k for k in range(self.n_qubits)])
                    i += 1
        return circ
    def get_initial_state(self):
        """
        Returns the initial circuit state
        """
        initial_state = basis(2, 0)
        for i in range(self.n_qubits - 1):
            initial_state = tensor(initial_state, basis(2, 0))
        return initial_state
    def get_final_state(self, angles):
        """
        Returns final state of circuit from initial state
        """
        circ = self.construct_circuit(angles)
        initial_state = self.get_initial_state()
        final_state = circ.run(initial_state)
        return final_state
    def evaluate_parameters(self, angles):
        """
        Constructs a circuit with given parameters
        and returns a cost from evaluating the circuit
        """
        final_state = self.get_final_state(angles)
        if self.cost_method == "BITSTRING":
            if self.cost_func == None:
                raise ValueError("self.cost_func not specified")
            return self.cost_func(highest_prob_bitstring(final_state))
        elif self.cost_method == "STATE":
            raise Exception("NOT IMPLEMENTED")
        elif self.cost_method == "OBSERVABLE":
            """
            Cost as expectation of observable in in state final_state
            """
            if self.cost_observable == None:
                raise ValueError("self.cost_observable not specified")
            #print(self.cost_observable)
            cost = final_state.dag() * self.cost_observable * final_state
            return abs(cost[0].item())
    def optimize_parameters(
            self, initial="random", method="COBYLA", use_jac=False,
            frechet=False, initialization="random", do_nothing=False,
            layer_by_layer=False):

        self.frechet = frechet
        """
        Set initial circuit parameters
        """
        n_free_params = self.get_free_parameters()
        if isinstance(initial, str):
            if initial == "random":
                angles = [random.random() for i in range(n_free_params)]
            elif initial == "ones":
                angles = [1 for i in range(n_free_params)]
            else:
                raise ValueError("Invalid initial condition string")
        elif isinstance(initial, list) \
                or isinstance(initial, np.ndarray):
            if len(initial) != n_free_params:
                raise ValueError(f"Expected {n_free_params} initial" \
                        +f" parameters, but got {len(initial)}.")
            angles = initial
        else:
            raise ValueError("Initial conditions were neither a list of"
                    " values, nor a string specifying initialization.")

        if do_nothing:
            return self.evaluate_parameters(angles)

        jac = self.compute_jac if use_jac else None

        """
        Run scipy minimization method.
        Train parameters either layer-by-layer,
        or all at once
        """
        if layer_by_layer:
            max_layers = self.n_layers
            n_params = 0
            params = []
            for l in range(1, max_layers+1):
                self.n_layers = l
                n_tot = self.get_free_parameters()
                # subset initialization parameters
                init = angles[n_params:n_tot]
                layer = lambda a, p: self.evaluate_parameters(np.append(p, a))
                if use_jac:
                    layer_jac = lambda a, p: self.compute_jac(
                            np.append(p, a), list(range(n_params, n_tot))
                            )
                else:
                    layer_jac = None
                res = minimize(
                        layer, init, args=(params), method=method, jac=layer_jac
                        )
                params = np.append(params, res.x)
                n_params += n_tot - n_params
            angles = params
        else:
            res = minimize(
                    self.evaluate_parameters, 
                    angles,
                    method=method,
                    jac=jac,
                    options={'disp': False}
                    )
            angles = res.x
        final_state = self.get_final_state(angles)
        result = Optimization_Result(res, final_state)
        return result

    def get_unitary_products(self, propagators, angles):
        """ To modify a unitary at the k'th position, 
        i.e U_k(angle), given N unitaries in the product,
        one could do  (where k = 0, ..., N-1)
        U_prods_back[N - 1 - k] * U_k(angle) * U_prods[k]
        """
        from qutip.qip.operations.gates import gate_sequence_product
        U_prods = [qeye([2 for _ in range(self.n_qubits)])]
        U_prods_back = [qeye([2 for _ in range(self.n_qubits)])]
        for i in range(0, len(propagators)):
            U_prods.append(propagators[i] * U_prods[-1])
            U_prods_back.append(U_prods_back[-1] * propagators[-i-1])
        prod = gate_sequence_product(propagators)
        return U_prods, U_prods_back

    def block_cost(self, U, dU):
        if self.cost_observable == None:
            raise ValueError("self.cost_observable not defined")
        init = self.get_initial_state()
        O = self.cost_observable
        dCost = (init.dag() * dU.dag()) * O * (U * init) \
                + (init.dag() * U.dag()) * O * (dU * init)
        return dCost[0].item().real

    def compute_jac(self, angles, indices_to_compute=None):
        if indices_to_compute is None:
            indices_to_compute = [i for i in range(len(angles))]
        from qutip.qip.operations.gates import gate_sequence_product
        circ = self.construct_circuit(angles)
        propagators = circ.propagators()
        U = gate_sequence_product(propagators)
        U_prods, U_prods_back = self.get_unitary_products(propagators, angles)
        # subtract one for the identity matrix
        N = len(U_prods) - 1
        def modify_unitary(k, U):
            return U_prods_back[N - 1 - k] * U * U_prods[k]
        jacobian = []
        i = 0
        for k, block in enumerate(self.get_block_series()):
            if block.n_parameters > 0:
                if i in indices_to_compute:
                    if self.frechet:
                        dBlock = block.get_unitary_frechet_derivative(angles[i])
                    else:
                        dBlock = block.get_unitary_derivative(angles[i])
                    dU = modify_unitary(k, dBlock)
                    jacobian.append(self.block_cost(U, dU))
                i += 1
        return np.array(jacobian)
        
    def export_image(self, filename="circuit.png"):
        circ = self.construct_circuit([1])
        f = open(filename, 'wb+')
        f.write(circ.png)
        f.close()
        print(f"Image saved to ./{filename}")


class Parameterized_Hamiltonian:
    def __init__(self, parameterized_terms=[], constant_term=None):
        """
        Parameters
        ----------
        parameterized_terms: list of Qobj
            Hamiltonian terms which each require a unique parameter
        constant_term: Qobj
            Hamiltonian term which does not require parameters.
        """
        self.p_terms = parameterized_terms
        self.c_term = constant_term
        self.N = len(parameterized_terms)
        if not len(self.p_terms) and not len(self.c_terms):
            raise ValueError("Parameterized Hamiltonian " \
                    + "initialised with no terms given")
    def get_angles(self, params):
        angles = [i for i in params]
        return angles
    def get_hamiltonian(self, params):
        if not len(params) == self.N:
            raise ValueError(f"params should be of length {self.N} but was {len(params)}")

        # Match each p_term with a parameter
        H_tot = sum(param * H for param, H in zip(self.p_terms, params)) \
           + (self.c_term if self.c_term else 0)
        return H_tot

class VQA_Block:
    """
    A "Block" is a constitutent part of a "layer".
    containing a single Hamiltonian or Unitary
    specified by the user. In the case that a Unitary
    is given, there is no associated circuit parameter
    for the block.
    If the operator is given as a string, it assumed
    to reference a default qutip_qip.operations gate.
    A "layer" is given by the product of all blocks.
    """
    def __init__(self, operator, is_unitary=False, name=None, targets=None, initial=False):
        self.operator = operator
        self.is_unitary = is_unitary
        self.name = name
        self.targets = targets
        self.is_native_gate = isinstance(operator, str)
        self.initial = initial
        self.n_parameters = 0
        self.fixed_parameters = []
        if not self.is_unitary and not self.is_native_gate:
            self.n_parameters = 1
        if self.is_native_gate:
            if targets == None:
                raise ValueError("Targets must be specified for native gates")
        else:
            if not isinstance(operator, Qobj):
                raise ValueError("Operator given was neither a gate name nor Qobj")
    def fix_parameters(angles):
        if len(angles) != self.n_parameters:
            raise ValueError(f"Expected {self.n_parameters} fixed parameters"
                    " but received {len(angles)}")
        self.fixed_parameters = angles
    def get_free_parameters(self):
        return self.n_parameters - len(self.fixed_parameters)
    def get_unitary(self, angle=None):
        if self.is_unitary:
            return self.operator
        else:
            if self.is_native_gate:
                raise TypeError("Can't compute unitary of native gate")
            if angle == None:
                # TODO: raise better exception?
                raise TypeError("No parameter given")
            return (-1j * angle * self.operator).expm()
    def get_unitary_derivative(self, angle):
        if self.is_unitary or self.is_native_gate:
            raise ValueError("Can only take derivative of block "
                    "specified by Hamiltonians")
        return self.get_unitary(angle) * -1j * self.operator

    def get_unitary_frechet_derivative(self, angle):
        if self.is_unitary or self.is_native_gate:
            raise ValueError("Can only take frechet derivative of block "
                    "specified by Hamiltonians")
        # TODO: impement for fully parameterised Hamiltonian
        arg = self.operator * angle * -1j
        direction = self.operator * -1j
        return Qobj(expm_frechet(arg, direction, compute_expm=False), dims=self.operator.dims)


class Optimization_Result:
    def __init__(self, res, final_state):
        """
        res : scipy optimisation result object
        """
        self.res = res
        self.angles = res.x
        self.min_cost = res.fun
        self.nfev = res.nfev
        self.final_state = final_state
    def get_top_bitstring(self):
        return "|" + highest_prob_bitstring(self.final_state) + ">"
    def __str__(self):
        return "Optimization Result:\n" +             \
                f"\tMinimum cost: {self.min_cost}\n" +  \
                f"\tNumber of function evaluations: {self.nfev}\n" + \
                f"\tParameters found: {self.angles}"
    def plot(self, S):
        state_probs_plot(self.final_state, S, self.min_cost)

def label_to_sets(S, bitstring):
    S1 = []
    S2 = []
    for i, c in enumerate(bitstring.strip('|').strip('>')):
        if c == '0':
            S1.append(S[i])
        else:
            S2.append(S[i])
    return (str(S1) + ' ' + str(S2)).replace('[', '{').replace(']', '}')

def state_probs_plot(state, S=None, min_cost=''):
    import itertools
    n_qubits = int(np.log2(state.shape[0]))
    probs = [abs(i.item())**2 for i in state]
    bitstrings = ["|" + format(i, f'0{n_qubits}b') + ">"   \
            for i in range(2**n_qubits)]
    labels = [label_to_sets(S, bitstring) for bitstring in bitstrings]
    barplot = plt.bar([i for i in range(2**n_qubits)], \
            probs, tick_label=labels, width=0.8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.xlabel('Measurement outcome')
    plt.ylabel('Probability')
    plt.title(f"Measurement Outcomes after Optimisation. Cost: {round(min_cost, 2)}")
    plt.tight_layout()
    plt.show()
