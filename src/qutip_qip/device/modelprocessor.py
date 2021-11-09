from collections.abc import Iterable
import numbers

import numpy as np

from qutip import Qobj, QobjEvo, tensor, mesolve
from ..operations import globalphase
from ..circuit import QubitCircuit
from .processor import Processor
from ..compiler import GateCompiler
from ..pulse import Pulse


__all__ = ["ModelProcessor"]


class ModelProcessor(Processor):
    """
    The base class for a circuit processor simulating a physical device,
    e.g cavityQED, spinchain.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.
    It cannot be used alone, please refer to the sub-classes.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`.Processor`)

    Parameters
    ----------
    num_qubits: int, optional
        The number of qubits.
        It replaces the old API ``N``.

    dims: list, optional
        The dimension of each component system.
        Default value is a qubit system of ``dim=[2,2,2,...,2]``.

    correct_global_phase: boolean, optional
        If true, the analytical solution will track the global phase. It
        has no effect on the numerical solution.

    **params:
        - t1: float or list, optional
            Characterize the amplitude damping for each qubit.
            A list of size `num_qubits` or a float for all qubits.
        - t2: float or list, optional
            Characterize the total dephasing for each qubit.
            A list of size `num_qubits` or a float for all qubits.
    """

    def __init__(
        self,
        num_qubits=None,
        dims=None,
        correct_global_phase=True,
        model=None,
        **params
    ):
        super(ModelProcessor, self).__init__(
            num_qubits=num_qubits, dims=dims, model=model, **params
        )
        self.correct_global_phase = correct_global_phase
        self.global_phase = 0.0
        self.native_gates = None
        self.transpile_functions = []
        self._default_compiler = None

    def set_up_params(self):
        """
        Save the parameters in the attribute `params` and check the validity.
        (Defined in subclasses)

        Notes
        -----
        All parameters will be multiplied by 2*pi for simplicity
        """
        raise NotImplementedError("Parameters should be defined in subclass.")

    def run_state(
        self, init_state=None, analytical=False, qc=None, states=None, **kwargs
    ):
        """
        If ``analytical`` is False, use :func:`qutip.mesolve` to
        calculate the time of the state evolution
        and return the result.
        Other arguments of :func:`qutip.mesolve` can be
        given as keyword arguments.
        If ``analytical`` is True, calculate the propagator
        with matrix exponentiation and return a list of matrices.

        Parameters
        ----------
        init_state: Qobj
            Initial density matrix or state vector (ket).

        analytical: boolean
            If True, calculate the evolution with matrices exponentiation.

        qc : :class:`.QubitCircuit`, optional
            A quantum circuit. If given, it first calls the ``load_circuit``
            and then calculate the evolution.

        states : :class:`qutip.Qobj`, optional
         Old API, same as init_state.

        **kwargs
           Keyword arguments for the qutip solver.

        Returns
        -------
        evo_result: :class:`qutip.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if qc is not None:
            self.load_circuit(qc)
        return super(ModelProcessor, self).run_state(
            init_state=init_state,
            analytical=analytical,
            states=states,
            **kwargs
        )

    def get_ops_and_u(self):
        """
        Get the labels for each Hamiltonian.

        Returns
        -------
        ctrls: list
            The list of Hamiltonians
        coeffs: array_like
            The transposed pulse matrix
        """
        return (self.ctrls, self.get_full_coeffs().T)

    def pulse_matrix(self, dt=0.01):
        """
        Generates the pulse matrix for the desired physical system.

        Returns
        -------
        t, u, labels:
            Returns the total time and label for every operation.
        """
        ctrls = self.ctrls
        coeffs = self.get_full_coeffs().T

        # FIXME This might becomes a problem if new tlist other than
        # int the default pulses are added.
        tlist = self.get_full_tlist()
        dt_list = tlist[1:] - tlist[:-1]
        t_tot = tlist[-1]
        num_step = int(np.ceil(t_tot / dt))

        t = np.linspace(0, t_tot, num_step)
        u = np.zeros((len(ctrls), num_step))

        t_start = 0
        for n in range(len(dt_list)):
            t_idx_len = int(np.floor(dt_list[n] / dt))
            mm = 0
            for m in range(len(ctrls)):
                u[mm, t_start : (t_start + t_idx_len)] = (
                    np.ones(t_idx_len) * coeffs[n, m]
                )
                mm += 1
            t_start += t_idx_len

        return t, u, self.get_operators_labels()

    def topology_map(self, qc):
        """
        Map the circuit to the hardware topology.
        """
        raise NotImplementedError

    def transpile(self, qc):
        """
        Convert the circuit to one that can be executed on given hardware.
        If there is a method ``topology_map`` defined,
        it will use it to map the circuit to the hardware topology.
        If the processor has a set of native gates defined, it will decompose
        the given circuit to the native gates.

        Parameters
        ----------
        qc: :class:`.QubitCircuit`
            The input quantum circuit.

        Returns
        -------
        qc: :class:`.QubitCircuit`
            The transpiled quantum circuit.
        """
        try:
            qc = self.topology_map(qc)
        except NotImplementedError:
            pass
        if self.native_gates is not None:
            qc = qc.resolve_gates(basis=self.native_gates)
        return qc

    def load_circuit(self, qc, schedule_mode="ASAP", compiler=None):
        """
        The default routine of compilation.
        It first calls the :meth:`.transpile` to convert the circuit to
        a suitable format for the hardware model.
        Then it calls the compiler and save the compiled pulses.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            Takes the quantum circuit to be implemented.

        schedule_mode: string
            "ASAP" or "ALAP" or None.

        compiler: subclass of :class:`.GateCompiler`
            The used compiler.

        Returns
        -------
        tlist, coeffs: dict of 1D NumPy array
            A dictionary of pulse label and the time sequence and
            compiled pulse coefficients.
        """
        qc = self.transpile(qc)
        # Choose a compiler and compile the circuit
        if compiler is None and self._default_compiler is not None:
            compiler = self._default_compiler(self.num_qubits, self.params)
        if compiler is not None:
            tlist, coeffs = compiler.compile(
                qc.gates, schedule_mode=schedule_mode
            )
        else:
            raise ValueError("No compiler defined.")
        # Save compiler pulses
        self.set_coeffs(coeffs)
        self.set_tlist(tlist)
        return tlist, coeffs


def _to_array(params, num_qubits):
    """
    Transfer a parameter to an array.
    """
    if isinstance(params, numbers.Real):
        return np.asarray([params] * num_qubits)
    elif isinstance(params, Iterable):
        return np.asarray(params)
