from copy import deepcopy

import numpy as np

from qutip import sigmax, sigmay, sigmaz, tensor
from ..circuit import QubitCircuit
from .processor import Model
from .modelprocessor import ModelProcessor, _to_array
from ..pulse import Pulse
from ..compiler import SpinChainCompiler
from ..transpiler import to_chain_structure


__all__ = ["SpinChain", "LinearSpinChain", "CircularSpinChain"]


class SpinChain(ModelProcessor):
    """
    The processor based on the physical implementation of
    a spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.
    This is the base class and should not be used directly.
    Please use :class:`.LinearSpinChain` or :class:`.CircularSpinChain`.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    **params:
        Hardware parameters. See :obj:`.SpinChainModel`.
    """

    def __init__(
        self, num_qubits, correct_global_phase=True, model=None, **params
    ):
        super(SpinChain, self).__init__(
            num_qubits=num_qubits,
            correct_global_phase=correct_global_phase,
            model=model,
            **params,
        )
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        self.native_gates = ["SQRTISWAP", "ISWAP", "RX", "RZ"]

    @property
    def sx_ops(self):
        """list: A list of sigmax Hamiltonians for each qubit."""
        return self.ctrls[: self.num_qubits]

    @property
    def sz_ops(self):
        """list: A list of sigmaz Hamiltonians for each qubit."""
        return self.ctrls[self.num_qubits : 2 * self.num_qubits]

    @property
    def sxsy_ops(self):
        """
        list: A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.
        """
        return self.ctrls[2 * self.num_qubits :]

    @property
    def sx_u(self):
        """array-like: Pulse coefficients for sigmax Hamiltonians."""
        return self.coeffs[: self.num_qubits]

    @property
    def sz_u(self):
        """array-like: Pulse coefficients for sigmaz Hamiltonians."""
        return self.coeffs[self.num_qubits : 2 * self.num_qubits]

    @property
    def sxsy_u(self):
        """
        array-like: Pulse coefficients for tensor(sigmax, sigmay)
        interacting Hamiltonians.
        """
        return self.coeffs[2 * self.num_qubits :]

    def load_circuit(self, qc, setup, schedule_mode="ASAP", compiler=None):
        if compiler is None:
            compiler = SpinChainCompiler(
                self.num_qubits, self.params, setup=setup
            )
        tlist, coeffs = super().load_circuit(
            qc, schedule_mode=schedule_mode, compiler=compiler
        )
        self.global_phase = compiler.global_phase
        return tlist, coeffs


class LinearSpinChain(SpinChain):
    """
    Spin chain model with open-end topology.
    For the control Hamiltonian please refer to :obj:`SpinChainModel`.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    **params:
        Hardware parameters. See :obj:`.SpinChainModel`.

    Examples
    --------

    .. testcode::

        import numpy as np
        import qutip
        from qutip_qip.circuit import QubitCircuit
        from qutip_qip.device import LinearSpinChain

        qc = QubitCircuit(2)
        qc.add_gate("RX", 0, arg_value=np.pi)
        qc.add_gate("RY", 1, arg_value=np.pi)
        qc.add_gate("ISWAP", [1, 0])

        processor = LinearSpinChain(2, g=0.1, t1=300)
        processor.load_circuit(qc)
        init_state = qutip.basis([2, 2], [0, 0])
        result = processor.run_state(init_state)
        print(round(qutip.fidelity(result.states[-1], qc.run(init_state)), 4))

    .. testoutput::

        0.994

    """

    def __init__(
        self,
        num_qubits=None,
        correct_global_phase=True,
        **params,
    ):
        model = SpinChainModel(num_qubits=num_qubits, setup="linear", **params)
        super(LinearSpinChain, self).__init__(
            num_qubits,
            correct_global_phase=correct_global_phase,
            model=model,
        )

    @property
    def sxsy_ops(self):
        """
        list: A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.
        """
        return self.ctrls[2 * self.num_qubits : 3 * self.num_qubits - 1]

    @property
    def sxsy_u(self):
        """
        array-like: Pulse coefficients for tensor(sigmax, sigmay)
        interacting Hamiltonians.
        """
        return self.coeffs[2 * self.num_qubits : 3 * self.num_qubits - 1]

    def load_circuit(self, qc, schedule_mode="ASAP", compiler=None):
        return super(LinearSpinChain, self).load_circuit(
            qc, "linear", schedule_mode=schedule_mode, compiler=compiler
        )

    def topology_map(self, qc):
        return to_chain_structure(qc, "linear")


class CircularSpinChain(SpinChain):
    """
    Spin chain model with circular topology. See :class:`.SpinChain`
    for details.
    For the control Hamiltonian please refer to :obj:`SpinChainModel`.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the system.

    correct_global_phase : float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    **params:
        Hardware parameters. See :obj:`.SpinChainModel`.

    Examples
    --------

    .. testcode::

        import numpy as np
        import qutip
        from qutip_qip.circuit import QubitCircuit
        from qutip_qip.device import CircularSpinChain

        qc = QubitCircuit(2)
        qc.add_gate("RX", 0, arg_value=np.pi)
        qc.add_gate("RY", 1, arg_value=np.pi)
        qc.add_gate("ISWAP", [1, 0])

        processor = CircularSpinChain(2, g=0.1, t1=300)
        processor.load_circuit(qc)
        init_state = qutip.basis([2, 2], [0, 0])
        result = processor.run_state(init_state)
        print(round(qutip.fidelity(result.states[-1], qc.run(init_state)), 4))

    .. testoutput::

        0.994
    """

    def __init__(
        self,
        num_qubits=None,
        correct_global_phase=True,
        **params,
    ):
        if num_qubits <= 1:
            raise ValueError(
                "Circuit spin chain must have at least 2 qubits. "
                "The number of qubits is increased to 2."
            )
        model = SpinChainModel(
            num_qubits=num_qubits, setup="circular", **params
        )
        super(CircularSpinChain, self).__init__(
            num_qubits,
            correct_global_phase=correct_global_phase,
            model=model,
        )

    @property
    def sxsy_ops(self):
        """
        list: A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.
        """
        return self.ctrls[2 * self.num_qubits : 3 * self.num_qubits]

    @property
    def sxsy_u(self):
        """
        array-like: Pulse coefficients for tensor(sigmax, sigmay)
        interacting Hamiltonians.
        """
        return self.coeffs[2 * self.num_qubits : 3 * self.num_qubits]

    def load_circuit(self, qc, schedule_mode="ASAP", compiler=None):
        return super(CircularSpinChain, self).load_circuit(
            qc, "circular", schedule_mode=schedule_mode, compiler=compiler
        )

    def topology_map(self, qc):
        return to_chain_structure(qc, "circular")


class SpinChainModel(Model):
    """
    The physical model for the spin chian processor
    (:obj:`CircularSpinChain` and :obj:`LinearSpinChain`).
    The interaction is only possible between adjacent qubits.
    The single-qubit control Hamiltonians are :math:`\\sigma_j^x`$`,
    :math:`\\sigma_j^z`, while the interaction is realized by
    the exchange Hamiltonian
    :math:`\\sigma^x_{j}\\sigma^x_{j+1}+\\sigma^y_{j}\\sigma^y_{j+1}`.
    The overall Hamiltonian model is written as:

    .. math::

        H=
        \\sum_{j=0}^{N-1}
        \\Omega^x_{j}(t) \\sigma^x_{j} +
        \\Omega^z_{j}(t) \\sigma^z_{j} + \\sum_{j=0}^{N-2}
        g_{j}(t)
        (\\sigma^x_{j}\\sigma^x_{j+1}+
        \\sigma^y_{j}\\sigma^y_{j+1}).

    Parameters
    ----------
    num_qubits: int
        The number of qubits, :math:`N`.
    setup : str
        "linear" for an open end and "circular" for a closed end chain.
    **params :
        Keyword arguments for hardware parameters, in the unit of frequency
        (MHz, GHz etc, the unit of time list needs to be adjusted accordingly).
        Parameters can either be a float or list with parameters
        for each qubits.

        - sx : float or list, optional
            The pulse strength of sigma-x control, :math:`\\Omega^x`,
            default ``0.25``.
        - sz : float or list, optional
            The pulse strength of sigma-z control, :math:`\\Omega^z`,
            default ``1.0``.
        - sxsy : float or list, optional
            The pulse strength for the exchange interaction, :math:`g`,
            default ``0.1``.
            It should be either a float or an array of the length
            :math:`N-1` for the linear setup or :math:`N` for
            the circular setup.
        - t1 : float or list, optional
            Characterize the amplitude damping for each qubit.
        - t2 : list of list, optional
            Characterize the total dephasing for each qubit.
    """

    def __init__(self, num_qubits, setup, **params):
        self.num_qubits = num_qubits
        self.dims = num_qubits * [2]
        self.setup = setup
        self.params = {  # default parameters, in the unit of frequency
            "sx": 0.25,
            "sz": 1.0,
            "sxsy": 0.1,
        }
        self._drift = []
        self.params.update(deepcopy(params))
        self.params.update(self._compute_params())
        self._controls = self._set_up_controls(self.num_qubits)
        self._noise = []

    @property
    def _old_index_label_map(self):
        num_qubits = self.num_qubits
        return (
            ["sx" + str(i) for i in range(num_qubits)]
            + ["sz" + str(i) for i in range(num_qubits)]
            + ["g" + str(i) for i in range(num_qubits)]
        )

    def _get_num_coupling(self):
        if self.setup == "linear":
            num_coupling = self.num_qubits - 1
        elif self.setup == "circular":
            num_coupling = self.num_qubits
        else:
            raise ValueError(
                "Parameter setup needs to be linear or circular, "
                f"not {self.params['setup']}"
            )
        return num_coupling

    def _set_up_controls(self, num_qubits):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        num_qubits: int
            The number of qubits in the system.
        """
        controls = {}
        # sx_controls
        for m in range(num_qubits):
            controls["sx" + str(m)] = (2 * np.pi * sigmax(), m)
        # sz_controls
        for m in range(num_qubits):
            controls["sz" + str(m)] = (2 * np.pi * sigmaz(), m)
        # sxsy_controls
        num_coupling = self._get_num_coupling()
        if num_coupling == 0:
            return controls
        operator = tensor([sigmax(), sigmax()]) + tensor([sigmay(), sigmay()])
        for n in range(num_coupling):
            controls["g" + str(n)] = (
                2 * np.pi * operator,
                [n, (n + 1) % num_qubits],
            )
        return controls

    def _compute_params(self):
        num_qubits = self.num_qubits
        computed_params = {}
        computed_params["sx"] = _to_array(self.params["sx"], num_qubits)
        computed_params["sz"] = _to_array(self.params["sz"], num_qubits)
        num_coupling = self._get_num_coupling()
        computed_params["sxsy"] = _to_array(self.params["sxsy"], num_coupling)
        return computed_params

    def get_control_latex(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        num_qubits = self.num_qubits
        num_coupling = self._get_num_coupling()
        return [
            {f"sx{m}": r"$\sigma_x^{}$".format(m) for m in range(num_qubits)},
            {f"sz{m}": r"$\sigma_z^{}$".format(m) for m in range(num_qubits)},
            {
                f"g{m}": r"$\sigma_x^{}\sigma_x^{} +"
                r" \sigma_y^{}\sigma_y^{}$".format(
                    m, (m + 1) % num_qubits, m, (m + 1) % num_qubits
                )
                for m in range(num_coupling)
            },
        ]
