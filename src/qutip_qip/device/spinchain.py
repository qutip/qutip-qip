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

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size ``num_qubits`` or a float for all qubits.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit. A list of size ``num_qubits`` or a float for all qubits.

    **params:
        Keyword argument for hardware parameters, in the unit of frequency
        (MHz, GHz etc, the unit of time list needs to be adjusted accordingly).
        Qubit parameters can either be a float or an array of the length
        ``num_qubits``.
        ``sxsy``, should be either a float or an array of the length
        ``num_qubits-1`` (for :class:`.LinearSpinChain`) or ``num_qubits``
        (for :`.CircularSpinChain`).

        - ``sx``: the pulse strength of sigma-x control, default ``0.25``
        - ``sz``: the pulse strength of sigma-z control, default ``1.0``
        - ``sxsy``: the pulse strength for the exchange interaction,
          default ``0.1``
    """

    def __init__(
        self, num_qubits, correct_global_phase, t1, t2, N=None, **params
    ):
        super(SpinChain, self).__init__(
            num_qubits,
            correct_global_phase=correct_global_phase,
            t1=t1,
            t2=t2,
            N=N,
        )
        self.params["num_qubits"] = num_qubits
        if params is not None:
            self.params.update(params)
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
    Spin chain model with open-end topology. See :class:`.SpinChain`
    for details.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size ``num_qubits`` or a float for all qubits.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit. A list of size ``num_qubits`` or a float for all qubits.

    **params:
        Keyword argument for hardware parameters, in the unit of frequency
        (MHz, GHz etc, the unit of time list needs to be adjusted accordingly).
        Qubit parameters can either be a float or an array of the length
        ``num_qubits``.
        ``sxsy``, should be either a float or an array of the length
        ``num_qubits-1``.

        - ``sx``: the pulse strength of sigma-x control, default ``0.25``
        - ``sz``: the pulse strength of sigma-z control, default ``1.0``
        - ``sxsy``: the pulse strength for the exchange interaction,
          default ``0.1``
    """

    def __init__(
        self,
        num_qubits=None,
        correct_global_phase=True,
        t1=None,
        t2=None,
        N=None,
        **params,
    ):
        super(LinearSpinChain, self).__init__(
            num_qubits,
            correct_global_phase=correct_global_phase,
            t1=t1,
            t2=t2,
            N=N,
            **params,
        )
        self.model = SpinChainModel(setup="linear", **self.params)
        self.params = self.model.params

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

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size ``num_qubits`` or a float for all qubits.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit. A list of size ``num_qubits`` or a float for all qubits.

    **params:
        Keyword argument for hardware parameters, in the unit of frequency
        (MHz, GHz etc, the unit of time list needs to be adjusted accordingly).
        Qubit parameters can either be a float or an array of the length
        ``num_qubits``.
        ``sxsy``, should be either a float or an array of the length
        ``num_qubits``.

        - ``sx``: the pulse strength of sigma-x control, default ``0.25``
        - ``sz``: the pulse strength of sigma-z control, default ``1.0``
        - ``sxsy``: the pulse strength for the exchange interaction,
          default ``0.1``
    """

    def __init__(
        self,
        num_qubits=None,
        correct_global_phase=True,
        t1=None,
        t2=None,
        N=None,
        **params,
    ):
        if num_qubits <= 1:
            raise ValueError(
                "Circuit spin chain must have at least 2 qubits. "
                "The number of qubits is increased to 2."
            )
        super(CircularSpinChain, self).__init__(
            num_qubits,
            correct_global_phase=correct_global_phase,
            t1=t1,
            t2=t2,
            N=N,
            **params,
        )
        self.model = SpinChainModel(setup="circular", **self.params)
        self.params = self.model.params

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
    def __init__(self, **params):
        self.params = {  # default parameters, in the unit of frequency
            "sx": 0.25,
            "sz": 1.0,
            "sxsy": 0.1,
        }
        self._drift = []
        self.params.update(deepcopy(params))
        self._controls = self._set_up_controls(self.params["num_qubits"])
        self.params.update(self._compute_params())

    def get_all_drift(self):
        return self._drift

    @property
    def _old_index_label_map(self):
        num_qubits = self.params["num_qubits"]
        return (
            ["sx" + str(i) for i in range(num_qubits)]
            + ["sz" + str(i) for i in range(num_qubits)]
            + ["g" + str(i) for i in range(num_qubits)]
        )

    def get_control(self, label):
        """label should be hashable"""
        _old_index_label_map = self._old_index_label_map
        if isinstance(label, int):
            label = _old_index_label_map[label]
        return self._controls[label]

    def get_control_labels(self):
        return list(self._controls.keys())

    def _get_num_coupling(self):
        if self.params["setup"] == "linear":
            num_coupling = self.params["num_qubits"] - 1
        elif self.params["setup"] == "circular":
            num_coupling = self.params["num_qubits"]
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
        num_qubits = self.params["num_qubits"]
        computed_params = {}
        computed_params["sx"] = _to_array(self.params["sx"], num_qubits)
        computed_params["sz"] = _to_array(self.params["sz"], num_qubits)
        num_coupling = self._get_num_coupling()
        computed_params["sxsy"] = _to_array(self.params["sxsy"], num_coupling)
        return computed_params

    def get_latex_str(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        num_qubits = self.params["num_qubits"]
        num_coupling = self._get_num_coupling()
        return [
            {f"sx{m}": f"$\sigma_x^{m}$" for m in range(num_qubits)},
            {f"sz{m}": f"$\sigma_z^{m}$" for m in range(num_qubits)},
            {f"g{m}": f"$\sigma_x^{m}\sigma_x^{m} + \sigma_y^{m}\sigma_y^{m}$" for m in range(num_coupling)},
        ]
