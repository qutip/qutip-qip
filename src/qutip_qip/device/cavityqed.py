import warnings
from copy import deepcopy

import numpy as np

from qutip import (
    tensor,
    identity,
    destroy,
    sigmax,
    sigmaz,
    basis,
    Qobj,
    QobjEvo,
)
from ..circuit import QubitCircuit
from ..operations import Gate
from .processor import Processor, Model
from .modelprocessor import ModelProcessor, _to_array
from ..operations import expand_operator
from ..pulse import Pulse
from ..compiler import GateCompiler, CavityQEDCompiler


__all__ = ["DispersiveCavityQED"]


class DispersiveCavityQED(ModelProcessor):
    """
    The processor based on the physical implementation of
    a dispersive cavity QED system.
    The available Hamiltonian of the system is predefined.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`.ModelProcessor`)

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float, optional
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    num_levels: int, optional
        The number of energy levels in the resonator.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `num_qubits` or a float for all qubits.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit. A list of size `num_qubits` or a float for all qubits.

    **params:
        Keyword argument for hardware parameters, in the unit of GHz.
        Qubit parameters can either be a float or a list of the length
        ``num_qubits``.

        - ``deltamax``: the pulse strength of sigma-x control, default ``1.0``
        - ``epsmax``: the pulse strength of sigma-z control, default ``9.5``
        - ``eps``: the bare transition frequency for each of the qubits,
          default ``9.5``
        - ``delta``: the coupling between qubit states, default ``0.0``
        - ``g``: the coupling strength between the resonator and the qubit,
          default ``1.0``
        - ``w0``: the bare frequency of the resonator. Should only be a float,
          default ``0.01``

        The dressed qubit frequency is `wq` is computed by
        :math:`w_q=\sqrt{\epsilon^2+\delta^2}`

    Attributes
    ----------
    wq: list of float
        The frequency of the qubits calculated from
        eps and delta for each qubit.

    Delta: list of float
        The detuning with respect to w0 calculated
        from wq and w0 for each qubit.
    """

    def __init__(
        self,
        num_qubits,
        correct_global_phase=True,
        num_levels=10,
        t1=None,
        t2=None,
        **params
    ):
        super(DispersiveCavityQED, self).__init__(
            num_qubits, correct_global_phase=correct_global_phase, t1=t1, t2=t2
        )
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        self.num_levels = num_levels
        self.params = {}
        if params is not None:
            self.params.update(params)
        self.params["num_qubits"] = num_qubits
        self.params["num_levels"] = num_levels
        self.model = DispersiveCavityQEDModel(**self.params)
        self.params = self.model.params
        self.dims = [num_levels] + [2] * num_qubits
        self.native_gates = ["SQRTISWAP", "ISWAP", "RX", "RZ"]

    @property
    def sx_ops(self):
        """
        list: A list of sigmax Hamiltonians for each qubit.
        """
        return self.ctrls[0 : self.num_qubits]

    @property
    def sz_ops(self):
        """
        list: A list of sigmaz Hamiltonians for each qubit.
        """
        return self.ctrls[self.num_qubits : 2 * self.num_qubits]

    @property
    def cavityqubit_ops(self):
        """
        list: A list of interacting Hamiltonians between cavity and each qubit.
        """
        return self.ctrls[2 * self.num_qubits : 3 * self.num_qubits]

    @property
    def sx_u(self):
        """array-like: Pulse matrix for sigmax Hamiltonians."""
        return self.coeffs[: self.num_qubits]

    @property
    def sz_u(self):
        """array-like: Pulse matrix for sigmaz Hamiltonians."""
        return self.coeffs[self.num_qubits : 2 * self.num_qubits]

    @property
    def g_u(self):
        """
        array-like: Pulse matrix for interacting Hamiltonians
        between cavity and each qubit.
        """
        return self.coeffs[2 * self.num_qubits : 3 * self.num_qubits]

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        """
        psi_proj = tensor(
            [basis(self.num_levels, 0)]
            + [identity(2) for n in range(self.num_qubits)]
        )
        return psi_proj.dag() * U * psi_proj

    def load_circuit(self, qc, schedule_mode="ASAP", compiler=None):
        if compiler is None:
            compiler = CavityQEDCompiler(
                self.num_qubits, self.params, global_phase=0.0
            )
        tlist, coeff = super().load_circuit(
            qc, schedule_mode=schedule_mode, compiler=compiler
        )
        self.global_phase = compiler.global_phase
        return tlist, coeff


class DispersiveCavityQEDModel(Model):
    def __init__(self, **params):
        self.params = {  # default parameters
            "deltamax": 1.0,
            "epsmax": 9.5,
            "w0": 10,
            "eps": 9.5,
            "delta": 0.0,
            "g": 0.01,
        }
        self._drift = []
        self.params.update(deepcopy(params))
        self._controls = self._set_up_controls()
        self._compute_params()

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

    def _set_up_controls(self):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        num_qubits: int
            The number of qubits in the system.
        """
        controls = {}
        num_qubits = self.params["num_qubits"]
        num_levels = self.params["num_levels"]
        # single qubit terms
        for m in range(num_qubits):
            controls["sx" + str(m)] = (2 * np.pi * sigmax(), [m + 1])
        for m in range(num_qubits):
            controls["sz" + str(m)] = (2 * np.pi * sigmaz(), [m + 1])
        # coupling terms
        a = tensor(
            [destroy(num_levels)] + [identity(2) for n in range(num_qubits)]
        )
        for n in range(num_qubits):
            # FIXME expanded?
            sm = tensor(
                [identity(num_levels)]
                + [
                    destroy(2) if m == n else identity(2)
                    for m in range(num_qubits)
                ]
            )
            controls["g" + str(n)] = (
                2 * np.pi * a.dag() * sm + 2 * np.pi * a * sm.dag(),
                list(range(num_qubits + 1)),
            )
        return controls

    def _compute_params(self):
        """
        Compute the qubit frequency and detune.
        """
        num_qubits = self.params["num_qubits"]
        w0 = self.params["w0"]  # only one resonator
        # same parameters for all qubits if it is not a list
        for name in ["epsmax", "deltamax", "eps", "delta", "g"]:
            self.params[name] = _to_array(self.params[name], num_qubits)

        # backward compatibility
        self.params["sz"] = self.params["epsmax"]
        self.params["sx"] = self.params["deltamax"]

        # computed
        wq = np.sqrt(self.params["eps"] ** 2 + self.params["delta"] ** 2)
        self.params["wq"] = wq
        self.params["Delta"] = wq - w0

        # rwa/dispersive regime tests
        if any(self.params["g"] / (w0 - wq) > 0.05):
            warnings.warn("Not in the dispersive regime")

        if any((w0 - wq) / (w0 + wq) > 0.05):
            warnings.warn(
                "The rotating-wave approximation might not be valid."
            )

    def get_latex_str(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        num_qubits = self.params["num_qubits"]
        return [
            {f"sx{m}": f"$\sigma_x^{m}$" for m in range(num_qubits)},
            {f"sz{m}": f"$\sigma_z^{m}$" for m in range(num_qubits)},
            {f"g{m}": f"$g^{m}$" for m in range(num_qubits)},
        ]
