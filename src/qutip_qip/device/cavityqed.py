# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import warnings
from copy import deepcopy

import numpy as np

from qutip.operators import tensor, identity, destroy, sigmax, sigmaz
from qutip.states import basis
from ..circuit import QubitCircuit, Gate
from .processor import Processor
from .modelprocessor import ModelProcessor
from ..operations import expand_operator
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from ..pulse import Pulse
from ..compiler.gatecompiler import GateCompiler
from ..compiler import CavityQEDCompiler


__all__ = ['DispersiveCavityQED']


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

    def __init__(self, num_qubits, correct_global_phase=True,
                 num_levels=10, t1=None, t2=None, **params):
        super(DispersiveCavityQED, self).__init__(
            num_qubits, correct_global_phase=correct_global_phase,
            t1=t1, t2=t2)
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        self.num_levels = num_levels
        self.params = {  # default parameters
            "deltamax": 1.0,
            "epsmax": 9.5,
            "w0": 10,
            "eps": 9.5,
            "delta": 0.0,
            "g": 0.01,
        }
        if params is not None:
            self.params.update(params)
        for key, value in self.params.items():
            if key != "w0":
                # if float, make it an array
                self.params[key] = self.to_array(value, self.num_qubits)
        self.set_up_params()
        self.set_up_ops(num_qubits)
        self.dims = [num_levels] + [2] * num_qubits
        self.pulse_dict = self.get_pulse_dict()
        self.native_gates = ["SQRTISWAP", "ISWAP", "RX", "RZ"]

    def set_up_ops(self, num_qubits):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        num_qubits: int
            The number of qubits in the system.
        """
        # single qubit terms
        for m in range(num_qubits):
            self.add_control(2*np.pi*sigmax(), [m+1], label="sx" + str(m))
        for m in range(num_qubits):
            self.add_control(2*np.pi*sigmaz(), [m+1], label="sz" + str(m))
        # coupling terms
        a = tensor(
            [destroy(self.num_levels)] +
            [identity(2) for n in range(num_qubits)])
        for n in range(num_qubits):
            sm = tensor([identity(self.num_levels)] +
                        [destroy(2) if m == n else identity(2)
                         for m in range(num_qubits)])
            self.add_control(
                2*np.pi * a.dag() * sm + 2*np.pi * a * sm.dag(),
                list(range(num_qubits+1)), label="g" + str(n)
            )

    def set_up_params(self):
        """
        Compute the qubit frequency and detune.
        """
        # backward compatibility
        self.params["sz"] = self.params["epsmax"]
        self.params["sx"] = self.params["deltamax"]
        eps = self.params["eps"]
        delta = self.params["delta"]
        w0 = self.params["w0"]
        g = self.params["g"]
        # computed
        self.wq = np.sqrt(eps**2 + delta**2)
        self.Delta = self.wq - w0

        # rwa/dispersive regime tests
        if any(g / (w0 - self.wq) > 0.05):
            warnings.warn("Not in the dispersive regime")

        if any((w0 - self.wq)/(w0 + self.wq) > 0.05):
            warnings.warn(
                "The rotating-wave approximation might not be valid.")
        print(self.params)

    @property
    def sx_ops(self):
        """
        list: A list of sigmax Hamiltonians for each qubit.
        """
        return self.ctrls[0: self.num_qubits]

    @property
    def sz_ops(self):
        """
        list: A list of sigmaz Hamiltonians for each qubit.
        """
        return self.ctrls[self.num_qubits: 2*self.num_qubits]

    @property
    def cavityqubit_ops(self):
        """
        list: A list of interacting Hamiltonians between cavity and each qubit.
        """
        return self.ctrls[2*self.num_qubits: 3*self.num_qubits]

    @property
    def sx_u(self):
        """array-like: Pulse matrix for sigmax Hamiltonians."""
        return self.coeffs[: self.num_qubits]

    @property
    def sz_u(self):
        """array-like: Pulse matrix for sigmaz Hamiltonians."""
        return self.coeffs[self.num_qubits: 2*self.num_qubits]

    @property
    def g_u(self):
        """
        array-like: Pulse matrix for interacting Hamiltonians
        between cavity and each qubit.
        """
        return self.coeffs[2*self.num_qubits: 3*self.num_qubits]

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        return ([[r"$\sigma_x^%d$" % n for n in range(self.num_qubits)],
                 [r"$\sigma_z^%d$" % n for n in range(self.num_qubits)],
                 [r"$g_{%d}$" % (n) for n in range(self.num_qubits)]])

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        """
        psi_proj = tensor(
            [basis(self.num_levels, 0)] +
            [identity(2) for n in range(self.num_qubits)])
        return psi_proj.dag() * U * psi_proj

    def load_circuit(
            self, qc, schedule_mode="ASAP", compiler=None):
        if compiler is None:
            compiler = CavityQEDCompiler(
                self.num_qubits, self.params, global_phase=0.)
        tlist, coeff = super().load_circuit(
            qc, schedule_mode=schedule_mode, compiler=compiler)
        self.global_phase = compiler.global_phase
        return tlist, coeff
