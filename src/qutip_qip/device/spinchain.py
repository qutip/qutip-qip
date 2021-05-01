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
from copy import deepcopy

import numpy as np

from qutip import sigmax, sigmay, sigmaz, tensor
from ..circuit import QubitCircuit
from .modelprocessor import ModelProcessor
from ..pulse import Pulse
from ..compiler import SpinChainCompiler
from ..transpiler import to_chain_structure


__all__ = ['SpinChain', 'LinearSpinChain', 'CircularSpinChain']


class SpinChain(ModelProcessor):
    """
    The processor based on the physical implementation of
    a spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.
    It is a base class and should not be used directly, please
    refer the the subclasses :class:`.LinearSpinChain` and
    :class:`.CircularSpinChain`.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`.ModelProcessor`)

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `num_qubits` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size `num_qubits` or a float for all qubits.
    """
    def __init__(self, num_qubits, correct_global_phase,
                 sx, sz, sxsy, t1, t2, N=None):
        super(SpinChain, self).__init__(
            num_qubits, correct_global_phase=correct_global_phase,
            t1=t1, t2=t2, N=N)
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        self.pulse_dict = self.get_pulse_dict()
        self.native_gates = ["SQRTISWAP", "ISWAP", "RX", "RZ"]
        # params and ops are set in the submethods

    def set_up_ops(self, num_qubits):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        num_qubits: int
            The number of qubits in the system.
        """
        # sx_ops
        for m in range(num_qubits):
            self.add_control(sigmax(), m, label="sx" + str(m))
        # sz_ops
        for m in range(num_qubits):
            self.add_control(sigmaz(), m, label="sz" + str(m))
        # sxsy_ops
        operator = tensor([sigmax(), sigmax()]) + tensor([sigmay(), sigmay()])
        for n in range(num_qubits - 1):
            self.add_control(operator, [n, n+1], label="g" + str(n))

    def set_up_params(self, sx, sz):
        """
        Save the parameters in the attribute `params` and check the validity.
        The keys of `params` including "sx", "sz", and "sxsy", each
        mapped to a list for parameters corresponding to each qubits.
        For coupling strength "sxsy", list element i is the interaction
        between qubits i and i+1.
        All parameters will be multiplied by 2*pi for simplicity

        Parameters
        ----------
        sx: float or list
            The coefficient of sigmax in the model

        sz: flaot or list
            The coefficient of sigmaz in the model

        Notes
        -----
        The coefficient of sxsy is defined in the submethods.
        """
        sx_para = 2 * np.pi * self.to_array(sx, self.num_qubits)
        self._params["sx"] = sx_para
        sz_para = 2 * np.pi * self.to_array(sz, self.num_qubits)
        self._params["sz"] = sz_para

    @property
    def sx_ops(self):
        """list: A list of sigmax Hamiltonians for each qubit."""
        return self.ctrls[: self.num_qubits]

    @property
    def sz_ops(self):
        """list: A list of sigmaz Hamiltonians for each qubit."""
        return self.ctrls[self.num_qubits: 2*self.num_qubits]

    @property
    def sxsy_ops(self):
        """
        list: A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.
        """
        return self.ctrls[2*self.num_qubits:]

    @property
    def sx_u(self):
        """array-like: Pulse coefficients for sigmax Hamiltonians."""
        return self.coeffs[: self.num_qubits]

    @property
    def sz_u(self):
        """array-like: Pulse coefficients for sigmaz Hamiltonians."""
        return self.coeffs[self.num_qubits: 2*self.num_qubits]

    @property
    def sxsy_u(self):
        """
        array-like: Pulse coefficients for tensor(sigmax, sigmay)
        interacting Hamiltonians.
        """
        return self.coeffs[2*self.num_qubits:]

    def load_circuit(
            self, qc, setup, schedule_mode="ASAP", compiler=None):
        if compiler is None:
            compiler = SpinChainCompiler(
                self.num_qubits, self.params, setup=setup)
        tlist, coeffs = super().load_circuit(
            qc, schedule_mode=schedule_mode, compiler=compiler)
        self.global_phase = compiler.global_phase
        return tlist, coeffs


class LinearSpinChain(SpinChain):
    """
    A processor based on the physical implementation of
    a linear spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, num_qubits=None, correct_global_phase=True,
                 sx=0.25, sz=1.0, sxsy=0.1, t1=None, t2=None, N=None):

        super(LinearSpinChain, self).__init__(
            num_qubits, correct_global_phase=correct_global_phase,
            sx=sx, sz=sz, sxsy=sxsy, t1=t1, t2=t2, N=N)
        self.set_up_params(sx=sx, sz=sz, sxsy=sxsy)
        self.set_up_ops(num_qubits)

    def set_up_ops(self, num_qubits):
        super(LinearSpinChain, self).set_up_ops(num_qubits)

    def set_up_params(self, sx, sz, sxsy):
        # Doc same as in the parent class
        super(LinearSpinChain, self).set_up_params(sx, sz)
        sxsy_para = 2 * np.pi * self.to_array(sxsy, self.num_qubits-1)
        self._params["sxsy"] = sxsy_para

    @property
    def sxsy_ops(self):
        """
        list: A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.
        """
        return self.ctrls[2*self.num_qubits: 3*self.num_qubits-1]

    @property
    def sxsy_u(self):
        """
        array-like: Pulse coefficients for tensor(sigmax, sigmay)
        interacting Hamiltonians.
        """
        return self.coeffs[2*self.num_qubits: 3*self.num_qubits-1]

    def load_circuit(
            self, qc, schedule_mode="ASAP", compiler=None):
        return super(LinearSpinChain, self).load_circuit(
            qc, "linear", schedule_mode=schedule_mode, compiler=compiler)

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        return ([[r"$\sigma_x^%d$" % n for n in range(self.num_qubits)],
                [r"$\sigma_z^%d$" % n for n in range(self.num_qubits)],
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, n + 1, n, n + 1) for n in range(self.num_qubits - 1)],
                 ])

    def topology_map(self, qc):
        return to_chain_structure(qc, "linear")


class CircularSpinChain(SpinChain):
    """
    A processor based on the physical implementation of
    a circular spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, num_qubits=None, correct_global_phase=True,
                 sx=0.25, sz=1.0, sxsy=0.1, t1=None, t2=None, N=None):
        if num_qubits <= 1:
            raise ValueError(
                "Circuit spin chain must have at least 2 qubits. "
                "The number of qubits is increased to 2.")
        super(CircularSpinChain, self).__init__(
            num_qubits, correct_global_phase=correct_global_phase,
            sx=sx, sz=sz, sxsy=sxsy, t1=t1, t2=t2, N=N)
        self.set_up_params(sx=sx, sz=sz, sxsy=sxsy)
        self.set_up_ops(num_qubits)

    def set_up_ops(self, num_qubits):
        super(CircularSpinChain, self).set_up_ops(num_qubits)
        operator = tensor([sigmax(), sigmax()]) + tensor([sigmay(), sigmay()])
        self.add_control(
            operator, [num_qubits-1, 0], label="g" + str(num_qubits-1))

    def set_up_params(self, sx, sz, sxsy):
        # Doc same as in the parent class
        super(CircularSpinChain, self).set_up_params(sx, sz)
        sxsy_para = 2 * np.pi * self.to_array(sxsy, self.num_qubits)
        self._params["sxsy"] = sxsy_para

    @property
    def sxsy_ops(self):
        """
        list: A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.
        """
        return self.ctrls[2*self.num_qubits: 3*self.num_qubits]

    @property
    def sxsy_u(self):
        """
        array-like: Pulse coefficients for tensor(sigmax, sigmay)
        interacting Hamiltonians.
        """
        return self.coeffs[2*self.num_qubits: 3*self.num_qubits]

    def load_circuit(
            self, qc, schedule_mode="ASAP", compiler=None):
        return super(CircularSpinChain, self).load_circuit(
            qc, "circular", schedule_mode=schedule_mode, compiler=compiler)

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        return ([[r"$\sigma_x^%d$" % n for n in range(self.num_qubits)],
                [r"$\sigma_z^%d$" % n for n in range(self.num_qubits)],
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, (n + 1) % self.num_qubits, n, (n + 1) % self.num_qubits)
                 for n in range(self.num_qubits)]])

    def topology_map(self, qc):
        return to_chain_structure(qc, "circular")
