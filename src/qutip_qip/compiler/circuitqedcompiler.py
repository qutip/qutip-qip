from functools import partial
import numpy as np

from ..circuit import Gate
from ..compiler import GateCompiler, Instruction


__all__ = ['SCQubitsCompiler']


class SCQubitsCompiler(GateCompiler):
    """
    Compiler for :class:`.SCQubits`.
    Compiled pulse strength is in the unit of GHz.

    Supported native gates: "RX", "RY", "CNOT".

    Default configuration (see :obj:`.GateCompiler.args` and
    :obj:`.GateCompiler.compile`):

        +-----------------+-----------------------+
        | key             | value                 |
        +=================+=======================+
        | ``shape``       | ``hann``              |
        +-----------------+-----------------------+
        |``num_samples``  | 1000                  |
        +-----------------+-----------------------+
        |``params``       | Hardware Parameters   |
        +-----------------+-----------------------+
    """
    def __init__(self, num_qubits, params):
        super(SCQubitsCompiler, self).__init__(num_qubits, params=params)
        self.gate_compiler.update({
            "RX": partial(
                self.default_single_qubit_compiler, param_name="omega_single"),
            "RY": partial(
                self.default_single_qubit_compiler, param_name="omega_single"),
            "CNOT": self.cnot_compiler,
            })
        self.args = {  # Default configuration
            "shape": "hann",
            "num_samples": 1000,
            "params": self.params,
            }

    def _normalized_gauss_pulse(self):
        """
        Return a truncated and normalize Gaussian curve.
        The returned pulse is truncated from a Gaussian distribution with
        -3*sigma < t < 3*sigma.
        The amplitude is shifted so that the pulse start from 0.
        In addition, the pulse is normalized so that
        the total integral area is 1.
        """
        #  td normalization so that the total integral area is 1
        td = 2.4384880692912567
        sigma = 1/6 * td  # 3 sigma
        tlist = np.linspace(0, td, 1000)
        max_pulse = 1 - np.exp(-(0-td/2)**2/2/sigma**2)
        coeff = (
            np.exp(-(tlist-td/2)**2/2/sigma**2)
            - np.exp(-(0-td/2)**2/2/sigma**2)
            ) / max_pulse
        return tlist, coeff

    def single_qubit_compiler(self, gate, args):
        """
        Compiler for the RX and RY gate.
        """
        return partial(
            self.default_single_qubit_compiler, param_name="omega_single")

    def cnot_compiler(self, gate, args):
        """
        Compiler for CNOT gate using the cross resonance iteraction.
        See
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.134507
        for reference.

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        result = []
        q1 = gate.controls[0]
        q2 = gate.targets[0]

        gate1 = Gate("RX", q2, arg_value=-np.pi/2)
        result += self.gate_compiler[gate1.name](gate1, args)

        zx_coeff = self.params["zx_coeff"][q1]
        tlist, coeff = self._normalized_gauss_pulse()
        amplitude = zx_coeff
        area = 1/2
        sign = np.sign(amplitude) * np.sign(area)
        tlist = tlist / amplitude * area * sign
        coeff = coeff * amplitude * sign
        from scipy.integrate import simps
        print(amplitude)
        print(simps(coeff, tlist))
        coeff, tlist = self.generate_pulse_shape(
            args["shape"], args["num_samples"], maximum=zx_coeff, area=area)
        print(simps(coeff, tlist))
        pulse_info = [("zx" + str(q1) + str(q2), coeff)]
        result += [Instruction(gate, tlist, pulse_info)]

        gate3 = Gate("RX", q1, arg_value=-np.pi/2)
        result += self.gate_compiler[gate3.name](gate3, args)
        gate4 = Gate("RY", q1, arg_value=-np.pi/2)
        result += self.gate_compiler[gate4.name](gate4, args)
        gate5 = Gate("RX", q1, arg_value=np.pi/2)
        result += self.gate_compiler[gate5.name](gate5, args)
        return result
