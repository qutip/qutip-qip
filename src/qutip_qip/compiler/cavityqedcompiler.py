import numpy as np

from ..circuit import QubitCircuit
from ..operations import Gate
from ..compiler import GateCompiler, Instruction


__all__ = ['CavityQEDCompiler']


class CavityQEDCompiler(GateCompiler):
    """
    Decompose a :class:`.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters.
        See :meth:`.DispersiveCavityQED.set_up_params` for the definition.

    global_phase: float, optional
        Record of the global phase change and will be returned.

    Attributes
    ----------
    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.
    """
    def __init__(
            self, num_qubits, params, global_phase=0.,
            pulse_dict=None, N=None):
        super(CavityQEDCompiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict, N=N)
        self.gate_compiler.update({
            "ISWAP": self.iswap_compiler,
            "SQRTISWAP": self.sqrtiswap_compiler,
            "RZ": self.rz_compiler,
            "RX": self.rx_compiler,
            "GLOBALPHASE": self.globalphase_compiler
            })
        self.wq = np.sqrt(self.params["eps"]**2 + self.params["delta"]**2)
        self.Delta = self.wq - self.params["w0"]
        self.global_phase = global_phase

    def rz_compiler(self, gate, args):
        """
        Compiler for the RZ gate
        """
        targets = gate.targets
        g = self.params["sz"][targets[0]]
        coeff = np.sign(gate.arg_value) * g
        tlist = abs(gate.arg_value) / (2*g) / np.pi / 2
        pulse_info = [("sz" + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def rx_compiler(self, gate, args):
        """
        Compiler for the RX gate
        """
        targets = gate.targets
        g = self.params["sx"][targets[0]]
        coeff = np.sign(gate.arg_value) * g
        tlist = abs(gate.arg_value) / (2*g) / np.pi / 2
        pulse_info = [("sx" + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def sqrtiswap_compiler(self, gate, args):
        """
        Compiler for the SQRTISWAP gate

        Notes
        -----
        This version of sqrtiswap_compiler has very low fidelity, please use
        iswap
        """
        # FIXME This decomposition has poor behaviour
        q1, q2 = gate.targets
        pulse_info = []
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q1] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q2] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q1)
        coeff = self.params["g"][q1]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q2)
        coeff = self.params["g"][q2]
        pulse_info += [(pulse_name, coeff)]

        J = self.params["g"][q1] * self.params["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        tlist = (4 * np.pi / abs(J)) / 8 / np.pi / 2
        instruction_list = [Instruction(gate, tlist, pulse_info)]

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=-np.pi/4)
        compiled_gate1 = self.rz_compiler(gate1, args)
        instruction_list += compiled_gate1
        gate2 = Gate("RZ", [q2], None, arg_value=-np.pi/4)
        compiled_gate2 = self.rz_compiler(gate2, args)
        instruction_list += compiled_gate2
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=-np.pi/4)
        self.globalphase_compiler(gate3, args)
        return instruction_list

    def iswap_compiler(self, gate, args):
        """
        Compiler for the ISWAP gate
        """
        q1, q2 = gate.targets
        pulse_info = []
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q1] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "sz" + str(q2)
        coeff = self.wq[q2] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q1)
        coeff = self.params["g"][q1]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q2)
        coeff = self.params["g"][q2]
        pulse_info += [(pulse_name, coeff)]

        J = self.params["g"][q1] * self.params["g"][q2] * (
            1 / self.Delta[q1] + 1 / self.Delta[q2]) / 2
        tlist = (4 * np.pi / abs(J)) / 4 / np.pi / 2
        instruction_list = [Instruction(gate, tlist, pulse_info)]

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=-np.pi/2.)
        compiled_gate1 = self.rz_compiler(gate1, args)
        instruction_list += compiled_gate1
        gate2 = Gate("RZ", [q2], None, arg_value=-np.pi/2)
        compiled_gate2 = self.rz_compiler(gate2, args)
        instruction_list += compiled_gate2
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=-np.pi/2)
        self.globalphase_compiler(gate3, args)
        return instruction_list

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
