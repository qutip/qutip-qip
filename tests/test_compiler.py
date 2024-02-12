import pytest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import integrate
from qutip_qip.compiler.gatecompiler import _default_window_t_max

from qutip_qip.device import (
    DispersiveCavityQED, CircularSpinChain, LinearSpinChain)
from qutip_qip.compiler import (
    SpinChainCompiler, CavityQEDCompiler, Instruction, GateCompiler
    )
from qutip_qip.circuit import QubitCircuit
from qutip import basis, fidelity


def test_compiling_with_scheduler():
    """
    Here we test if the compiling with scheduler works properly.
    The non scheduled pulse should be twice as long as the scheduled one.
    The numerical results are tested in test_device.py
    """
    circuit = QubitCircuit(2)
    circuit.add_gate("X", 0)
    circuit.add_gate("X", 1)
    processor = DispersiveCavityQED(2)

    processor.load_circuit(circuit, schedule_mode=None)
    tlist = processor.get_full_tlist()
    time_not_scheduled = tlist[-1]-tlist[0]

    processor.load_circuit(circuit, schedule_mode="ASAP")
    tlist = processor.get_full_tlist()
    time_scheduled1 = tlist[-1]-tlist[0]

    processor.load_circuit(circuit, schedule_mode="ALAP")
    tlist = processor.get_full_tlist()
    time_scheduled2 = tlist[-1]-tlist[0]

    assert(abs(time_scheduled1 * 2 - time_not_scheduled) < 1.0e-10)
    assert(abs(time_scheduled2 * 2 - time_not_scheduled) < 1.0e-10)


def test_compiling_gates_different_sampling_number():
    """
    Define compiler without a physical model.
    Test compiling gates to pulses with different sampling number.
    """

    class MockCompiler(GateCompiler):
        def __init__(self, num_qubits, params=None):
            super().__init__(num_qubits, params=params)
            self.gate_compiler["U1"] = self.single_qubit_gate_compiler
            self.gate_compiler["U2"] = self.two_qubit_gate_compiler
            self.args.update({"params": params})

        def single_qubit_gate_compiler(self, gate, args):
            pulse_info = [("x", np.array([1.0] * 3))]
            return [
                Instruction(
                    gate, tlist=np.linspace(0, 2, 3), pulse_info=pulse_info
                )
            ]

        def two_qubit_gate_compiler(self, gate, args):
            pulse_info = [("xx", np.array([2.0] * 5))]
            return [
                Instruction(
                    gate, tlist=np.linspace(0, 4, 5), pulse_info=pulse_info
                )
            ]

    num_qubits = 2
    circuit = QubitCircuit(num_qubits)
    circuit.add_gate("U1", targets=0, arg_value=1.0)
    circuit.add_gate("U2", targets=[0, 1], arg_value=1.0)
    circuit.add_gate("U1", targets=0, arg_value=1.0)

    compiler = MockCompiler(num_qubits=2)
    compiled_tlists, compiled_coeffs = compiler.compile(circuit)

    # Filter out the nonzero part of the pulse
    # and check if they are correct.
    np.testing.assert_array_equal(
        compiled_tlists["x"][np.nonzero(compiled_coeffs["x"])[0]],
        np.array([1, 2, 7, 8]),
    )
    np.testing.assert_array_equal(
        compiled_tlists["xx"][np.nonzero(compiled_coeffs["xx"])[0]],
        np.array([3, 4, 5, 6]),
    )


# Test the compiler with a physical model.
class MyCompiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params):
        super(MyCompiler, self).__init__(num_qubits, params=params)
        # pass our compiler function as a compiler for RX (rotation around X) gate.
        self.gate_compiler["RX"] = self.rx_compiler
        self.args.update({"params": params})

    def rx_compiler(self, gate, args):
        targets = gate.targets
        coeff, tlist = self.generate_pulse_shape(
            "hann",
            1000,
            maximum=args["params"]["sx"][targets[0]],
            # The operator is Pauli Z/X/Y, without 1/2.
            area=gate.arg_value / 2.0 / np.pi * 0.5,
        )
        pulse_info = [("sx" + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]


spline_kind = [
    pytest.param("step_func", id = "discrete"),
    pytest.param("cubic", id = "continuos"),
]
schedule_mode = [
    pytest.param("ASAP", id = "ASAP"),
    pytest.param("ALAP", id="ALAP"),
    pytest.param(False, id = "No schedule"),
]
@pytest.mark.parametrize("spline_kind", spline_kind)
@pytest.mark.parametrize("schedule_mode", schedule_mode)
def test_compiler_with_continous_pulse(spline_kind, schedule_mode):
    num_qubits = 2
    circuit = QubitCircuit(num_qubits)
    circuit.add_gate("X", targets=0)
    circuit.add_gate("X", targets=1)
    circuit.add_gate("X", targets=0)

    processor = CircularSpinChain(num_qubits)
    processor.spline_kind = spline_kind
    gauss_compiler = MyCompiler(num_qubits, processor.params)
    processor.load_circuit(
        circuit, schedule_mode = schedule_mode, compiler=gauss_compiler)
    result = processor.run_state(init_state = basis([2,2], [0,0]))
    assert(abs(fidelity(result.states[-1],basis([2,2],[0,1])) - 1) < 1.e-5)


def rx_compiler_without_pulse_dict(gate, args):
    """
    Define a gate compiler that does not use pulse_dict but directly
    give the index of control pulses in the Processor.
    """
    targets = gate.targets
    g = args["params"]["sx"][targets[0]]
    coeff = np.sign(gate.arg_value) * g
    tlist = abs(gate.arg_value) / (2 * g) / np.pi/ 2
    pulse_info = [(targets[0], coeff)]
    return [Instruction(gate, tlist, pulse_info)]


def test_compiler_without_pulse_dict():
    """
    Test for a compiler function without pulse_dict and using args.
    """
    num_qubits = 2
    circuit = QubitCircuit(num_qubits)
    circuit.add_gate("X", targets=[0])
    circuit.add_gate("X", targets=[1])
    processor = CircularSpinChain(num_qubits)
    compiler = SpinChainCompiler(num_qubits, params=processor.params, setup="circular")
    compiler.gate_compiler["RX"] = rx_compiler_without_pulse_dict
    compiler.args = {"params": processor.params}
    processor.load_circuit(circuit, compiler=compiler)
    result = processor.run_state(basis([2,2], [0,0]))
    assert(abs(fidelity(result.states[-1], basis([2,2], [1,1])) - 1.) < 1.e-6 )


def test_compiler_result_format():
    """
    Test if compiler return correctly different kind of compiler result
    and if processor can successfully read them.
    """
    num_qubits = 1
    circuit = QubitCircuit(num_qubits)
    circuit.add_gate("RX", targets=[0], arg_value=np.pi/2)
    processor = LinearSpinChain(num_qubits)
    compiler = SpinChainCompiler(num_qubits, params=processor.params, setup="circular")

    tlist, coeffs = compiler.compile(circuit)
    assert(isinstance(tlist, dict))
    assert("sx0" in tlist)
    assert(isinstance(coeffs, dict))
    assert("sx0" in coeffs)
    processor.coeffs = coeffs
    processor.set_all_tlist(tlist)
    assert_array_equal(processor.pulses[0].coeff, coeffs["sx0"])
    assert_array_equal(processor.pulses[0].tlist, tlist["sx0"])

    compiler.gate_compiler["RX"] = rx_compiler_without_pulse_dict
    tlist, coeffs = compiler.compile(circuit)
    assert(isinstance(tlist, dict))
    assert(0 in tlist)
    assert(isinstance(coeffs, dict))
    assert(0 in coeffs)
    processor.coeffs = coeffs
    processor.set_all_tlist(tlist)
    assert_array_equal(processor.pulses[0].coeff, coeffs[0])
    assert_array_equal(processor.pulses[0].tlist, tlist[0])


@pytest.mark.parametrize(
    "shape", list(_default_window_t_max.keys()))
def test_pulse_shape(shape):
    """Test different pulse shape functions"""
    coeff, tlist = GateCompiler.generate_pulse_shape(
        shape, 1001, maximum=1.0, area=1.0)
    assert pytest.approx(coeff[500], 1.e-2) == 1  # max
    result = integrate.trapezoid(coeff, tlist)
    assert pytest.approx(result, rel=1.e-2) == 1  # area
