from concurrent.futures import ProcessPoolExecutor  # for parallel simulations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qutip import (
    fidelity,
    sigmax,
    sigmay,
    sigmaz,
    basis,
    qeye,
    tensor,
    Qobj,
    fock_dm,
)
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import AngleParametricGate
from qutip_qip.operations.gates import RX, RY, Z
from qutip_qip.device import ModelProcessor, Model
from qutip_qip.compiler import GateCompiler, PulseInstruction
from qutip_qip.noise import Noise

plt.rcParams.update({"text.usetex": False, "font.size": 10})
LINEWIDTH = 3.48692403487
TEXTWIDTH = 7.1398920714


class ROT(AngleParametricGate):
    num_qubits = 1
    num_params = 1

    def __init__(self, arg_value):
        super().__init__(arg_value)

    def get_qobj(self, dtype) -> Qobj:
        # This is not required, because Pulse level implementation
        # of this gate is already provided.
        pass


class MyModel(Model):
    """A custom Hamiltonian model with sigmax and sigmay control."""

    def get_control(self, label):
        """
        Get an avaliable control Hamiltonian.
        For instance, sigmax control on the zeroth qubits is labeld "sx0".

        Args:
            label (str): The label of the Hamiltonian

        Returns:
            The Hamiltonian and target qubits as a tuple (qutip.Qobj, list).
        """
        targets = int(label[2:])
        if label[:2] == "sx":
            return 2 * np.pi * sigmax() / 2, [targets]
        elif label[:2] == "sy":
            return 2 * np.pi * sigmay() / 2, [targets]
        else:
            raise ValueError("Unknown control.")


class MyCompiler(GateCompiler):
    """Custom compiler for generating pulses from gates using the base class
    GateCompiler.

    Args:
        num_qubits (int): The number of qubits in the processor
        params (dict): A dictionary of parameters for gate pulses such as
                       the pulse amplitude.
    """

    def __init__(self, num_qubits, params):
        super().__init__(num_qubits, params=params)
        self.params = params
        self.gate_compiler = {
            ROT: self.rotation_with_phase_compiler,
            RX: self.single_qubit_gate_compiler,
            RY: self.single_qubit_gate_compiler,
        }

    def generate_pulse(self, circ_op, tlist, coeff, phase=0.0):
        """Generates the pulses.

        Args:
            circ_op (qutip_qip.circuit.GateInstruction): A GateInstruction object.
            tlist (array): A list of times for the evolution.
            coeff (array): An array of coefficients for the gate pulses
            phase (float): The value of the phase for the gate.

        Returns:
            PulseInstruction (qutip_qip.compiler.instruction.PulseInstruction):
            A PulseInstruction to implement a gate containing the control pulses.
        """

        pulse_info = [
            # (control label, coeff)
            ("sx" + str(circ_op.targets[0]), np.cos(phase) * coeff),
            ("sy" + str(circ_op.targets[0]), np.sin(phase) * coeff),
        ]
        return [PulseInstruction(circ_op, tlist=tlist, pulse_info=pulse_info)]

    def single_qubit_gate_compiler(self, circ_op, args):
        """Compiles single qubit gates to pulses.

        Args:
            circ_op (qutip_qip.circuit.GateInstruction)

        Returns:
            PulseInstruction (qutip_qip.compiler.instruction.PulseInstruction):
            A pulse instruction to implement a gate containing the control pulses.
        """
        # gate.arg_value is the rotation angle
        gate = circ_op.operation
        theta = gate.arg_value[0]

        tlist = np.abs(theta) / self.params["pulse_amplitude"]
        coeff = self.params["pulse_amplitude"] * np.sign(theta)
        coeff /= 2 * np.pi
        if gate.name == "RX":
            return self.generate_pulse(circ_op, tlist, coeff, phase=0.0)
        elif gate.name == "RY":
            return self.generate_pulse(circ_op, tlist, coeff, phase=np.pi / 2)

    def rotation_with_phase_compiler(self, circ_op, args):
        """Compiles gates with a phase term.

        Args:
            circ_op (qutip_qip.circuit.GateInstruction):

        Returns:
            PulseInstruction (qutip_qip.compiler.instruction.PulseInstruction):
            A pulse instruction to implement a gate containing the control pulses.
        """
        # gate.arg_value is the pulse phase
        phase = circ_op.operation.arg_value[0]
        tlist = self.params["duration"]
        coeff = self.params["pulse_amplitude"]
        coeff /= 2 * np.pi
        return self.generate_pulse(circ_op, tlist, coeff, phase=phase)


# Define a circuit and run the simulation
num_qubits = 1

circuit = QubitCircuit(1)
circuit.add_gate(RX(np.pi / 2), targets=0)
circuit.add_gate(Z, targets=0)
result1 = circuit.run(basis(2, 0))

mycompiler = MyCompiler(num_qubits, {"pulse_amplitude": 0.02})

myprocessor = ModelProcessor(model=MyModel(num_qubits))
myprocessor.native_gates = ["RX", "RY"]
myprocessor.load_circuit(circuit, compiler=mycompiler)

result2 = myprocessor.run_state(basis(2, 0)).states[-1]
assert abs(fidelity(result1, result2) - 1) < 1.0e-5

fig, ax = myprocessor.plot_pulses(
    figsize=(LINEWIDTH * 0.7, LINEWIDTH / 2 * 0.7),
    dpi=200,
    use_control_latex=False,
)
ax[-1].set_xlabel("$t$")
fig.tight_layout()
fig.savefig("custom_compiler_pulse.pdf")
fig.show()


class ClassicalCrossTalk(Noise):
    def __init__(self, ratio):
        self.ratio = ratio

    def get_noisy_pulses(self, dims=None, pulses=None, systematic_noise=None):
        """Adds noise to the control pulses.

        Args:
            dims: Dimension of the system, e.g., [2,2,2,...] for qubits.
            pulses: A list of Pulse objects, representing the compiled pulses.
            systematic_noise: A Pulse object with no ideal control, used to represent
            pulse-independent noise such as decoherence (not used in this example).
        Returns:
            pulses: The list of modified pulses according to the noise model.
            systematic_noise: A Pulse object (not used in this example).
        """
        for i, pulse in enumerate(pulses):
            if "sx" not in pulse.label and "sy" not in pulse.label:
                continue  # filter out other pulses, e.g. drift
            target = pulse.targets[0]
            if target != 0:  # add pulse to the left neighbour
                pulses[i].add_control_noise(
                    self.ratio * pulse.qobj,
                    targets=[target - 1],
                    coeff=pulse.coeff,
                    tlist=pulse.tlist,
                )
            if target != len(dims) - 1:  # add pulse to the right neighbour
                pulses[i].add_control_noise(
                    self.ratio * pulse.qobj,
                    targets=[target + 1],
                    coeff=pulse.coeff,
                    tlist=pulse.tlist,
                )
        return pulses, systematic_noise


def single_crosstalk_simulation(num_gates):
    """A single simulation, with num_gates representing the number of rotations.

    Args:
        num_gates (int): The number of random gates to add in the simulation.

    Returns:
        result (qutip.solver.Result): A qutip Result object obtained from any of the
                                      solver methods such as mesolve.
    """
    num_qubits = (
        2  # Qubit-0 is the target qubit. Qubit-1 suffers from crosstalk.
    )
    myprocessor = ModelProcessor(model=MyModel(num_qubits))
    # Add qubit frequency detuning 1.852MHz for the second qubit.
    myprocessor.add_drift(2 * np.pi * (sigmaz() + 1) / 2 * 1.852, targets=1)
    myprocessor.native_gates = None  # Remove the native gates
    mycompiler = MyCompiler(
        num_qubits, {"pulse_amplitude": 0.02, "duration": 25}
    )
    myprocessor.add_noise(ClassicalCrossTalk(1.0))

    # Define a randome circuit.
    gates_set = [
        ROT(arg_value=0),
        ROT(np.pi / 2),
        ROT(np.pi),
        ROT(np.pi / 2 * 3),
    ]
    circuit = QubitCircuit(num_qubits)
    for ind in np.random.randint(0, 4, num_gates):
        circuit.add_gate(gates_set[ind], targets=0)

    # Simulate the circuit.
    myprocessor.load_circuit(circuit, compiler=mycompiler)
    init_state = tensor(
        [Qobj([[init_fid, 0], [0, 0.025]]), Qobj([[init_fid, 0], [0, 0.025]])]
    )
    options = {"nsteps": 10000}  # increase the maximal allowed steps
    e_ops = [tensor([qeye(2), fock_dm(2)])]  # observable

    # compute results of the run using a solver of choice with custom options
    result = myprocessor.run_state(
        init_state, solver="mesolve", options=options, e_ops=e_ops
    )
    result = result.expect[0][-1]  # measured expectation value at the end
    return result


init_fid = 0.975

if __name__ == "__main__":
    num_sample = 2
    # num_sample = 1600
    fidelity = []
    fidelity_error = []
    num_gates_list = [250]
    # num_gates_list = [250, 500, 750, 1000, 1250, 1500]

    # The full simulation may take several hours
    # so we just choose num_sample=2 and num_gates=250 as a test
    for num_gates in num_gates_list:
        args = [num_gates] * num_sample
        with ProcessPoolExecutor() as executor:
            expect = list(executor.map(single_crosstalk_simulation, args))

        fidelity.append(np.mean(expect))
        fidelity_error.append(np.std(expect) / np.sqrt(num_sample))

    # Recorded result
    num_gates_list = [250, 500, 750, 1000, 1250, 1500]
    data_y = [
        0.9566768747558925,
        0.9388905075892828,
        0.9229470389282218,
        0.9075513000339529,
        0.8941659320508855,
        0.8756519016627652,
    ]

    data_y_error = [
        0.00042992029265330223,
        0.0008339882813741004,
        0.0012606632769758602,
        0.0014643550337816722,
        0.0017695604671714809,
        0.0020964978542167617,
    ]

    def rb_curve(x, a):
        return (1 / 2 + np.exp(-2 * a * x) / 2) * 0.975

    pos, cov = curve_fit(rb_curve, num_gates_list, data_y, p0=[0.001])

    xline = np.linspace(0, 1700, 200)
    yline = rb_curve(xline, *pos)

    fig, ax = plt.subplots(figsize=(LINEWIDTH, 0.65 * LINEWIDTH), dpi=200)
    ax.errorbar(
        num_gates_list,
        data_y,
        yerr=data_y_error,
        fmt=".",
        capsize=2,
        color="slategrey",
    )
    ax.plot(xline, yline, color="slategrey")
    ax.set_ylabel("Average fidelity")
    ax.set_xlabel(r"Number of $\pi$ rotations")
    ax.set_xlim((0, 1700))

    fig.tight_layout()
    fig.savefig("fig4_cross_talk.pdf")
    fig.show()
