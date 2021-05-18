TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    global_setup(fontsize = 10)
except:
    pass

import numpy as np
from qutip import sigmax, sigmay, sigmaz, basis, qeye, tensor, Qobj, fock_dm
from qutip_qip.circuit import QubitCircuit, Gate
from qutip_qip.device import ModelProcessor
from qutip_qip.compiler import GateCompiler, Instruction

class MyProcessor(ModelProcessor):
    """Custom processor built using ModelProcessor as the base class."""
    def __init__(self, num_qubits, t1=None, t2=None):
        super().__init__(num_qubits, t1=t1, t2=t2)
        self.pulse_mode = "discrete"  # The control pulse is discrete or continous.
        self.set_up_ops(num_qubits)  # set up the available Hamiltonians
        self.dims = [2] * num_qubits  # The dimension of the quantum system
        self.num_qubits = num_qubits
        self.native_gates = ["RX", "RY"]

    def set_up_ops(self, num_qubits):
        """Sets up the single qubit control operators for each qubit."""
        for m in range(num_qubits):
            self.add_control(2 * np.pi * sigmax()/2, m, label="sx" + str(m))
        for m in range(num_qubits):
            self.add_control(2 * np.pi * sigmay()/2, m, label="sy" + str(m))

class MyCompiler(GateCompiler):
    """ Custom compiler for generating pulses from gates."""
    def __init__(self, num_qubits, params):
        super().__init__(num_qubits, params=params)
        self.params = params
        self.gate_compiler={"ROT": self.rotation_with_phase_compiler,
                            "RX": self.single_qubit_gate_compiler,
                            "RY": self.single_qubit_gate_compiler}

    def generate_pulse(self, gate, tlist, coeff, phase=0.):
        """Generates the pulses"""
        pulse_info = [("sx" + str(gate.targets[0]), np.cos(phase) * coeff),  
                      ("sy" + str(gate.targets[0]), np.sin(phase) * coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def single_qubit_gate_compiler(self, gate, args):
        """Compiles single qubit gates to pulses"""
        # gate.arg_value is the rotation angle
        tlist = np.abs(gate.arg_value) / self.params["pulse_amplitude"]
        coeff = self.params["pulse_amplitude"] * np.sign(gate.arg_value)
        if gate.name == "RX":
            return self.generate_pulse(gate, tlist, coeff, phase=0.)
        elif gate.name == "RY":
            return self.generate_pulse(gate, tlist, coeff, phase=np.pi/2)

    def rotation_with_phase_compiler(self, gate, args):
        """Compiles gates with a phase term."""
        # gate.arg_value is the pulse phase
        tlist = self.params["duration"]
        coeff = self.params["pulse_amplitude"]
        return self.generate_pulse(gate, tlist, coeff, phase=gate.arg_value)

# Define a circuit and run the simulation
num_qubits = 1
circuit = QubitCircuit(1)
circuit.add_gate("RX", targets=0, arg_value=np.pi/2)
circuit.add_gate("Z", targets=0)

myprocessor = MyProcessor(1)
mycompiler = MyCompiler(num_qubits, {"pulse_amplitude":0.02})
myprocessor.load_circuit(circuit, compiler=mycompiler)
result = myprocessor.run_state(basis(2,0))

fig, ax = myprocessor.plot_pulses(figsize=(LINEWIDTH*0.7,LINEWIDTH/2*0.7), dpi=200)
ax[-1].set_xlabel("Time")
fig.tight_layout()
fig.savefig("figures/customize.pdf")
fig.show()

from joblib import Parallel, delayed
from qutip import Options
from qutip_qip.noise import Noise

class ClassicalCrossTalk(Noise):
    def __init__(self, ratio):
        self.ratio = ratio

    def get_noisy_dynamics(
            self, dims=None, pulses=None, systematic_noise=None):
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
            if "sx" or "sy" not in pulse.label:
                pass  # filter out other pulses, e.g. drift
            target = pulse.targets[0]
            if target != 0:  # add pulse to the left neighbour
                pulses[i].add_control_noise(
                    self.ratio * pulse.qobj, targets=[target - 1],
                    coeff=pulse.coeff, tlist=pulse.tlist)
            if target != len(dims)-1:  # add pulse to the right neighbour
                pulses[i].add_control_noise(
                    self.ratio * pulse.qobj, targets=[target + 1],
                    coeff=pulse.coeff,tlist=pulse.tlist)
        return pulses, systematic_noise

def single_crosstalk_simulation(num_gates):
    """ A single simulation, with num_gates representing the number of rotations."""
    num_qubits = 2  # Qubit-0 is the target qubit. Qubit-1 suffers from crosstalk.
    myprocessor = MyProcessor(num_qubits)
    # Add qubit frequency detuning 1.852MHz for the second qubit.
    myprocessor.add_drift(2 * np.pi * (sigmaz() + 1) / 2 * 1.852, targets=1)
    myprocessor.native_gates = None  # Remove the native gates
    mycompiler = MyCompiler(num_qubits, {"pulse_amplitude":0.02, "duration":25})
    myprocessor.add_noise(ClassicalCrossTalk(1.))
    # Define a randome circuit.
    gates_set = [Gate("ROT", 0, arg_value=0),
                Gate("ROT", 0, arg_value=np.pi/2),
                Gate("ROT", 0, arg_value=np.pi),
                Gate("ROT", 0, arg_value=np.pi/2*3)]
    circuit = QubitCircuit(num_qubits)
    for ind in np.random.randint(0, 4, num_gates):
        circuit.add_gate(gates_set[ind])
    # Simulate the circuit.
    myprocessor.load_circuit(circuit, compiler=mycompiler)
    init_state = tensor(
        [Qobj([[init_fid, 0],[0, 0.025]]), Qobj([[init_fid, 0],[0, 0.025]])])
    options = Options(nsteps=10000)  # increase the maximal allowed steps
    e_ops = [tensor([qeye(2), fock_dm(2)])]  # observable
    result = myprocessor.run_state(init_state, options=options, e_ops=e_ops)
    result = result.expect[0][-1]  # measured expectation value at the end
    return result

num_qubits = 2
num_sample = 1600
fidelity = []
fidelity_error = []
init_fid = 0.975
num_gates_list = [250, 500, 750, 1000, 1250, 1500]

# # The following code may take several hours
# for num_gates in num_gates_list:
#     expect = Parallel(n_jobs=8)(delayed(single_crosstalk_simulation)(num_gates) for i in range(num_sample))
#     fidelity.append(np.mean(expect))
#     fidelity_error.append(np.std(expect)/np.sqrt(num_sample))

# plotting
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Recorded result
data_y = [0.9577285560461476, 0.9384849070716464, 0.9230217713086177, 0.9062344084919285, 0.889009550855518, 0.8749290612064392]
data_y_error = [0.000431399017208067, 0.0008622091914303468, 0.0012216267555118497, 0.001537120153687202, 0.0018528957172559654, 0.0020169257334183596]
def linear(x, a):
    return (1/2 + np.exp(-2 * a * x)/2) * 0.975
pos, cov = curve_fit(linear, num_gates_list, data_y, p0=[0.001])
xline = np.linspace(0, 1700, 200)
yline = linear(xline, *pos)
fig, ax = plt.subplots(figsize = (LINEWIDTH, 0.65 * LINEWIDTH), dpi=200)
ax.errorbar(num_gates_list, data_y, yerr=data_y_error, fmt=".", capsize=2, color="slategrey")
ax.plot(xline, yline, color="slategrey")
ax.set_ylabel("Average fidelity")
ax.set_xlabel(r"Number of $\pi$ rotations")
ax.set_xlim((0, 1700))
fig.tight_layout()
fig.savefig("figures\cross_talk.pdf")
fig.show()
