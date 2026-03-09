TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from quantum_plots import global_setup

    global_setup(fontsize=10)
except:
    pass
plt.rcParams.update({"text.usetex": False, "font.size": 10})
import numpy as np
import scipy
from qutip import sigmaz, basis, sigmax, fidelity
from qutip_qip.operations import hadamard_transform
from qutip_qip.pulse import Pulse
from qutip_qip.device import LinearSpinChain
from qutip_qip.circuit import QubitCircuit

pi = np.pi
num_samples = 500
amp = 0.1
f = 0.5
t2 = 10 / f

# Define a processor.
proc = LinearSpinChain(num_qubits=1, sx=amp / 2, t2=t2)
ham_idle = 2 * pi * sigmaz() / 2 * f
resonant_sx = 2 * pi * sigmax() - ham_idle / (amp / 2)
proc.add_drift(ham_idle, targets=0)
proc.add_control(resonant_sx, targets=0, label="sx0")


# Define a Ramsey experiment.
def ramsey(t, proc):
    qc = QubitCircuit(1)
    qc.add_gate("RX", targets=0, arg_value=pi / 2)
    qc.add_gate("IDLE", targets=0, arg_value=t)
    qc.add_gate("RX", targets=0, arg_value=pi / 2)
    proc.load_circuit(qc)
    result = proc.run_state(init_state=basis(2, 0), e_ops=sigmaz())
    return result.expect[0][-1]


idle_tlist = np.linspace(0.0, 30.0, num_samples)
measurements = np.asarray([ramsey(t, proc) for t in idle_tlist])

fig, ax = plt.subplots(figsize=(LINEWIDTH, LINEWIDTH * 0.60), dpi=200)

rx_gate_time = 1 / 4 / amp  # pi/2
total_time = 2 * rx_gate_time + idle_tlist[-1]

tlist = np.linspace(0.0, total_time, num_samples)
ax.plot(
    idle_tlist[:], measurements[:], "-", label="Simulation", color="slategray"
)

peak_ind = scipy.signal.find_peaks(measurements)[0]
decay_func = lambda t, t2, f0: f0 * np.exp(-1.0 / t2 * t)
(t2_fit, f0_fit), _ = scipy.optimize.curve_fit(
    decay_func, idle_tlist[peak_ind], measurements[peak_ind]
)
print("T2:", t2)
print("Fitted T2:", t2_fit)

ax.plot(
    idle_tlist,
    decay_func(idle_tlist, t2_fit, f0_fit),
    "--",
    label="Theory",
    color="slategray",
)
ax.set_xlabel(r"Idling time $t$ [$\mu$s]")
ax.set_ylabel("Ramsey signal", labelpad=2)
ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1]))
ax.set_position([0.18, 0.2, 0.75, 0.75])
ax.grid()

fig.savefig("fig5_decoherence.pdf")
fig.show()


circuit = QubitCircuit(1)
circuit.add_gate("RX", targets=0, arg_value=pi / 2)
circuit.add_gate("IDLE", targets=0, arg_value=15.0)
circuit.add_gate("RX", targets=0, arg_value=pi / 2)
proc.load_circuit(circuit)
fig2, axis = proc.plot_pulses(
    figsize=(LINEWIDTH, LINEWIDTH * 0.15), use_control_latex=False, dpi=200
)
axis[0].set_ylim((0.0 - axis[0].get_ylim()[1] * 0.05, axis[0].get_ylim()[1]))
axis[0].set_position([0.18, 0.39, 0.75, 0.60])
axis[0].set_ylabel("sx0", labelpad=25)
axis[0].yaxis.set_label_coords(-0.13, 0.25)
axis[0].set_xlabel("Ramsey pulse")
fig2.savefig("fig5_decoherence_pulse.pdf")
fig2.show()

# Test for time-dependent decoherence
from qutip_qip.noise import DecoherenceNoise
from qutip import sigmam

tlist = np.linspace(0, 30.0, 100)
coeff = tlist * 0.01
proc.add_noise(DecoherenceNoise(sigmam(), targets=0, coeff=coeff, tlist=tlist))
result = proc.run_state(init_state=basis(2, 0), e_ops=sigmaz())
