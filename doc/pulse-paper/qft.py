TEXTWIDTH = 5.93
LINEWIDTH = 3.22
import matplotlib.pyplot as plt

try:
    from quantum_plots import global_setup

    global_setup(fontsize=10)
except:
    pass
plt.rcParams.update({"text.usetex": False, "font.size": 10})


import numpy as np
from qutip import basis, fidelity
from qutip_qip.device import LinearSpinChain
from qutip_qip.algorithms import qft_gate_sequence

num_qubits = 10
# The QFT circuit
qc = qft_gate_sequence(num_qubits, swapping=False, to_cnot=True)
# Gate-level simulation
state1 = qc.run(basis([2] * num_qubits, [0] * num_qubits))
# Pulse-level simulation
processor = LinearSpinChain(num_qubits)
processor.load_circuit(qc)
options = {"max_step": 5000, "rtol": 1.0e-8}
state2 = processor.run_state(
    basis([2] * num_qubits, [0] * num_qubits), options=options
).states[-1]

assert abs(1 - fidelity(state1, state2)) < 1.0e-4


def get_control_latex(model):
    """
    Get the labels for each Hamiltonian.
    It is used in the method method :meth:`.Processor.plot_pulses`.
    It is a 2-d nested list, in the plot,
    a different color will be used for each sublist.
    """
    num_qubits = model.num_qubits
    num_coupling = model._get_num_coupling()
    return [
        {f"sx{m}": r"$\sigma_x^{}$".format(m) for m in range(num_qubits)},
        {f"sz{m}": r"$\sigma_z^{}$".format(m) for m in range(num_qubits)},
        {f"g{m}": r"$g_{}$".format(m) for m in range(num_coupling)},
    ]


fig, axes = processor.plot_pulses(
    figsize=(TEXTWIDTH, TEXTWIDTH),
    dpi=200,
    pulse_labels=get_control_latex(processor.model),
)
axes[-1].set_xlabel("$t$")
fig.tight_layout()
fig.savefig("qft_pulse.pdf")


import time

compiling_time = []
simulation_time = []
for num_qubits in range(1, 3):  # Test up to 3 qubits
    # for num_qubits in range(1, 11):
    compiling_time_sample = []
    simulation_time_sample = []
    for _ in range(20):
        qc = qft_gate_sequence(num_qubits, swapping=False, to_cnot=True)
        state1 = qc.run(basis([2] * num_qubits, [0] * num_qubits))
        processor = LinearSpinChain(num_qubits)
        start = time.time()
        processor.load_circuit(qc)
        end = time.time()
        compiling_time_sample.append(end - start)
        start = time.time()
        state2 = processor.run_state(
            basis([2] * num_qubits, [0] * num_qubits)
        ).states[-1]
        end = time.time()
        simulation_time_sample.append(end - start)
    compiling_time.append(np.average(compiling_time_sample))
    simulation_time.append(np.average(simulation_time_sample))

# Recorded results
compiling_time = [
    0.0012659549713134766,
    0.008200037479400634,
    0.028502225875854492,
    0.07625222206115723,
    0.14860657453536988,
    0.26267787218093874,
    0.4118484020233154,
    0.644087290763855,
    0.9500880241394043,
    1.287346065044403,
]
simulation_time = [
    0.007144343852996826,
    0.019202685356140135,
    0.04985215663909912,
    0.11975035667419434,
    0.28220993280410767,
    0.6953113317489624,
    1.624202561378479,
    4.255168056488037,
    11.539598500728607,
    31.34040207862854,
]

fig, ax = plt.subplots(figsize=(TEXTWIDTH, LINEWIDTH / 3 * 2), dpi=200)
ax.plot(
    range(1, 11),
    compiling_time,
    "-s",
    markersize=4,
    label=r"Compiler (\texttt{Processor.load\_circuit)}",
)
ax.plot(
    range(1, 11),
    simulation_time,
    "-D",
    markersize=4,
    label=r"Solver (\texttt{Processor.run\_state})",
)
ax.set_ylabel("Simulation time [s]")
ax.set_xlabel("Number of qubits")
ax.set_yscale("log")
ax.set_ylim((ax.get_ylim()[0], 150))
ax.legend()
fig.tight_layout()
fig.savefig("runtime.pdf")
plt.show()
