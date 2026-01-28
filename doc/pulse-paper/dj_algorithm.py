TEXTWIDTH = 5.93
LINEWIDTH = 3.22
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from quantum_plots import global_setup

    global_setup(fontsize=10)
except:
    pass
plt.rcParams.update({"text.usetex": False, "font.size": 10})


import numpy as np
import matplotlib.pyplot as plt
from qutip_qip.device import (
    OptPulseProcessor,
    LinearSpinChain,
    SCQubits,
    SpinChainModel,
)
from qutip_qip.circuit import QubitCircuit
from qutip import basis

num_qubits = 3
np.random.seed(2)

# Deutsch-Josza algorithm
dj_circuit = QubitCircuit(num_qubits)
dj_circuit.add_gate("X", targets=2)
dj_circuit.add_gate("SNOT", targets=0)
dj_circuit.add_gate("SNOT", targets=1)
dj_circuit.add_gate("SNOT", targets=2)

# Oracle function f(x)
dj_circuit.add_gate("CNOT", controls=0, targets=2)
dj_circuit.add_gate("CNOT", controls=1, targets=2)

dj_circuit.add_gate("SNOT", targets=0)
dj_circuit.add_gate("SNOT", targets=1)

# Spin chain model
spinchain_processor = LinearSpinChain(num_qubits=num_qubits, t2=30)  # T2 = 30
spinchain_processor.load_circuit(dj_circuit)
initial_state = basis([2, 2, 2], [0, 0, 0])  # 3 qubits in the 000 state
t_record = np.linspace(0, 20, 300)
result1 = spinchain_processor.run_state(initial_state, tlist=t_record)

# Superconducting qubits
scqubits_processor = SCQubits(num_qubits=num_qubits)
scqubits_processor.load_circuit(dj_circuit)
initial_state = basis([3, 3, 3], [0, 0, 0])  #  3-level
result2 = scqubits_processor.run_state(initial_state)

# Optimal control model
setting_args = {
    "SNOT": {"num_tslots": 6, "evo_time": 2},
    "X": {"num_tslots": 1, "evo_time": 0.5},
    "CNOT": {"num_tslots": 12, "evo_time": 5},
}
opt_processor = OptPulseProcessor(
    num_qubits=num_qubits, model=SpinChainModel(3, setup="linear")
)
opt_processor.load_circuit(  # Provide parameters for the algorithm
    dj_circuit,
    setting_args=setting_args,
    merge_gates=False,
    verbose=True,
    amp_ubound=5,
    amp_lbound=0,
)
initial_state = basis([2, 2, 2], [0, 0, 0])
result3 = opt_processor.run_state(initial_state)

# For plotting
width = TEXTWIDTH / 3
fig, ax = opt_processor.plot_pulses(figsize=(width, width * 3 / 2.9), dpi=200)

ax[0].set_ylabel(r"$\Omega^x_{0}$")
ax[1].set_ylabel(r"$\Omega^x_{1}$")
ax[2].set_ylabel(r"$\Omega^x_{2}$")
ax[3].set_ylabel(r"$\Omega^z_{0}$")
ax[4].set_ylabel(r"$\Omega^z_{1}$")
ax[5].set_ylabel(r"$\Omega^z_{2}$")
ax[6].set_ylabel(r"$g_{0}$")
ax[7].set_ylabel(r"$g_{1}$")
ax[7].set_xlabel(r"$t$")

full_tlist = opt_processor.get_full_tlist()
two_qubit_gate_region = [[0, 13, -12]] * 8
for i, (point1, point2, point3) in enumerate(two_qubit_gate_region):
    vmin, vmax = ax[i].get_ylim()
    ax[i].fill_between(
        [full_tlist[point2], full_tlist[point3]],
        [vmin, vmin],
        [vmax, vmax],
        color="lightgray",
        alpha=0.5,
    )
    ax[i].vlines(
        [full_tlist[point2], full_tlist[point3]],
        vmin,
        vmax,
        "gray",
        "--",
        linewidth=0.8,
        alpha=0.5,
    )

fig.savefig("optimal_control_pulse.pdf")
fig.show()

width = TEXTWIDTH / 3
fig2, ax2 = spinchain_processor.plot_pulses(
    figsize=(width, width * 3 / 2.9), dpi=200
)
fig2.tight_layout()
ax2[0].set_ylabel(r"$\Omega^x_{0}$")
ax2[1].set_ylabel(r"$\Omega^x_{1}$")
ax2[2].set_ylabel(r"$\Omega^x_{2}$")
ax2[3].set_ylabel(r"$\Omega^z_{0}$")
ax2[4].set_ylabel(r"$\Omega^z_{1}$")
ax2[5].set_ylabel(r"$\Omega^z_{2}$")
ax2[6].set_ylabel(r"$g_{0}$")
ax2[7].set_ylabel(r"$g_{1}$")
ax2[7].set_xlabel(r"$t$")

full_tlist = spinchain_processor.get_full_tlist()
two_qubit_gate_region = [
    [0, 7, -15],
    [0, 7, -6],
    [0, 8, -2],
    [0, 7, -15],
    [0, 7, -6],
    [0, 8, -2],
    [0, 7, -6],
    [0, 7, -1],
]
for i, (point1, point2, point3) in enumerate(two_qubit_gate_region):
    vmin, vmax = ax2[i].get_ylim()
    ax2[i].fill_between(
        [full_tlist[point2], full_tlist[point3]],
        [vmin, vmin],
        [vmax, vmax],
        color="lightgray",
        alpha=0.5,
    )
    ax2[i].vlines(
        [full_tlist[point2], full_tlist[point3]],
        vmin,
        vmax,
        "gray",
        "--",
        linewidth=0.8,
        alpha=0.5,
    )

fig2.tight_layout()
fig2.savefig("spin_chain_pulse.pdf")
fig2.show()

width = TEXTWIDTH / 3
# fig3, ax3 = scqubits_processor.plot_pulses(figsize=(width, width*3/2.43), dpi=200);
fig3, ax3 = scqubits_processor.plot_pulses(
    figsize=(width, width * 3 / 2.4), dpi=200
)
ax3[0].set_ylabel(r"$\Omega^x_{0}$")
ax3[1].set_ylabel(r"$\Omega^x_{1}$")
ax3[2].set_ylabel(r"$\Omega^x_{2}$")
ax3[3].set_ylabel(r"$\Omega^y_{0}$")
ax3[4].set_ylabel(r"$\Omega^y_{1}$")
ax3[5].set_ylabel(r"$\Omega^y_{2}$")
ax3[6].set_ylabel(r"$\Omega^{\rm{cr}1}_{0}$")
ax3[7].set_ylabel(r"$\Omega^{\rm{cr}2}_{0}$")
ax3[8].set_ylabel(r"$\Omega^{\rm{cr}1}_{1}$")
ax3[9].set_ylabel(r"$\Omega^{\rm{cr}2}_{1}$")
ax3[9].set_xlabel(r"$t$")

full_tlist = scqubits_processor.get_full_tlist()

fig3.savefig("transmon_pulse.pdf")
fig3.show()


plt.rcParams.update({"text.usetex": False, "font.size": 9})
# Plot hinton
from qutip import hinton

fig4, ax4 = plt.subplots(figsize=(LINEWIDTH * 0.9, LINEWIDTH * 0.7), dpi=200)
first_two_qubits = result1.states[-1].ptrace([0, 1])
_, ax4 = hinton(
    first_two_qubits,
    ax=ax4,
    x_basis=[
        "$|00\\rangle$",
        "$|01\\rangle$",
        "$|10\\rangle$",
        "$|11\\rangle$",
    ],
    y_basis=[
        "$|00\\rangle$",
        "$|01\\rangle$",
        "$|10\\rangle$",
        "$|11\\rangle$",
    ],
)
fig4.savefig("hinton.pdf")
fig4.show()

# Plot trajectory
expect = []
for state in result1.states:
    tmp = state.ptrace([0, 1])
    tmp = basis([2, 2], [0, 0]).dag() * tmp * basis([2, 2], [0, 0])
    expect.append(np.real(tmp))

fig5, ax5 = plt.subplots(figsize=(LINEWIDTH, LINEWIDTH * 0.7), dpi=200)
ax5.plot(t_record, expect, color="slategrey")
ax5.set_ylabel(r"Population of $|00\rangle$")
ax5.set_xlabel(r"Time [$\mu$s]")
fig5.tight_layout()
fig5.savefig("population.pdf")
fig5.show()
