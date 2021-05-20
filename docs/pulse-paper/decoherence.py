TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from quantum_plots import global_setup
    global_setup(fontsize = 10)
except:
    pass

import matplotlib.pyplot as plt

import numpy as np
from qutip import sigmaz, basis
from qutip_qip.operations import hadamard_transform
from qutip_qip.pulse import Pulse
from qutip_qip.device import LinearSpinChain
from qutip_qip.circuit import QubitCircuit

# define a circuit with only idling
t = 30.
circuit = QubitCircuit(1)
circuit.add_gate("IDLE", 0, arg_value=t)
# define a processor with t2 relaxation
f = 0.5
t2 = 10 / f
processor = LinearSpinChain(1, t2=t2)
processor.add_drift(
    2*np.pi*sigmaz()/2*f, targets=[0])
processor.load_circuit(circuit)
# Record the expectation value
plus_state = \
    (basis(2,1) + basis(2,0)).unit()
result = processor.run_state(
    init_state=plus_state,
    tlist = np.linspace(0., t, 1000),
    # observable
    e_ops=[plus_state*plus_state.dag()])

tlist = np.linspace(0., t, 1000)
fig, ax = plt.subplots(figsize = (LINEWIDTH, LINEWIDTH*0.65), dpi=200)
# detail about lenght of tlist needs to be fixed
ax.plot(tlist[:-1], result.expect[0][:-1], '-', label="Simulation", color="slategray")
ax.plot(tlist[:-1], np.exp(-1./t2 * tlist[:-1])*0.5 + 0.5, '--', label="Theory", color="slategray")
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel("Ramsey signal")
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig("figures/decoherence.pdf")
fig.show()