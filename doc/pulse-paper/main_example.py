import numpy as np
from qutip import basis
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import LinearSpinChain, SpinChainModel, Processor
from qutip_qip.noise import RelaxationNoise
qc = QubitCircuit(3)
qc.add_gate("X", targets=2)
qc.add_gate("SNOT", targets=0)
qc.add_gate("SNOT", targets=1)
qc.add_gate("SNOT", targets=2)

# Oracle function f(x)
qc.add_gate(
    "CNOT", controls=0, targets=2)
qc.add_gate(
    "CNOT", controls=1, targets=2)

qc.add_gate("SNOT", targets=0)
qc.add_gate("SNOT", targets=1)

init_state = basis([2,2,2], [0,0,0])
final_state = qc.run(init_state)

processor = LinearSpinChain(
    num_qubits=3, sx=0.25)

processor.load_circuit(qc)

tlist = np.linspace(0, 20, 300)
result = processor.run_state(
    init_state, tlist=tlist)

processor.add_noise(
    RelaxationNoise(t2=30))
    
# Section 4.1 Model
model = SpinChainModel(
    num_qubits=3, setup="circular", g=1)
processor = Processor(model=model)

model.get_control(label="sx0")

model.get_control_labels()

# Section 4.3 Scheduler
from qutip_qip.compiler import Scheduler, Instruction

Scheduler("ASAP").schedule(qc)

inst_list = []
for gate in qc.gates:
    if gate.name in ("SNOT", "X"):
        inst_list.append(
            Instruction(gate, duration=1
            )
        )
    else:
        inst_list.append(
            Instruction(gate, duration=2
            )
        )
Scheduler("ALAP").schedule(inst_list)