from qutip_qip.qasm import read_qasm
from qutip_qip.qasm import save_qasm

qc = read_qasm("deutsch-jozsa.qasm")
save_qasm(qc, "deutsch-jozsa-qutip.qasm")
