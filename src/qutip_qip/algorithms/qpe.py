import numpy as np
from ..operations import Gate, snot, cphase, expand_operator
from ..circuit import QubitCircuit
from qutip import Qobj
from ..decompose import decompose_one_qubit_gate

__all__ = ["qpe"]

def qpe(U, num_counting_qubits, target_qubits=None, to_cnot=False):
    """
    Quantum Phase Estimation circuit implementation for QuTiP.
    
    Parameters
    ----------
    U : Qobj
        Unitary operator whose eigenvalue we want to estimate.
        Should be a unitary quantum operator.
    num_counting_qubits : int
        Number of counting qubits to use for the phase estimation.
        More qubits provide higher precision.
    target_qubits : int, optional
        Index of the target qubit where the eigenstate is prepared.
        If None, target_qubits is set to num_counting_qubits.
    measurement : bool, optional
        Flag to include measurement gates on counting qubits at the end.
    to_cnot : bool, optional
        Flag to decompose controlled phase gates to CNOT gates.
        
    Returns
    -------
    qc : instance of QubitCircuit
        Gate sequence implementing Quantum Phase Estimation.
    """
    if num_counting_qubits < 1:
        raise ValueError("Minimum value of counting qubits must be 1")
    
    if target_qubits is None:
        dim = U.dims[0][0]  
        num_target_qubits = int(np.log2(dim))
        if 2**num_target_qubits != dim:
            raise ValueError(f"Unitary operator dimension {dim} is not a power of 2")
        target_qubits = list(range(num_counting_qubits, num_counting_qubits + num_target_qubits))
    elif isinstance(target_qubits, int):
        target_qubits = [target_qubits]
        num_target_qubits = 1
    else:
        num_target_qubits = len(target_qubits)
    
    total_qubits = num_counting_qubits + num_target_qubits

    qc = QubitCircuit(total_qubits)
    
    for i in range(num_counting_qubits):
        qc.add_gate("SNOT", targets=[i])
    
    for i in range(num_counting_qubits):
        power = 2**(num_counting_qubits - i - 1)
        
        qc.add_gate("CPHASE", controls=[i], targets=list(range(num_counting_qubits, total_qubits)), 
                   arg_label=f"U^{power}", arg_value=power)
    
    inverse_qft = inverse_qft_gate_sequence(num_counting_qubits, swapping=True, to_cnot=to_cnot)
    
    for gate in inverse_qft.gates:
        qc.add_gate(gate)
            
    return qc

def inverse_qft_gate_sequence(N=1, swapping=True, to_cnot=False):
    
    if N < 1:
        raise ValueError("Minimum value of N can be 1")
    
    qc = QubitCircuit(N)
    
    if N == 1:
        qc.add_gate("SNOT", targets=[0])
    else:
        if swapping:
            for i in range(N // 2):
                qc.add_gate("SWAP", targets=[N - i - 1, i])
        
        for i in range(N - 1, -1, -1):
            qc.add_gate("SNOT", targets=[i])
            
            for j in range(i - 1, -1, -1):
                if not to_cnot:
                    qc.add_gate(
                        "CPHASE",
                        targets=[j],
                        controls=[i],
                        arg_label=r"{-\pi/2^{%d}}" % (i - j),
                        arg_value=-np.pi / (2 ** (i - j)),
                    )
                else:
                    decomposed_gates = _cphase_to_cnot(
                        [j], [i], -np.pi / (2 ** (i - j))
                    )
                    qc.gates.extend(decomposed_gates)
    
    return qc

def _cphase_to_cnot(targets, controls, arg_value):
    rotation = Qobj([[1.0, 0.0], [0.0, np.exp(1.0j * arg_value)]])
    decomposed_gates = list(
        decompose_one_qubit_gate(rotation, method="ZYZ_PauliX")
    )
    new_gates = []
    gate = decomposed_gates[0]
    gate.targets = targets
    new_gates.append(gate)
    new_gates.append(Gate("CNOT", targets=targets, controls=controls))
    gate = decomposed_gates[4]
    gate.targets = targets
    new_gates.append(gate)
    new_gates.append(Gate("CNOT", targets=targets, controls=controls))
    new_gates.append(Gate("RZ", targets=controls, arg_value=arg_value / 2))
    gate = decomposed_gates[7]
    gate.arg_value += arg_value / 4
    new_gates.append(gate)
    return new_gates

