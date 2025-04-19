import numpy as np
from qutip import Qobj
from qutip_qip.operations import Gate, ControlledGate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.decompose import decompose_one_qubit_gate

__all__ = ["qpe", "CustomGate"]


class CustomGate(Gate):
    """
    Custom gate that wraps an arbitrary quantum operator.
    """
    def __init__(self, targets, U, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.targets = targets if isinstance(targets, list) else [targets]
        self.U = U  
        self.kwargs = kwargs
        self.latex_str = r"U"  
        
    def get_compact_qobj(self):
        return self.U


def create_controlled_unitary(controls, targets, U, control_value=1):
    """
    Create a controlled unitary gate.
    
    Parameters
    ----------
    controls : list
        Control qubits
    targets : list
        Target qubits
    U : Qobj
        Unitary operator to apply on target qubits
    control_value : int, optional
        Control value (default: 1)
        
    Returns
    -------
    ControlledGate
        The controlled unitary gate
    """
    return ControlledGate(
        controls=controls,
        targets=targets,
        control_value=control_value,
        target_gate=CustomGate,
        U=U
    )


def _cphase_to_cnot(targets, controls, arg_value):
    """
    Decompose a controlled phase gate into CNOTs and single-qubit gates.
    
    Parameters
    ----------
    targets : list
        Target qubits
    controls : list
        Control qubits
    arg_value : float
        Phase angle
        
    Returns
    -------
    list
        List of decomposed gates
    """
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


def inverse_qft_gate_sequence(N=1, swapping=True, to_cnot=False):
    """
    Create the inverse quantum Fourier transform gate sequence.
    
    Parameters
    ----------
    N : int, optional
        Number of qubits (default: 1)
    swapping : bool, optional
        Flag to include SWAP gates at the beginning (default: True)
    to_cnot : bool, optional
        Flag to decompose controlled phase gates to CNOT gates (default: False)
        
    Returns
    -------
    QubitCircuit
        Gate sequence implementing inverse QFT
    """
    if N < 1:
        raise ValueError("Minimum value of N can be 1")
    
    qc = QubitCircuit(N)
    
    if N == 1:
        qc.add_gate("SNOT", targets=[0])
    else:
        if swapping:
            for i in range(N // 2):
                qc.add_gate("SWAP", targets=[N - i - 1, i])
        
        # Inverse QFT main algorithm
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
    target_qubits : int or list, optional
        Index or indices of the target qubit(s) where the eigenstate is prepared.
        If None, target_qubits is set automatically based on U's dimension.
    to_cnot : bool, optional
        Flag to decompose controlled phase gates to CNOT gates (default: False)
        
    Returns
    -------
    qc : instance of QubitCircuit
        Gate sequence implementing Quantum Phase Estimation.
    """
    if num_counting_qubits < 1:
        raise ValueError("Minimum value of counting qubits must be 1")
    
    # Handle target qubits specification
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
    
    # Apply Hadamard gates to all counting qubits
    for i in range(num_counting_qubits):
        qc.add_gate("SNOT", targets=[i])
    
    # Apply controlled-U gates with increasing powers
    for i in range(num_counting_qubits):
        power = 2**(num_counting_qubits - i - 1)
        # Create U^power
        U_power = U if power == 1 else U ** power
        
        # Add controlled-U^power gate
        controlled_u = create_controlled_unitary(controls=[i], targets=target_qubits, U=U_power)
        qc.add_gate(controlled_u)
    
    # Add inverse QFT on counting qubits
    inverse_qft_circuit = inverse_qft_gate_sequence(num_counting_qubits, swapping=True, to_cnot=to_cnot)
    for gate in inverse_qft_circuit.gates:
        qc.add_gate(gate)
            
    return qc