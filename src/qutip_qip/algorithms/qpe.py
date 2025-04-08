"""
This module provides the circuit implementation for Quantum Phase Estimation.
"""

import numpy as np
from ..operations import Gate, snot, cphase, expand_operator
from ..circuit import QubitCircuit
from qutip import Qobj
from .qft import qft_gate_sequence


__all__ = ["phase_estimation"]


def phase_estimation(unitary, num_ancilla=3, target_qubit=None):
    """
    Creates a circuit that performs quantum phase estimation on a unitary gate.
    
    The circuit estimates the phase θ in e^(2πiθ) of an eigenvalue of the unitary
    operator. It requires ancilla qubits to store the estimated phase and target
    qubits that should be prepared in the eigenstate of the unitary operator.
    
    Parameters
    ----------
    unitary : Qobj or Gate
        Unitary operator whose phase is being estimated.
        If a Gate is provided, the gate's targets parameter specifies which qubits
        it operates on.
    num_ancilla : int
        Number of ancilla qubits used for phase estimation. This determines the
        precision of the phase estimate.
    target_qubit : int or list, optional
        Index of the target qubit(s) for the unitary operation. If not specified,
        it is assumed to be the qubit immediately following the ancilla qubits.
        
    Returns
    -------
    qc : QubitCircuit
        Quantum circuit implementing the phase estimation algorithm.
    
    Notes
    -----
    The precision of the phase estimation is 2^(-num_ancilla).
    The circuit follows these steps:
    1. Apply Hadamard gates to all ancilla qubits
    2. Apply controlled-U^(2^j) operations
    3. Apply inverse QFT to the ancilla qubits
    4. Measure the ancilla qubits (optional, measurement not included)
    """
    # Determine the target qubit if not specified
    if target_qubit is None:
        target_qubit = num_ancilla
    
    # If target_qubit is a single integer, convert to list
    if isinstance(target_qubit, int):
        target_qubit = [target_qubit]
    
    # Calculate total number of qubits needed
    max_target = max(target_qubit)
    total_qubits = max(num_ancilla, max_target + 1)
    
    # Create quantum circuit
    qc = QubitCircuit(total_qubits)
    
    # Step 1: Apply Hadamard gates to all ancilla qubits
    for i in range(num_ancilla):
        qc.add_gate("SNOT", targets=[i])
    
    # Step 2: Apply controlled-U^(2^j) operations
    # For each ancilla qubit, we apply controlled-U^(2^j) operations
    for i in range(num_ancilla):
        # Calculate U^(2^i)
        power = 2**(num_ancilla - i - 1)
        
        # Add controlled unitary gate
        if isinstance(unitary, Gate):
            # If unitary is already a Gate object, create a controlled version
            controlled_gate = unitary.copy()
            controlled_gate.controls = [i]
            
            # Apply the gate power times
            for _ in range(power):
                qc.add_gate(controlled_gate)
        else:
            # If unitary is a Qobj, create a controlled-U gate
            for _ in range(power):
                qc.add_gate("CPHASE", targets=target_qubit, 
                            controls=[i], arg_value=unitary)
    
    # Step 3: Apply inverse QFT to the ancilla qubits
    # First, get the QFT circuit
    qft_circuit = qft_gate_sequence(num_ancilla)
    
    # Reverse the order of gates to get inverse QFT and add them to the circuit
    for gate in reversed(qft_circuit.gates):
        # Adjust CPHASE gates to have negative phase for inverse QFT
        if gate.name == "CPHASE":
            gate.arg_value = -gate.arg_value
        qc.add_gate(gate)
    
    return qc


def phase_estimation_example():
    """
    Creates an example circuit for phase estimation of a simple phase rotation.
    
    This function creates a circuit that estimates the phase of a simple phase
    rotation gate (RZ gate) with a known phase of π/4. This serves as a simple
    demonstration of the phase estimation algorithm.
    
    Returns
    -------
    qc : QubitCircuit
        Example quantum circuit implementing phase estimation for a rotation gate.
    """
    # Create a simple phase rotation gate with phase π/4
    phase_gate = Gate("RZ", targets=[3], arg_value=np.pi/4)
    
    # Create phase estimation circuit with 3 ancilla qubits
    qc = phase_estimation(phase_gate, num_ancilla=3, target_qubit=3)
    
    return qc
