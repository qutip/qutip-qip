from qutip_qip.decompose.decompose_single_qubit_gate import decompose_one_qubit_gate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit import gate_sequence_product

__all__ = ["optimise"]

def optimise(circuit):
    optimized_gates = []
    i = 0
    while i < len(circuit.gates):
        gate = circuit.gates[i]
        if gate.targets and len(gate.targets) == 1 and not gate.controls:
            qubit = gate.targets[0]
            sq_gates = []
            # Gather all consecutive single-qubit gates on the same qubit
            while i < len(circuit.gates):
                g = circuit.gates[i]
                if g.targets == [qubit] and not g.controls:
                    sq_gates.append(g)
                    i += 1
                else:
                    break
            if len(sq_gates) == 1:
                optimized_gates.append(sq_gates[0])
            else:
                try:                    
                    # Get all gate unitaries for the sequence
                    unitaries = [g.get_unitary() for g in sq_gates]
                    
                    # Combine them into a single unitary
                    u_combined = gate_sequence_product(unitaries)
                    
                    # Use the decompose_one_qubit_gate function to decompose into ZYZ form
                    
                    # Decompose the combined unitary into ZYZ gates
                    zyz_gates = decompose_one_qubit_gate(u_combined, decomposition="ZYZ", target=qubit)
                    
                    # Add decomposed gates to the optimized circuit
                    optimized_gates.extend(zyz_gates)
                except Exception as e:
                    print(f"Decomposition failed: {e}")
                    # Fallback to original gates if decomposition fails
                    optimized_gates.extend(sq_gates)
        else:
            optimized_gates.append(gate)
            i += 1
    
    new_circuit = QubitCircuit(N=circuit.N)
    new_circuit.gates = optimized_gates
    return new_circuit

def run_example():
    """Run a complete example of quantum circuit optimization."""
    print("=== Quantum Circuit Optimization Example ===")
    
    # Example 1: A circuit with 4 single-qubit gates
    print("\nExample 1: Four consecutive single-qubit gates")
    qc1 = QubitCircuit(N=1)
    qc1.add_gate("RX", targets=[0], arg_value=0.2)
    qc1.add_gate("RY", targets=[0], arg_value=0.1)
    qc1.add_gate("RY", targets=[0], arg_value=0.1)
    qc1.add_gate("RZ", targets=[0], arg_value=0.3)
    
    QubitCircuit.draw(qc1)    
    optimized_qc1 = optimise(qc1)
    QubitCircuit.draw(optimized_qc1)
    
    # Example 2: A circuit with mixed single and multi-qubit gates
    print("\n\nExample 2: Mixed single and multi-qubit gates")
    qc2 = QubitCircuit(N=2)
    qc2.add_gate("RX", targets=[0], arg_value=0.2)
    qc2.add_gate("RY", targets=[0], arg_value=0.3)
    qc2.add_gate("CNOT", controls=[0], targets=[1])
    qc2.add_gate("RZ", targets=[1], arg_value=0.1)
    qc2.add_gate("RX", targets=[1], arg_value=0.4)
    
    QubitCircuit.draw(qc2)    
    optimized_qc2 = optimise(qc2)
    QubitCircuit.draw(optimized_qc2)


run_example()
