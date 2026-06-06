import numpy as np
from qutip import basis, qeye
from qutip_qip.operations.gates import OracleGate, CNOT
from qutip_qip.circuit import QubitCircuit


def test_oracle_identity():
    qc = QubitCircuit(num_qubits=3)
    logic_func = lambda x: 0
    my_oracle = OracleGate(num_qubits=3, logic_func=logic_func, num_target_qubits=1)
    qc.add_gate(my_oracle, targets=[0, 1, 2])
    U = qc.compute_unitary()

    expected = qeye([2, 2, 2]).full()
    # Check if your oracle is equal to a standard CNOT
    assert np.allclose(U.full(), expected), "Test IDENTITY: Failed!"
    print("Test IDENTITY: PASSED")


def test_oracle_not():
    qc = QubitCircuit(num_qubits=2)
    logic_func = lambda x: x
    my_oracle = OracleGate(num_qubits=2, logic_func=logic_func, num_target_qubits=1)
    qc.add_gate(my_oracle, targets=[0, 1])
    U = qc.compute_unitary()

    expected = CNOT.get_qobj().full()
    # Check if your oracle is equal to a standard CNOT
    assert np.allclose(U.full(), expected), "Test NOT: Failed!"
    print("Test NOT: PASSED")


def test_oracle_multi_target():
    qc = QubitCircuit(num_qubits=3)
    logic_func = lambda x: 3
    my_oracle = OracleGate(num_qubits=3, logic_func=logic_func, num_target_qubits=2)
    qc.add_gate(my_oracle, targets=[0, 1, 2])
    U = qc.compute_unitary()
    print(U.full())


def test_unitarity():
    qc = QubitCircuit(5)
    func = lambda x: (x * 3 + 1) % 2  # random
    oracle_class = OracleGate(num_qubits=5, logic_func=func, num_target_qubits=1)
    qc.add_gate(oracle_class, targets=range(5))
    U = qc.compute_unitary()

    # U_dagger * U should be Identity
    identity_check = U.dag() * U
    expected_id = qeye([2] * 5)
    assert (identity_check - expected_id).norm() < 1e-12
    print("Unitarity Stress Test: PASSED")


def my_logic(x):
    if x % 2 == 0:
        return 1
    return 0


def test_named_function():
    oracle_class = OracleGate(num_qubits=2, logic_func=my_logic)
    # Check serialization
    print(f"Captured Source:\n{oracle_class._source}")
    assert "def my_logic" in oracle_class._source
    print("Named Function Serialization: PASSED")


def test_serialization():
    gate_class = OracleGate(num_qubits=2, logic_func=lambda x: x)
    assert gate_class._source is not None
    print("Serialization Metadata: PASSED")


def test_grover_oracle():
    def mark_3(x):
        return 1 if x == 3 else 0

    qc = QubitCircuit(num_qubits=3)
    my_oracle = OracleGate(num_qubits=3, logic_func=mark_3, num_target_qubits=1)
    qc.add_gate(my_oracle, targets=[0, 1, 2])
    U = qc.compute_unitary()

    k6 = basis(8, 6)
    k6.dims = [[2, 2, 2], [1, 1, 1]]

    k7 = basis(8, 7)
    k7.dims = [[2, 2, 2], [1, 1, 1]]

    k4 = basis(8, 4)
    k4.dims = [[2, 2, 2], [1, 1, 1]]

    product = U * k6
    assert np.allclose(product.full(), k7.full()), "Test 1 Grover: Failed!"
    print("Test 1 Grover: Passed!")
    product = U * k4
    assert np.allclose(product.full(), k4.full()), "Test 2 Grover: Failed!"
    print("Test 2 Grover: Passed!")
