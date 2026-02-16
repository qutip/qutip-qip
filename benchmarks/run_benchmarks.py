import time
from qutip import basis, tensor
from .circuits import ghz_circuit, random_layered_circuit


def zero_state(n_qubits: int):
    """Return |0...0> initial state."""
    return tensor([basis(2, 0) for _ in range(n_qubits)])


def benchmark(circuit_fn, circuit_params: dict, runs: int = 20):
    """
    Benchmark average execution time for a circuit.

    Parameters
    ----------
    circuit_fn : callable
        Function that builds the circuit.
    circuit_params : dict
        Dictionary containing circuit inputs (e.g. n_qubits, depth, etc.).
    runs : int
        Number of timing runs.
    """
    qc = circuit_fn(**circuit_params)

    n_qubits = circuit_params.get("n_qubits")
    if n_qubits is None:
        raise ValueError("circuit_params must include 'n_qubits'.")

    # Warmup run to avoid first-call overhead distortion
    qc.run(zero_state(n_qubits))

    total_time = 0.0

    for _ in range(runs):
        state = zero_state(n_qubits)
        start = time.perf_counter()
        qc.run(state)
        end = time.perf_counter()
        total_time += end - start

    return total_time / runs


if __name__ == "__main__":
    print("Benchmarking circuit simulation performance\n")

    for n in [3, 4, 5, 6, 7, 8]:
        ghz_time = benchmark(ghz_circuit, {"n_qubits": n})
        rand_time = benchmark(random_layered_circuit, {"n_qubits": n})

        print(f"{n} qubits:")
        print(f"  GHZ circuit: {ghz_time:.6f} sec")
        print(f"  Random layered circuit: {rand_time:.6f} sec")
        print()
