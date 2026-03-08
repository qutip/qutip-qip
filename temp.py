import numpy as np
import jax
import jax.numpy as jnp
import time
import string
import matplotlib.pyplot as plt
from functools import partial

# ---------------------------------------------------------
# 1. Helper: Your Proposed String Generation
# ---------------------------------------------------------
def generate_einsum_eq(targets, num_qubits):
    l_chars = string.ascii_lowercase
    u_chars = string.ascii_uppercase
    
    state_in = list(l_chars[:num_qubits])
    gate_out = list(u_chars[:len(targets)])
    gate_in = [state_in[t] for t in targets]
    
    state_out = state_in.copy()
    for i, t in enumerate(targets):
        state_out[t] = gate_out[i]
        
    gate_str = "".join(gate_out + gate_in)
    state_in_str = "".join(state_in)
    state_out_str = "".join(state_out)
    
    return f"{gate_str},{state_in_str}->{state_out_str}"

# ---------------------------------------------------------
# 2. Old Method: List-based np.einsum (Current QuTiP)
# ---------------------------------------------------------
def old_evolve_state(gate_array, targets_indices, state, num_qubits):
    ancillary_indices = range(num_qubits, num_qubits + len(targets_indices))
    index_list = range(num_qubits)
    new_index_list = list(index_list)
    for j, k in enumerate(targets_indices):
        new_index_list[k] = j + num_qubits

    # Mimicking the current _evolve_state_einsum execution
    return np.einsum(
        gate_array,
        list(ancillary_indices) + list(targets_indices),
        state,
        list(index_list),
        new_index_list,
    )

# ---------------------------------------------------------
# 3. New Method: String-based JAX with JIT
# ---------------------------------------------------------
# We must declare 'eq' as a static argument because JAX cannot trace Python strings
@partial(jax.jit, static_argnames=['eq'])
def new_evolve_state_jax(eq, gate_array, state):
    return jnp.einsum(eq, gate_array, state)

# ---------------------------------------------------------
# Benchmarking Loop
# ---------------------------------------------------------
def run_benchmark():
    qubit_counts = range(10, 24) # Testing from 10 up to 23 qubits
    target_qubit = 5             # Arbitrary target qubit in the middle
    
    times_old_np = []
    times_new_jax_compile = []
    times_new_jax_run = []

    print("Starting Benchmark...")
    print(f"{'Qubits':<8} | {'Old NumPy':<12} | {'JAX (Compile + Run)':<22} | {'JAX (Run Only)':<15}")
    print("-" * 65)

    for n in qubit_counts:
        # Create dummy state tensor (shape: 2, 2, ..., 2)
        state_np = np.random.rand(*(2,) * n) + 1j * np.random.rand(*(2,) * n)
        state_jax = jnp.array(state_np)
        
        # Create dummy 1-qubit gate (shape: 2, 2)
        gate_np = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
        gate_jax = jnp.array(gate_np)
        
        # Target list
        targets = [target_qubit]
        
        # 1. Time Old NumPy Method
        start = time.perf_counter()
        _ = old_evolve_state(gate_np, targets, state_np, n)
        times_old_np.append(time.perf_counter() - start)
        
        # Generate equation string for new method
        eq = generate_einsum_eq(targets, n)
        
        # 2. Time JAX (First Run = Compilation + Execution)
        start = time.perf_counter()
        res_jax = new_evolve_state_jax(eq, gate_jax, state_jax)
        res_jax.block_until_ready() # Force JAX to finish async execution
        times_new_jax_compile.append(time.perf_counter() - start)
        
        # 3. Time JAX (Subsequent Run = Execution Only)
        start = time.perf_counter()
        res_jax = new_evolve_state_jax(eq, gate_jax, state_jax)
        res_jax.block_until_ready()
        times_new_jax_run.append(time.perf_counter() - start)
        
        # print(f"{n:<8} | {times_old_np[-1]:.6f} s   | {times_new_jax_compile[-1]:.6f} s            | {times_new_jax_run[-1]:.6f} s")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_counts, times_old_np, marker='o', label='Current Method (NumPy List-based)')
    plt.plot(qubit_counts, times_new_jax_compile, marker='s', linestyle='--', label='Proposed JAX (First Run / Compile + Exec)')
    plt.plot(qubit_counts, times_new_jax_run, marker='^', label='Proposed JAX (Subsequent Runs / Exec Only)')
    
    plt.yscale('log')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Execution Time (seconds) [Log Scale]')
    plt.title('Performance Benchmark: Current np.einsum vs JAX-accelerated String einsum')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()