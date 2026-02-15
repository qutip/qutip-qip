# Circuit Simulation Benchmark Suite

This directory provides a minimal and reproducible benchmark
framework for evaluating circuit simulation performance in qutip-qip.

Included circuits:
- GHZ state preparation (entanglement growth)
- Random layered circuit (single-qubit rotations + CNOT chain)

The benchmark:
- Uses |0...0> initial state
- Performs a warmup run
- Reports average runtime over multiple executions
- Demonstrates scaling with increasing qubit count

Run with:

    python -m benchmarks.run_benchmarks

This framework is intended to support future performance comparisons
between sparse and tensor-based execution paths.

