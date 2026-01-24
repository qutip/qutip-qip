from qutip_qip.operations.gateclass import Gate

__all__ = ["resolve_decomposition", "fallback_decomposition"]

def resolve_decomposition(gate: Gate, basis_1q: list[str], basis_2q: list[str]) -> list[Gate]:
    """
    Attempt to decompose a gate into the given basis.
    """
    try:
        return gate.decompose_to_basis(basis_1q, basis_2q)
    except NotImplementedError:
        return fallback_decomposition(gate, basis_1q, basis_2q)


# Optional graph-based fallback resolver
GATE_DECOMP_GRAPH = {
    "MS": ["CNOT"],
    "CNOT": ["ISWAP"],
    # You can add more conversion steps here
}


def fallback_decomposition(gate: Gate, basis_1q: list[str], basis_2q: list[str]) -> list[Gate]:
    from collections import deque

    visited = set()
    queue = deque([(gate.name, [])])

    while queue:
        current_gate, path = queue.popleft()
        if current_gate in basis_2q:
            # Reconstruct intermediate decompositions
            decomposed = gate
            for intermediate in path:
                decomposed = decomposed.decompose_to_basis(basis_1q, [intermediate])
                decomposed = sum(
                    (g.decompose_to_basis(basis_1q, basis_2q) for g in decomposed),
                    start=[]
                )
            return decomposed

        for neighbor in GATE_DECOMP_GRAPH.get(current_gate, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    raise NotImplementedError(
        f"No valid decomposition path from {gate.name} to basis {basis_2q}"
    )
