"""Positive MaxCut QAOA and the corrected p=1 edge formula.

Parameters are radians, [gamma_1, ..., gamma_p, beta_1, ..., beta_p].
We prepare exp(-i beta_p B) exp(-i gamma_p C) ... |+>, with positive
C = sum((I - Z_u Z_v)/2) and B = sum(X_v). No optimizer is defined here.
"""

import networkx as nx
import numpy as np
import pennylane as qml
from autograd import value_and_grad
from pennylane import numpy as pnp

from qaoa_study.graphs import validate_graph


def validate_depth(p: int) -> int:
    """Return a positive integer circuit depth without rounding user input."""
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError("Expected p to be a positive integer.")
    return int(p)


def _circuit_graph(graph, p):
    validate_graph(graph)
    validate_depth(p)
    nodes = tuple(graph.nodes)
    if not nodes:
        raise ValueError("A QAOA circuit requires at least one node.")
    wire = {node: i for i, node in enumerate(nodes)}
    return len(nodes), tuple((wire[u], wire[v]) for u, v in graph.edges)


def _apply_qaoa(theta, p, n, edges):
    """Apply the shared exact/sampled circuit in the study's gate convention."""
    if qml.math.shape(theta) != (2 * p,):
        raise ValueError(f"Expected {2 * p} angles in [gamma..., beta...] order.")
    angles = qml.math.cast(theta, "float64")
    for v in range(n):
        qml.Hadamard(wires=v)
    for layer in range(p):
        for u, v in edges:
            qml.IsingZZ(-angles[layer], wires=(u, v))
        for v in range(n):
            qml.RX(2 * angles[p + layer], wires=v)


def make_qaoa(graph: nx.Graph, p: int, *, backend: str = "default.qubit") -> qml.QNode:
    """Build an exact-expectation QNode at any positive integer depth.

    Node insertion order determines the internal wires; graph labels may be any
    hashable objects. The edge list is captured at construction, so later graph
    mutations do not change this circuit. All supported backends use a
    complex128 state, float64 angles, Autograd and adjoint first derivatives.
    ``lightning.gpu`` requires a separately prepared GPU environment; device
    construction errors propagate and never select a different backend.
    The single-term transform keeps one tape and scalar result. Device VJP
    contracts its cotangents before the adjoint sweep, so default.qubit keeps
    one weighted bra rather than one state per edge. No dense global observable
    or full measurement Jacobian is needed for this scalar first derivative.
    """
    n, edges = _circuit_graph(graph, p)
    if backend not in ("default.qubit", "lightning.qubit", "lightning.gpu"):
        raise ValueError("backend must be default.qubit, lightning.qubit or lightning.gpu.")
    coefficients = [len(edges) / 2] + [-0.5] * len(edges)
    observables = [qml.Identity(0)] + [qml.Z(u) @ qml.Z(v) for u, v in edges]
    cut_operator = qml.Hamiltonian(coefficients, observables)
    options = {"c_dtype": np.complex128} if backend.startswith("lightning.") else {}
    device = qml.device(backend, wires=n, shots=None, **options)

    @qml.qnode(device, interface="autograd", diff_method="adjoint", device_vjp=True)
    def circuit(theta):
        _apply_qaoa(theta, p, n, edges)
        return qml.expval(cut_operator)

    # The adjoint kernel otherwise densifies a global LinearCombination. The
    # official transform measures local terms on one tape and restores the sum.
    # Keep the constant zero Identity for edgeless graphs: removing every term
    # leaves no measurement, which PennyLane 0.43's adjoint cannot differentiate.
    return qml.transforms.split_to_single_terms(circuit) if edges else circuit


def make_qaoa_sampler(graph: nx.Graph, p: int, *, shots: int, seed: int,
                      backend: str = "default.qubit") -> qml.QNode:
    """Sample ideal QAOA bitstrings, with columns in graph node insertion order.

    Each call consumes fresh draws from this device's seeded RNG. Reconstructing
    with the same seed reproduces the call sequence; no derivative or gate noise
    is used. Topology is captured at construction as in ``make_qaoa``. At least
    two shots are required to estimate whole-cut variance. Backend errors
    propagate. GPU sampling requires the user-run backend verification before
    a research pool; it never silently selects a CPU implementation.
    """
    n, edges = _circuit_graph(graph, p)
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, (int, np.integer)) or shots < 2:
        raise ValueError("shots must be an integer of at least 2.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if backend not in ("default.qubit", "lightning.qubit", "lightning.gpu"):
        raise ValueError("sampled backend must be default.qubit, lightning.qubit or lightning.gpu.")
    options = {"c_dtype": np.complex128} if backend.startswith("lightning.") else {}
    device = qml.device(backend, wires=n, seed=int(seed), **options)

    @qml.set_shots(shots=int(shots))
    @qml.qnode(device, interface=None, diff_method=None)
    def circuit(theta):
        _apply_qaoa(theta, p, n, edges)
        return qml.sample(wires=range(n))

    return circuit


def cut_sample_statistics(graph: nx.Graph, samples) -> dict:
    """Summarize complete cut values, preserving correlations between edges.

    Rows are independent bitstrings and columns follow ``tuple(graph.nodes)``.
    ``sample_variance`` uses ddof=1; the standard error is its square root
    divided by sqrt(shots). An observed zero variance is not a guarantee that
    the true variance is zero or a confidence certificate for success.
    """
    validate_graph(graph)
    bits = np.asarray(samples)
    if bits.ndim != 2 or bits.shape[1] != len(graph) or bits.shape[0] < 2:
        raise ValueError("samples must have shape (shots >= 2, number of graph nodes).")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("samples must contain only binary 0/1 values.")
    wire = {node: i for i, node in enumerate(graph)}
    cuts = np.zeros(len(bits), dtype=np.int64)
    for u, v in graph.edges:
        cuts += bits[:, wire[u]] != bits[:, wire[v]]
    variance = float(np.var(cuts, ddof=1))
    return {"C_mean": float(np.mean(cuts)), "sample_variance": variance,
            "standard_error": float(np.sqrt(variance / len(bits))), "shots": len(bits)}


def qaoa_loss(theta, circuit: qml.QNode):
    """Return -<C> without detaching the Autograd computation graph."""
    return -circuit(theta)


def qaoa_gradient(theta, circuit: qml.QNode, *, loss: bool = False) -> np.ndarray:
    """Return the float64 first derivative of <C>, or of -<C> if loss=True.

    Adjoint differentiation handles each parameter's use in multiple gates.
    No finite differences or whole-layer parameter-shift shortcut is used.
    """
    angles = pnp.array(theta, dtype=np.float64, requires_grad=True)
    objective = (lambda x: qaoa_loss(x, circuit)) if loss else circuit
    return np.asarray(qml.grad(objective)(angles), dtype=np.float64)


def qaoa_loss_and_gradient(theta, circuit: qml.QNode) -> tuple[float, np.ndarray]:
    """Return loss and its float64 gradient from one differentiated QNode call.

    This tuple is suitable for SciPy ``jac=True``. The positive score is
    ``-loss``; obtaining it requires no additional circuit execution. Autograd
    retains the primal value while applying its vector-Jacobian product.
    Device execution and derivative work must still be counted separately.
    """
    angles = pnp.array(theta, dtype=np.float64, requires_grad=True)
    loss, gradient = value_and_grad(lambda x: qaoa_loss(x, circuit))(angles)
    return float(loss), np.asarray(gradient, dtype=np.float64)


def _p1_edge_values(graph: nx.Graph, gamma, beta) -> dict:
    """Evaluate Wang et al., arXiv:1706.02998v2 Eq. (14), for every edge.

    a and b exclude the opposite endpoint; t counts common neighbors. The
    triangle correction is necessary even for p=1. Keys follow graph.edges.
    This analytic reference applies to simple unit-weight graphs and the
    positive-C convention above, with arbitrary real radian angles.
    """
    validate_graph(graph)
    neighbors = {v: set(graph[v]) for v in graph}
    cos_gamma = np.cos(gamma)
    cos_double_gamma = np.cos(2 * gamma)
    linear = np.sin(4 * beta) * np.sin(gamma) / 4
    triangle_factor = np.sin(2 * beta) ** 2 / 4
    values = {}
    for u, v in graph.edges:
        a, b = len(neighbors[u]) - 1, len(neighbors[v]) - 1
        t = len(neighbors[u] & neighbors[v])
        values[u, v] = (
            0.5
            + linear * (cos_gamma**a + cos_gamma**b)
            - triangle_factor
            * cos_gamma ** (a + b - 2 * t)
            * (1 - cos_double_gamma**t)
        )
    return values


def p1_edge_expectations(graph: nx.Graph, gamma: float, beta: float) -> dict:
    """Return scalar Wang 1706.02998v2 Eq. (14) expectations for every edge."""
    return {edge: float(value) for edge, value in _p1_edge_values(graph, gamma, beta).items()}


def p1_expectation(graph: nx.Graph, gamma: float, beta: float) -> float:
    """Sum the corrected analytic p=1 expectations over all graph edges."""
    return float(sum(p1_edge_expectations(graph, gamma, beta).values()))


def p1_expectation_grid(graph: nx.Graph, gamma, beta) -> np.ndarray:
    """Evaluate the same edge formula on NumPy-broadcastable angle arrays.

    For a rectangular grid pass ``gamma[:, None]`` and ``beta[None, :]``.
    Callers choose chunk sizes; no QNode or state vector is constructed.
    """
    gamma, beta = np.broadcast_arrays(np.asarray(gamma, dtype=np.float64),
                                      np.asarray(beta, dtype=np.float64))
    result = np.zeros(gamma.shape, dtype=np.float64)
    for value in _p1_edge_values(graph, gamma, beta).values():
        result += value
    return result
