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


def make_qaoa(graph: nx.Graph, p: int, *, backend: str = "default.qubit") -> qml.QNode:
    """Build the study's p=1 or p=2 exact-expectation QNode once for a graph.

    Node insertion order determines the internal wires; graph labels may be any
    hashable objects. The edge list is captured at construction, so later graph
    mutations do not change this circuit. Both supported backends use a
    complex128 state, float64 angles, Autograd and adjoint first derivatives.
    ``lightning.gpu`` requires a separately prepared GPU environment; device
    construction errors propagate and never select a different backend.
    The single-term transform keeps one tape and scalar result. Device VJP
    contracts its cotangents before the adjoint sweep, so default.qubit keeps
    one weighted bra rather than one state per edge. No dense global observable
    or full measurement Jacobian is needed for this scalar first derivative.
    """
    validate_graph(graph)
    if p not in (1, 2):
        raise ValueError("This study implements p=1 and p=2 only.")
    if backend not in ("default.qubit", "lightning.gpu"):
        raise ValueError("backend must be default.qubit or lightning.gpu.")
    nodes = tuple(graph.nodes)
    if not nodes:
        raise ValueError("A QAOA circuit requires at least one node.")
    wire = {node: i for i, node in enumerate(nodes)}
    edges = tuple((wire[u], wire[v]) for u, v in graph.edges)
    coefficients = [len(edges) / 2] + [-0.5] * len(edges)
    observables = [qml.Identity(0)] + [qml.Z(u) @ qml.Z(v) for u, v in edges]
    cut_operator = qml.Hamiltonian(coefficients, observables)
    options = {"c_dtype": np.complex128} if backend == "lightning.gpu" else {}
    device = qml.device(backend, wires=len(nodes), shots=None, **options)

    @qml.qnode(device, interface="autograd", diff_method="adjoint", device_vjp=True)
    def circuit(theta):
        if qml.math.shape(theta) != (2 * p,):
            raise ValueError(f"Expected {2 * p} angles in [gamma..., beta...] order.")
        angles = qml.math.cast(theta, "float64")
        for v in range(len(nodes)):
            qml.Hadamard(wires=v)
        for layer in range(p):
            for u, v in edges:
                qml.IsingZZ(-angles[layer], wires=(u, v))
            for v in range(len(nodes)):
                qml.RX(2 * angles[p + layer], wires=v)
        return qml.expval(cut_operator)

    # The adjoint kernel otherwise densifies a global LinearCombination. The
    # official transform measures local terms on one tape and restores the sum.
    # Keep the constant zero Identity for edgeless graphs: removing every term
    # leaves no measurement, which PennyLane 0.43's adjoint cannot differentiate.
    return qml.transforms.split_to_single_terms(circuit) if edges else circuit


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
