"""Topology-only features for Plan A, with explicit definitions and missingness.

All variances are population variances. The local endpoint distribution samples
both endpoints of every undirected edge, so insertion order has no meaning.
Spectral moments are tr(A**k)/n, not normalized-Laplacian moments. Definitions
are frozen by FEATURE_VERSION; no generator, partition or QAOA metadata is read.
"""

from math import fsum

import networkx as nx
import numpy as np

from .graphs import validate_graph


FEATURE_VERSION = "1"
LOCAL_FEATURES = (
    "triangle_edge_mean", "triangle_edge_variance",
    "endpoint_excess_degree_mean", "endpoint_excess_degree_variance",
    "degree_assortativity",
)
ML_FEATURES = (
    "n", "m", "density", "mean_degree", "degree_min", "degree_max",
    "degree_variance", "mean_clustering", "transitivity", "degree_assortativity",
    "triangle_edge_mean", "triangle_edge_variance",
    "endpoint_excess_degree_mean", "endpoint_excess_degree_variance",
    "adjacency_radius", "adjacency_gap", "laplacian_lambda2",
    "normalized_laplacian_lambda2", "normalized_laplacian_max",
    "normalized_laplacian_mean", "normalized_laplacian_variance",
    "adjacency_moment_2", "adjacency_moment_3", "adjacency_moment_4",
)
# Connectivity is retained for validation, but is constant in the connected
# production population and is deliberately absent from the ML allowlist.


def graph_features(graph: nx.Graph) -> dict:
    """Compute simple unweighted-graph features and reasons for undefined values.

    Let A be adjacency, D the degree diagonal, and L=D-A. The adjacency gap
    is the largest minus second-largest *algebraic* eigenvalue. Algebraic
    connectivity is the second-smallest eigenvalue of L. Normalized L is
    D**(-1/2) L D**(-1/2), assigning zero rows/columns to isolated vertices;
    its summary is lambda2, maximum, mean and population variance. Theoretical
    [0, infinity) and [0,2] spectral ranges remove floating-point roundoff.

    Per-edge triangles are |N(u) intersect N(v)|. Excess degrees d(v)-1 are
    pooled over the 2m edge endpoints. Their Pearson degree correlation counts
    each edge in both orientations; it equals ordinary degree assortativity.
    Local clustering assigns zero to vertices with degree below two. Global
    transitivity is undefined when there are no length-two wedges.

    Empty/disconnected graphs are supported for tests. Production sampling
    separately requires connected graphs. All returned scalars are JSON-safe;
    undefined values are None with a machine-readable reason in ``missing``.
    """
    validate_graph(graph)
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    values = dict.fromkeys((*ML_FEATURES, "is_connected"))
    values.update(n=n, m=m)
    missing = {}
    if not n:
        missing = {key: "no_vertices" for key in values if key not in ("n", "m")}
        return {"values": values, "missing": missing}

    nodes = list(graph)
    degree = dict(graph.degree())
    degrees = np.asarray([degree[v] for v in nodes], dtype=np.float64)
    values.update(
        mean_degree=float(2 * m / n), degree_min=int(degrees.min()),
        degree_max=int(degrees.max()), degree_variance=float(degrees.var()),
        mean_clustering=fsum(nx.clustering(graph, weight=None).values()) / n,
        is_connected=nx.is_connected(graph),
    )
    if n > 1:
        values["density"] = 2 * m / (n * (n - 1))
    else:
        missing["density"] = "fewer_than_two_vertices"

    triangles = sum(nx.triangles(graph).values()) // 3
    wedges = sum(d * (d - 1) // 2 for d in degree.values())
    if wedges:
        values["transitivity"] = 3 * triangles / wedges
    else:
        missing["transitivity"] = "no_length_two_wedges"

    if m:
        neighbors = {v: set(graph[v]) for v in nodes}
        edge_triangles = np.asarray(
            [len(neighbors[u] & neighbors[v]) for u, v in graph.edges], dtype=np.float64
        )
        endpoints = np.asarray(
            [degree[v] - 1 for edge in graph.edges for v in edge], dtype=np.float64
        )
        mean_excess, variance_excess = float(endpoints.mean()), float(endpoints.var())
        values.update(
            triangle_edge_mean=float(edge_triangles.mean()),
            triangle_edge_variance=float(edge_triangles.var()),
            endpoint_excess_degree_mean=mean_excess,
            endpoint_excess_degree_variance=variance_excess,
        )
        if variance_excess:
            cross_moment = sum((degree[u] - 1) * (degree[v] - 1) for u, v in graph.edges) / m
            values["degree_assortativity"] = (cross_moment - mean_excess**2) / variance_excess
        else:
            missing["degree_assortativity"] = "constant_edge_endpoint_degree"
    else:
        missing.update({key: "no_edges" for key in LOCAL_FEATURES})

    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None, dtype=np.float64)
    adjacency_eigenvalues = np.linalg.eigvalsh(adjacency)
    laplacian_eigenvalues = np.maximum(np.linalg.eigvalsh(np.diag(degrees) - adjacency), 0)
    inverse_sqrt = np.divide(1, np.sqrt(degrees), out=np.zeros(n), where=degrees > 0)
    normalized = np.diag((degrees > 0).astype(np.float64)) - (
        inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    )
    normalized_eigenvalues = np.clip(np.linalg.eigvalsh(normalized), 0, 2)
    values.update(
        adjacency_radius=float(np.abs(adjacency_eigenvalues).max()),
        normalized_laplacian_max=float(normalized_eigenvalues[-1]),
        normalized_laplacian_mean=float(normalized_eigenvalues.mean()),
        normalized_laplacian_variance=float(normalized_eigenvalues.var()),
    )
    if n > 1:
        values.update(
            adjacency_gap=float(adjacency_eigenvalues[-1] - adjacency_eigenvalues[-2]),
            laplacian_lambda2=float(laplacian_eigenvalues[1]),
            normalized_laplacian_lambda2=float(normalized_eigenvalues[1]),
        )
    else:
        missing.update({key: "fewer_than_two_vertices" for key in (
            "adjacency_gap", "laplacian_lambda2", "normalized_laplacian_lambda2"
        )})
    # Trace powers give the same spectral moments, without eigenvalue roundoff.
    power = adjacency @ adjacency
    for order in (2, 3, 4):
        values[f"adjacency_moment_{order}"] = float(np.trace(power) / n)
        if order < 4:
            power = power @ adjacency
    return {"values": values, "missing": missing}
