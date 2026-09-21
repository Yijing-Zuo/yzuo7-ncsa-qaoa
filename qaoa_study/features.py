"""Topology-only features for Plan A, with explicit definitions and missingness.

All variances are population variances. The local endpoint distribution samples
both endpoints of every undirected edge, so insertion order has no meaning.
Spectral moments are tr(A**k)/n, not normalized-Laplacian moments. Definitions
are frozen by FEATURE_VERSION; no generator, partition or QAOA metadata is read.
"""

from math import fsum, isfinite

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


B4_FEATURE_VERSION = "b4-features-v1"
B4_GROUPS = ("L", "U", "F", "J+U", "J+U+S")
B4_JOINT_TYPES = tuple((a, b, c) for a in range(23)
                       for b in range(a, 23) for c in range(a + 1))
_B4_JOINT_INDEX = {kind: index for index, kind in enumerate(B4_JOINT_TYPES)}
_B4_JOINT_NAMES = tuple(f"joint_edge_frequency_{a}_{b}_{c}" for a, b, c in B4_JOINT_TYPES)
_B4_MISSING = "degree_assortativity_missing"
_B4_MISSING_REASON = "constant_edge_endpoint_degree"


def b4_feature_names(group: str) -> tuple:
    """Return the frozen complete schema, before training-only constant masking."""
    if group not in B4_GROUPS:
        raise ValueError(f"Unknown B4 feature group: {group}")
    base = LOCAL_FEATURES if group == "L" else ML_FEATURES[:14]
    if group in ("F", "J+U+S"):
        base = ML_FEATURES
    joint = _B4_JOINT_NAMES if group.startswith("J+") else ()
    return joint + base + (_B4_MISSING,)


def _b4_values(values, missing):
    """Validate original scalars; only declared assortativity null is missing."""
    if (not isinstance(values, dict) or not isinstance(missing, dict)
            or not set(ML_FEATURES).issubset(values)):
        raise ValueError("B4 requires all original feature columns and missingness declarations.")
    declared = {"degree_assortativity": _B4_MISSING_REASON}
    if missing not in ({}, declared):
        raise ValueError("Only declared constant-degree assortativity may be missing in B4.")
    for name in ML_FEATURES:
        value = values[name]
        if value is None:
            if name != "degree_assortativity" or missing != declared:
                raise ValueError(f"Unexpected missing B4 feature: {name}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"B4 feature must be finite numeric or a declared null: {name}")
    if bool(missing) != (values["degree_assortativity"] is None):
        raise ValueError("B4 assortativity value and missingness declaration disagree.")
    n, m = values["n"], values["m"]
    if (int(n) != n or not 4 <= n <= 24 or int(m) != m
            or not n - 1 <= m <= n * (n - 1) // 2):
        raise ValueError("B4 requires simple connected graphs with 4 <= n <= 24.")


def b4_feature_row(graph: nx.Graph, stored_features=None) -> dict:
    """Create a graph-only sidecar row without modifying historical features.

    Optional stored_features has graph_features' {values, missing} structure;
    it is checked against the topology and retains the frozen original scalars.
    The connected normalized-L mean is then exactly one in this new sidecar.
    Callers attach library graph IDs, topology hashes and split provenance.
    """
    validate_graph(graph)
    if not 4 <= len(graph) <= 24 or not nx.is_connected(graph):
        raise ValueError("B4 requires simple connected graphs with 4 <= n <= 24.")
    computed = graph_features(graph)
    selected = computed if stored_features is None else stored_features
    _b4_values(selected["values"], selected["missing"])
    if selected["missing"] != computed["missing"]:
        raise ValueError("Stored B4 feature missingness disagrees with topology.")
    for name in ML_FEATURES:
        actual, expected = selected["values"][name], computed["values"][name]
        if (expected is None and actual is not None) or (
                expected is not None and (actual is None or abs(actual - expected) > 1e-10)):
            raise ValueError(f"Stored B4 feature disagrees with topology: {name}")
    degree = dict(graph.degree())
    counts = [0] * len(B4_JOINT_TYPES)
    for u, v in graph.edges:
        a, b = sorted((degree[u] - 1, degree[v] - 1))
        counts[_B4_JOINT_INDEX[(a, b, len(set(graph[u]) & set(graph[v])))]] += 1
    parities = {value % 2 for value in degree.values()}
    values = {name: selected["values"][name] for name in ML_FEATURES}
    values["normalized_laplacian_mean"] = 1.0
    return {"feature_version": B4_FEATURE_VERSION, "values": values,
            "missing": dict(selected["missing"]), "joint_counts": counts,
            "degree_parity": "even" if parities == {0} else "odd" if parities == {1} else "mixed"}


def b4_feature_matrix(rows, group: str) -> np.ndarray:
    """Build float64 inputs; legitimate assortativity null alone becomes NaN.

    The learner must fit its imputer on training rows. An input NaN/Inf is an
    error, distinct from a declared JSON null. Joint frequencies retain all
    2300 positions, including impossible and training-unseen types.
    """
    names = b4_feature_names(group)
    matrix = np.empty((len(rows), len(names)), dtype=np.float64)
    for index, row in enumerate(rows):
        if row.get("feature_version") != B4_FEATURE_VERSION:
            raise ValueError("Unsupported B4 feature version.")
        values, missing, counts = row["values"], row["missing"], row["joint_counts"]
        _b4_values(values, missing)
        n, m = values["n"], values["m"]
        if (len(counts) != len(B4_JOINT_TYPES)
                or any(type(count) is not int or count < 0 for count in counts)
                or sum(counts) != m):
            raise ValueError("B4 joint counts require 2300 nonnegative integers summing to m.")
        active = [(kind, count) for kind, count in zip(B4_JOINT_TYPES, counts) if count]
        if any(a + b - c > n - 2 or (a, b, c) == (0, 0, 0) for (a, b, c), _ in active):
            raise ValueError("B4 joint counts contain an impossible connected-graph type.")
        if abs(fsum(count * (1 / (a + 1) + 1 / (b + 1))
                    for (a, b, c), count in active) - n) > 1e-10:
            raise ValueError("B4 joint counts disagree with vertex count.")
        parity = {(d + 1) % 2 for (a, b, c), _ in active for d in (a, b)}
        expected_parity = "even" if parity == {0} else "odd" if parity == {1} else "mixed"
        if row.get("degree_parity") != expected_parity or values["normalized_laplacian_mean"] != 1.0:
            raise ValueError("B4 parity or exact normalized-L constant is inconsistent.")
        vector = [count / m for count in counts] if group.startswith("J+") else []
        original = names[len(vector):-1]
        vector.extend(np.nan if values[name] is None else values[name] for name in original)
        vector.append(float(bool(missing)))
        matrix[index] = vector
    return matrix
