"""Classical cut scores and bounded-memory exact MaxCut enumeration."""

from collections.abc import Hashable, Iterable
from operator import index

import networkx as nx
import numpy as np

from .graphs import validate_graph


def cut_value(graph: nx.Graph, partition: Iterable[Hashable]) -> int:
    """Return the number of edges crossing one side of an unweighted cut.

    ``partition`` contains nodes on one side; all remaining graph nodes form
    the other side. Node labels need not be integers. The empty side is valid.
    """
    validate_graph(graph)
    side = set(partition)
    if not side.issubset(graph):
        raise ValueError("The cut partition contains nodes outside the graph.")
    return sum((u in side) != (v in side) for u, v in graph.edges)


def exact_maxcut(
    graph: nx.Graph, *, chunk_size: int = 65_536
) -> tuple[int, set[Hashable]]:
    """Return the exact MaxCut value and one optimal cut side for n <= 24.

    Enumerate bit masks in graph node iteration order, fixing the first node
    to bit zero because complementary cuts have the same score. At most
    ``chunk_size`` masks are held at once, so memory does not grow as 2**n.
    The runtime remains exponential; this function is scoped to Plan A's
    maximum of 24 nodes. The returned side consists of bit-one nodes.

    Ties select the first mask in ascending integer order, independently of
    chunk size. Relabeling nodes preserves the score; tied witnesses can
    change if the graph's node iteration order changes.
    """
    validate_graph(graph)
    chunk_size = index(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    nodes = list(graph.nodes)
    n = len(nodes)
    if n > 24:
        raise ValueError("Exact enumeration is limited to Plan A's n <= 24.")
    if graph.number_of_edges() == 0:
        return 0, set()

    position = {node: i for i, node in enumerate(nodes)}
    edges = [(position[u], position[v]) for u, v in graph.edges]
    best_value, best_mask = -1, 0
    for start in range(0, 1 << (n - 1), chunk_size):
        stop = min(start + chunk_size, 1 << (n - 1))
        masks = np.arange(start, stop, dtype=np.uint64) << np.uint64(1)
        values = np.zeros(stop - start, dtype=np.int64)
        for u, v in edges:
            crosses = ((masks >> np.uint64(u)) ^ (masks >> np.uint64(v))) & 1
            values += crosses.astype(np.int64)
        winner = int(np.argmax(values))
        if values[winner] > best_value:
            best_value = int(values[winner])
            best_mask = int(masks[winner])

    side = {node for i, node in enumerate(nodes) if (best_mask >> i) & 1}
    return best_value, side
