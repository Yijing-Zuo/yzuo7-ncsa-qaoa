"""Unweighted graphs, reproducible rejection sampling and isomorphism classes."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json

import networkx as nx
import numpy as np


def validate_graph(graph: nx.Graph) -> None:
    """Require an undirected simple graph with no loops and unit edge weights.

    Missing weights mean one. Connectivity belongs to the dataset sampler,
    so the numerical core also permits disconnected and edgeless test cases.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("MaxCut expects an undirected simple graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Self-loops are outside the study's graph contract.")
    if any(data.get("weight", 1) != 1 for _, _, data in graph.edges(data=True)):
        raise ValueError("Only unweighted (unit-weight) edges are supported.")


WL_SETTINGS = {"iterations": 3, "digest_size": 16, "node_attr": None,
               "edge_attr": None, "networkx_version": nx.__version__}


def graph_from_record(record: dict) -> nx.Graph:
    """Restore the saved graph identity, including any isolated vertices."""
    graph = nx.Graph()
    graph.add_nodes_from(range(record["n"]))
    graph.add_edges_from(record["edges"])
    return graph


def _indexed_graph(graph):
    """Give the first representative insertion-order integer node labels."""
    validate_graph(graph)
    indices = {node: i for i, node in enumerate(graph)}
    edges = sorted(tuple(sorted((indices[u], indices[v]))) for u, v in graph.edges)
    return graph_from_record({"n": len(graph), "edges": edges})


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _add_candidate(candidate, records, buckets, namespace):
    """WL only filters candidates; GraphMatcher certifies every merge."""
    graph = _indexed_graph(candidate["graph"])
    provenance = deepcopy(candidate["provenance"])
    edges = [list(edge) for edge in sorted(graph.edges)]
    topology_hash = _hash({"n": len(graph), "edges": edges})
    wl_hash = nx.weisfeiler_lehman_graph_hash(
        graph, iterations=WL_SETTINGS["iterations"], digest_size=WL_SETTINGS["digest_size"],
    )
    bucket = buckets[len(graph), wl_hash]
    for record, representative in bucket:
        matcher = nx.isomorphism.GraphMatcher(graph, representative)
        if matcher.is_isomorphic():
            record["provenance"].append(provenance)
            record["iso_mappings"].append({
                "provenance": provenance, "candidate_edges": edges,
                "candidate_hash": topology_hash,
                "candidate_to_representative": [matcher.mapping[i] for i in range(len(graph))],
            })
            record["families"] = sorted({source["family"] for source in record["provenance"]})
            record["lofo_eligible"] = len(record["families"]) == 1
            return record, False, topology_hash
    record = {
        "iso_class_id": f"{namespace}:n{len(graph)}:{topology_hash}",
        "n": len(graph), "edges": edges, "representative_hash": topology_hash,
        "wl_hash": wl_hash, "representative_cell": provenance["cell_id"],
        "provenance": [provenance], "iso_mappings": [],
        "tier": 1, "graph_split": "not_applicable", "families": [provenance["family"]],
        "lofo_eligible": True,
    }
    records.append(record)
    bucket.append((record, graph))
    return record, True, topology_hash


def deduplicate_graphs(candidates, *, namespace="development-stage3") -> list[dict]:
    """Merge isomorphic candidates across all families/bands of the same n.

    Each candidate contains ``graph`` and ``provenance`` (including family and
    cell_id). The first encountered member owns the class and its quota. The
    representative hash is an edge-serialization hash, not a canonical label;
    stable replay requires the same candidate order and library namespace.
    Node/edge metadata never participates in WL hashing or isomorphism matching.
    """
    records, buckets = [], defaultdict(list)
    for candidate in candidates:
        _add_candidate(candidate, records, buckets, namespace)
    return records


def _draw_graph(n, family, target_degree, seed):
    """Draw once from the adopted generator, without connectivity repair."""
    if family == "regular":
        parameters = {"d": target_degree}
        graph = nx.random_regular_graph(target_degree, n, seed=seed)
    elif family == "er":
        probability = target_degree / (n - 1)
        parameters = {"probability": probability}
        graph = nx.gnp_random_graph(n, probability, seed=seed)
    elif family == "ba":
        m = {3: 2, 5: 3}[target_degree]
        initial = nx.star_graph(m)
        parameters = {"m": m, "initial_graph": "star_graph(m)",
                      "initial_graph_nodes": m + 1,
                      "initial_graph_edges": [list(edge) for edge in initial.edges]}
        graph = nx.barabasi_albert_graph(n, m, seed=seed, initial_graph=initial)
    elif family == "sbm":
        p_out = target_degree / (2 * n - 3)
        p_in = 3 * p_out
        sizes = [n // 2, n // 2]
        parameters = {"sizes": sizes, "p_in": p_in, "p_out": p_out, "ratio": 3}
        graph = nx.stochastic_block_model(
            sizes, [[p_in, p_out], [p_out, p_in]], seed=seed,
            directed=False, selfloops=False,
        )
    else:
        raise ValueError(f"Unknown graph family: {family}")
    return graph, parameters


def _allocate_counts(capacities, total):
    """Proportional largest remainder; ties follow stable cell insertion order."""
    size = sum(capacities.values())
    if size == 0:
        return dict.fromkeys(capacities, 0)
    total = min(total, size)
    counts = {key: total * count // size for key, count in capacities.items()}
    ranking = sorted(capacities, key=lambda key: -(total * capacities[key] % size))
    for key in ranking[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _stratified_sample(records, total, rng):
    groups = defaultdict(list)
    for record in records:
        groups[record["representative_cell"]].append(record)
    allocation = _allocate_counts({key: len(group) for key, group in groups.items()}, total)
    selected = []
    for key, group in groups.items():
        order = rng.permutation(len(group))
        selected.extend(group[int(i)] for i in order[:allocation[key]])
    return selected, allocation


def assign_splits(records, *, tier2_count, training_count, split_seed):
    """Fix graph subsets after deduplication; no QAOA result enters selection.

    Tier2 and training counts are allocated proportionally by representative
    cell with largest remainder, ties in generation order. Independent RNG
    streams permute members within each cell. A Tier2 shortfall retains the
    requested train/evaluation ratio, rounding a tied remainder to training.
    """
    if not 0 <= training_count <= tier2_count:
        raise ValueError("Require 0 <= training_count <= tier2_count.")
    for record in records:
        record.update(tier=1, graph_split="not_applicable")
    tier2_seed, training_seed = np.random.SeedSequence(split_seed).spawn(2)
    tier2, tier2_allocation = _stratified_sample(records, tier2_count, np.random.default_rng(tier2_seed))
    actual_training = (2 * len(tier2) * training_count + tier2_count) // (2 * tier2_count) if tier2_count else 0
    training, training_allocation = _stratified_sample(tier2, actual_training, np.random.default_rng(training_seed))
    for record in tier2:
        record.update(tier=2, graph_split="evaluation")
    for record in training:
        record["graph_split"] = "training"
    requested = {"tier2": tier2_count, "training": training_count,
                 "evaluation": tier2_count - training_count}
    actual = {"tier2": len(tier2), "training": len(training),
              "evaluation": len(tier2) - len(training)}
    return {
        "requested": requested, "actual": actual,
        "shortfall": {key: requested[key] - actual[key] for key in requested},
        "provisional": len(tier2) < tier2_count,
        "tier2_allocation": tier2_allocation, "training_allocation": training_allocation,
        "algorithm": "capacity-proportional largest remainder by representative cell; "
                     "ties follow generation order; SeedSequence(split_seed).spawn(2) separates "
                     "Tier2 and training permutations; shortfall scales train/evaluation ratio "
                     "with a tied remainder assigned to training",
        "lofo_rule": "Exclude every multi-family class from LOFO training and evaluation; "
                     "retain it for the global random graph split.",
    }


def generate_library(config: dict) -> dict:
    """Generate the configured cells, preserving every draw and collision.

    Order is increasing n, configured family order, increasing degree band.
    Only a new class owned by the current cell fills its quota. Connectivity
    uses rejection only; collisions and rejections consume attempts. Reaching
    the cap establishes a sampling shortfall, never exhaustion of a graph class.
    """
    config = deepcopy(config)
    sizes, families, bands = config["sizes"], config["families"], config["degree_bands"]
    if not sizes or not set(sizes) <= {12, 16, 20, 24}:
        raise ValueError("Generation sizes must be selected from 12, 16, 20, 24.")
    if not families or not set(families) <= {"regular", "er", "ba", "sbm"}:
        raise ValueError("Generation requires the Plan A graph families.")
    if not bands or not set(bands) <= {3, 5}:
        raise ValueError("Target degree bands must be selected from 3 and 5.")
    if any(len(items) != len(set(items)) for items in (sizes, families, bands)):
        raise ValueError("Duplicate sizes, families or bands would repeat cells.")
    max_attempts = config["max_attempts_per_cell"]
    if max_attempts < 1 or int(max_attempts) != max_attempts:
        raise ValueError("max_attempts_per_cell must be a positive integer.")
    records, attempts, cells, buckets = [], [], [], defaultdict(list)
    rng = np.random.default_rng(config["seed"])
    for n in sorted(sizes):
        for family in families:
            for band in sorted(bands):
                quota = config["quota_per_cell"]
                if isinstance(quota, dict):
                    quota = quota[str(n)]
                if quota < 0 or int(quota) != quota:
                    raise ValueError("Cell quotas must be nonnegative integers.")
                cell_id = f"n{n}:{family}:k{band}"
                stats = {"cell_id": cell_id, "n": n, "family": family, "target_degree": band,
                         "quota": quota, "attempts": 0, "accepted": 0, "duplicates": 0,
                         "rejected_disconnected": 0, "generator_errors": 0}
                while stats["accepted"] < quota and stats["attempts"] < max_attempts:
                    seed = int(rng.integers(0, 2**32, dtype=np.uint64))
                    stats["attempts"] += 1
                    source = {"family": family, "target_degree": band, "n": n,
                              "cell_id": cell_id, "seed": seed, "attempt": stats["attempts"],
                              "parameters": None}
                    try:
                        graph, parameters = _draw_graph(n, family, band, seed)
                    except nx.NetworkXException as error:
                        stats["generator_errors"] += 1
                        attempts.append({**source, "num_edges": None, "iso_class_id": None,
                                         "status": "generator_error",
                                         "error": f"{type(error).__name__}: {error}"})
                        continue
                    source["parameters"] = parameters
                    log = {**source, "num_edges": graph.number_of_edges(), "iso_class_id": None}
                    if not nx.is_connected(graph):
                        stats["rejected_disconnected"] += 1
                        log.update(status="rejected_disconnected", components=nx.number_connected_components(graph))
                    else:
                        record, is_new, candidate_hash = _add_candidate(
                            {"graph": graph, "provenance": source}, records, buckets, config["dataset_namespace"],
                        )
                        stats["accepted" if is_new else "duplicates"] += 1
                        log.update(status="accepted" if is_new else "duplicate",
                                   iso_class_id=record["iso_class_id"], candidate_hash=candidate_hash,
                                   representative_cell=record["representative_cell"])
                    attempts.append(log)
                stats.update(shortfall=quota - stats["accepted"], exhaustive=False,
                             status="satisfied" if stats["accepted"] == quota else "shortfall_unknown_population")
                cells.append(stats)
    split_summary = assign_splits(records, tier2_count=config["tier2_count"],
                                  training_count=config["training_count"], split_seed=config["split_seed"])
    identity = {"config": config, "graphs": records, "attempts": attempts,
                "wl_settings": WL_SETTINGS, "split_summary": split_summary}
    return {"graphs": records, "attempts": attempts, "cells": cells, "config": config,
            "library_id": f"{config['dataset_namespace']}:{_hash(identity)}",
            "wl_settings": deepcopy(WL_SETTINGS), "split_summary": split_summary}
