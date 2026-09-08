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


SPLIT_VERSION = "seeded-largest-remainder-v2"


def _allocate_counts(capacities, total, tie_rng):
    """Proportional largest remainder, breaking equal remainders by seed.

    Sorting keys before the independent random ranking removes any dependency
    on dictionary insertion order. The ranking never uses graph/QAOA outcomes.
    """
    size = sum(capacities.values())
    if size == 0:
        return dict.fromkeys(capacities, 0)
    total = min(total, size)
    counts = {key: total * count // size for key, count in capacities.items()}
    keys = sorted(capacities)
    random_rank = dict(zip(keys, tie_rng.permutation(len(keys)), strict=True))
    ranking = sorted(keys, key=lambda key: (-(total * capacities[key] % size), random_rank[key]))
    for key in ranking[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _stratified_sample(records, total, member_rng, tie_rng):
    groups = defaultdict(list)
    for record in records:
        groups[record["representative_cell"]].append(record)
    allocation = _allocate_counts({key: len(group) for key, group in groups.items()}, total, tie_rng)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: record["iso_class_id"])
        order = member_rng.permutation(len(group))
        selected.extend(group[int(i)] for i in order[:allocation[key]])
    return selected, allocation


def graph_split_coverage(records, cells=None):
    """Count graph-split coverage by the fixed representative's ownership.

    Multi-family provenance remains in each graph record; counting every source
    as a separate graph would inflate coverage. Missing training or evaluation
    cells (including cells absent from sampling) make this a smoke test, not a
    balanced random generalization study. Coverage alone is not a power claim.
    """
    metadata = {cell["cell_id"]: {"n": cell["n"], "family": cell["family"],
                                  "degree_band": cell["target_degree"]} for cell in cells or []}
    for record in records:
        source = record.get("provenance", [{}])[0]
        metadata.setdefault(record["representative_cell"], {
            "n": record.get("n", source.get("n", "unknown")),
            "family": source.get("family", "unknown"),
            "degree_band": source.get("target_degree", "unknown"),
        })
    tables = {}
    for name, dimension in (("by_n", "n"), ("by_family", "family"),
                            ("by_degree_band", "degree_band"), ("by_cell", "cell_id")):
        counts = {}
        for cell_id, dimensions in metadata.items():
            key = cell_id if dimension == "cell_id" else dimensions[dimension]
            columns = dimensions if dimension == "cell_id" else {}
            counts.setdefault(key, {**columns, dimension: key, "training": 0, "evaluation": 0,
                                    "not_applicable": 0, "tier2": 0, "total": 0})
        for record in records:
            cell_id = record["representative_cell"]
            key = cell_id if dimension == "cell_id" else metadata[cell_id][dimension]
            row = counts[key]
            row[record["graph_split"]] += 1
            row["tier2"] += int(record["tier"] == 2)
            row["total"] += 1
        tables[name] = [counts[key] for key in sorted(counts, key=str)]
    incomplete = [row["cell_id"] for row in tables["by_cell"]
                  if row["training"] == 0 or row["evaluation"] == 0]
    return {**tables, "smoke_test": bool(incomplete) or not metadata,
            "coverage_status": "smoke_test_missing_split_cells" if incomplete or not metadata
                               else "all_cells_represented_no_power_claim",
            "missing_training_or_evaluation_cells": incomplete,
            "ownership": "first_representative_cell"}


def assign_splits(records, *, tier2_count, training_count, split_seed, cells=None):
    """Fix graph subsets after deduplication; no QAOA result enters selection.

    Tier2 and training counts are allocated proportionally by representative
    cell with largest remainder and independently seeded random tie rankings.
    Separate RNG streams permute members within each cell. A Tier2 shortfall retains the
    requested train/evaluation ratio, rounding a tied remainder to training.
    """
    if not 0 <= training_count <= tier2_count:
        raise ValueError("Require 0 <= training_count <= tier2_count.")
    for record in records:
        record.update(tier=1, graph_split="not_applicable")
    streams = [np.random.default_rng(seed) for seed in np.random.SeedSequence(split_seed).spawn(4)]
    tier2, tier2_allocation = _stratified_sample(records, tier2_count, streams[0], streams[1])
    actual_training = (2 * len(tier2) * training_count + tier2_count) // (2 * tier2_count) if tier2_count else 0
    training, training_allocation = _stratified_sample(tier2, actual_training, streams[2], streams[3])
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
        "split_version": SPLIT_VERSION, "split_seed": split_seed,
        "coverage": graph_split_coverage(records, cells),
        "algorithm": "capacity-proportional largest remainder by representative cell; "
                     "SeedSequence(split_seed).spawn(4) separates Tier2 member, Tier2 tie, "
                     "training member and training tie streams; sorted cell/identity order; "
                     "shortfall scales train/evaluation ratio "
                     "with a tied remainder assigned to training",
        "lofo_rule": "Exclude every multi-family class from LOFO training and evaluation; "
                     "retain it for the global random graph split.",
    }


def resplit_library(library, *, split_seed, tier2_count=None, training_count=None):
    """Create a new split revision without regenerating/relabeling any graph.

    The caller saves to a new directory. Original topology, iso IDs, features,
    exact witnesses, provenance and draw logs are copied byte-for-value.
    """
    revised = deepcopy(library)
    config = revised["config"]
    config.update(split_seed=split_seed,
                  tier2_count=config["tier2_count"] if tier2_count is None else tier2_count,
                  training_count=config["training_count"] if training_count is None else training_count)
    summary = assign_splits(revised["graphs"], tier2_count=config["tier2_count"],
                            training_count=config["training_count"], split_seed=split_seed,
                            cells=revised.get("cells"))
    revised.update(parent_library_id=library["library_id"], split_summary=summary,
                   split_revision=library.get("split_revision", 0) + 1)
    identity = {"parent_library_id": library["library_id"], "split_summary": summary,
                "splits": [(row["iso_class_id"], row["tier"], row["graph_split"])
                           for row in revised["graphs"]]}
    revised["library_id"] = f"{config['dataset_namespace']}:{_hash(identity)}"
    return revised


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
                                  training_count=config["training_count"], split_seed=config["split_seed"],
                                  cells=cells)
    identity = {"config": config, "graphs": records, "attempts": attempts,
                "wl_settings": WL_SETTINGS, "split_summary": split_summary}
    return {"graphs": records, "attempts": attempts, "cells": cells, "config": config,
            "library_id": f"{config['dataset_namespace']}:{_hash(identity)}",
            "wl_settings": deepcopy(WL_SETTINGS), "split_summary": split_summary}
