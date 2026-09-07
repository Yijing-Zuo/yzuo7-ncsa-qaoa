"""Small, explicit run records and post-hoc threshold evaluation."""

import json
import math
from pathlib import Path
import copy
import hashlib
from importlib.metadata import version
import platform


def evaluate_trace(result: dict, C_ref: float, epsilon: float = 0.5) -> dict:
    """Evaluate existing scores without computing a circuit or changing a run.

    first_hit includes finite unaccepted trials. Numerical/evaluation failures
    cannot be terminal successes; their last valid accepted point is retained
    as C_final for inspection. Nonconvergence alone does not imply threshold failure.
    """
    if not math.isfinite(C_ref) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("C_ref must be finite and epsilon finite and nonnegative.")
    threshold = C_ref - epsilon
    first_hit = next((row["call_id"] for row in result["trace"]
                      if row["C"] is not None and math.isfinite(row["C"]) and row["C"] >= threshold), None)
    final = result["C_final"]
    invalid_run = result["stop_reason"] in ("numerical_error", "evaluation_error", "optimizer_failed")
    return {"C_ref": C_ref, "epsilon": epsilon, "first_hit": first_hit,
            "hit": first_hit is not None,
            "terminal_success": bool(not invalid_run and final is not None and
                                     math.isfinite(final) and final >= threshold),
            "reference_exceeded": any(row["C"] is not None and row["C"] > C_ref for row in result["trace"])}


def save_run(path, result: dict) -> None:
    """Atomically save one completed attempt; no optimizer-state checkpointing."""
    if not result.get("run_completed"):
        raise ValueError("Only a completed attempt can be saved as a run record.")
    write_json(path, result)


def read_run(path) -> dict:
    """Read a completed attempt; interrupted work must restart as a new attempt."""
    result = read_json(path)
    if not result.get("run_completed"):
        raise ValueError("Incomplete run record.")
    return result


def write_json(path, value) -> None:
    """Replace a small JSON file after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_json(path):
    """Load a small UTF-8 JSON configuration or record."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def annotate_library(library, exact_sizes=(12, 16), chunk_size=65_536):
    """Add graph-only features and exact witnesses, never using QAOA outcomes.

    Development may omit expensive exact annotations at larger sizes. Such
    values are null with an explicit status, not zero or an approximate optimum.
    """
    from .exact import exact_maxcut
    from .features import FEATURE_VERSION, graph_features
    from .graphs import graph_from_record

    annotated = copy.deepcopy(library)
    for record in annotated["graphs"]:
        graph = graph_from_record(record)
        features = graph_features(graph)
        record.update(features=features["values"], feature_missing=features["missing"],
                      feature_version=FEATURE_VERSION)
        if record["n"] in exact_sizes:
            optimum, side = exact_maxcut(graph, chunk_size=chunk_size)
            record.update(C_star=optimum, optimal_cut_side=sorted(side), exact_status="computed")
        else:
            record.update(C_star=None, optimal_cut_side=None, exact_status="not_computed_development")
    annotated["annotations"] = {"exact_sizes": list(exact_sizes), "exact_chunk_size": chunk_size,
                                "feature_version": FEATURE_VERSION}
    return annotated


def _split_manifest(records):
    splits = {name: [] for name in ("training", "evaluation", "not_applicable")}
    seen = set()
    for record in records:
        identity, split = record["iso_class_id"], record["graph_split"]
        if identity in seen:
            raise ValueError("Repeated iso_class_id; graph splits must be disjoint.")
        seen.add(identity)
        if record["tier"] not in (1, 2) or (record["tier"] == 2 and split not in ("training", "evaluation")) or (
                record["tier"] == 1 and split != "not_applicable"):
            raise ValueError("graph_split does not match tier membership.")
        splits[split].append(identity)
    return splits


def save_library(directory, library):
    """Freeze a new graph library as Parquet plus small JSON/JSONL sidecars.

    Existing directories are never overwritten. The final manifest is the
    completion marker; an interrupted directory is not a readable library.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    splits = _split_manifest(library["graphs"])
    if library["graphs"]:
        first = library["graphs"][0]
        for record in library["graphs"]:
            if set(record) != set(first) or set(record["features"]) != set(first["features"]):
                raise ValueError("Graph records and feature fields must have a consistent schema.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    rows = []
    # These varying dictionaries are provenance, not numeric model columns.
    json_columns = ("provenance", "iso_mappings", "feature_missing")
    for record in library["graphs"]:
        row = dict(record)
        for name in json_columns:
            row[name + "_json"] = json.dumps(row.pop(name), sort_keys=True, allow_nan=False)
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata({b"qaoa_study_graph_schema": b"1"})
    pq.write_table(table, directory / "graphs.parquet", compression="zstd")
    write_json(directory / "config.json", library["config"])
    write_json(directory / "splits.json", splits)
    with (directory / "generation.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for attempt in library["attempts"]:
            stream.write(json.dumps(attempt, sort_keys=True, allow_nan=False) + "\n")
    files = ("graphs.parquet", "config.json", "generation.jsonl", "splits.json")
    manifest = {
        "schema_version": 1, "complete": True,
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files},
        "library_metadata": {key: value for key, value in library.items() if key not in ("graphs", "attempts", "config")},
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in ("networkx", "numpy", "scipy", "pyarrow")}},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("graphs.py", "exact.py", "features.py", "records.py")},
    }
    write_json(directory / "manifest.json", manifest)


def read_library(directory):
    """Read frozen research data and verify file identity and graph boundaries."""
    import pyarrow.parquet as pq

    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Unsupported or incomplete graph library.")
    for name in ("graphs.parquet", "config.json", "generation.jsonl", "splits.json"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Graph library checksum mismatch: {name}")
    table = pq.read_table(directory / "graphs.parquet")
    if (table.schema.metadata or {}).get(b"qaoa_study_graph_schema") != b"1":
        raise ValueError("Unsupported Parquet graph schema.")
    records = table.to_pylist()
    for record in records:
        for name in ("provenance", "iso_mappings", "feature_missing"):
            record[name] = json.loads(record.pop(name + "_json"))
    if _split_manifest(records) != read_json(directory / "splits.json"):
        raise ValueError("Saved split manifest disagrees with graph records.")
    return {**manifest["library_metadata"], "graphs": records,
            "config": read_json(directory / "config.json"),
            "attempts": [json.loads(line) for line in
                         (directory / "generation.jsonl").read_text(encoding="utf-8").splitlines()]}


def check_library(library):
    """Audit restored topology, stored cut witnesses, features and graph splits."""
    import networkx as nx
    from .exact import cut_value
    from .features import graph_features
    from .graphs import graph_from_record, validate_graph

    splits = _split_manifest(library["graphs"])
    max_feature_error = 0.0
    exact_checked = 0
    for record in library["graphs"]:
        graph = graph_from_record(record)
        validate_graph(graph)
        if not nx.is_connected(graph):
            raise ValueError("A study library contains a disconnected graph.")
        if record["exact_status"] == "computed":
            if cut_value(graph, record["optimal_cut_side"]) != record["C_star"]:
                raise ValueError("Stored exact cut witness disagrees with its score.")
            exact_checked += 1
        elif record["C_star"] is not None or record["optimal_cut_side"] is not None:
            raise ValueError("Uncomputed exact answer must be null.")
        recomputed = graph_features(graph)
        if recomputed["missing"] != record["feature_missing"]:
            raise ValueError("Feature missingness changed after loading.")
        for name, expected in recomputed["values"].items():
            actual = record["features"][name]
            if expected is None:
                if actual is not None:
                    raise ValueError(f"Expected missing feature: {name}")
            else:
                if actual is None or not math.isfinite(actual):
                    raise ValueError(f"Expected finite feature: {name}")
                error = abs(actual - expected)
                max_feature_error = max(max_feature_error, error)
                if error > 1e-10:
                    raise ValueError(f"Stored feature disagrees with graph: {name}")
    return {"graphs_checked": len(library["graphs"]), "exact_witnesses_checked": exact_checked,
            "max_feature_recomputation_error": max_feature_error,
            "split_counts": {key: len(value) for key, value in splits.items()}}
