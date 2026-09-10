"""Small, explicit run records and post-hoc threshold evaluation."""

import json
import math
from pathlib import Path
import copy
import hashlib
from importlib.metadata import version
import platform
import os
import tempfile
import io
import re
import tarfile
import gzip


EXECUTION_FAILURES = frozenset(("numerical_error", "evaluation_error", "optimizer_failed", "program_error"))


def execution_failed(result: dict) -> bool:
    """Separate execution faults from valid budget/iteration/line-search stops."""
    return result["stop_reason"] in EXECUTION_FAILURES


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
    invalid_run = execution_failed(result)
    return {"C_ref": C_ref, "epsilon": epsilon, "first_hit": first_hit,
            "hit": first_hit is not None,
            "terminal_success": bool(not invalid_run and final is not None and
                                     math.isfinite(final) and final >= threshold),
            "reference_exceeded": any(row["C"] is not None and math.isfinite(row["C"]) and
                                      row["C"] > C_ref for row in result["trace"])}


def save_run(path, result: dict) -> None:
    """Publish a completed, checksummed attempt without replacing any record.

    Retrying requires a new attempt ID/path. Concurrent writers may race, but
    exactly one can publish the destination; all others get FileExistsError.
    """
    if not result.get("run_completed"):
        raise ValueError("Only a completed attempt can be saved as a run record.")
    checksum = hashlib.sha256(_canonical_json(result)).hexdigest()
    write_once_json(path, {**result, "_run_file_version": 1,
                          "_integrity": {"algorithm": "sha256", "value": checksum}})


def read_run(path, *, require_integrity=False, input_hashes=None, contents=None) -> dict:
    """Read a completed attempt; interrupted work must restart as a new attempt."""
    stored = read_json(path, input_hashes=input_hashes, contents=contents)
    if require_integrity and "_integrity" not in stored:
        raise ValueError("Batch resume requires checksummed new attempts, not legacy records.")
    if "_run_file_version" in stored or "_integrity" in stored:
        if stored.get("_run_file_version") != 1 or stored.get("_integrity", {}).get("algorithm") != "sha256":
            raise ValueError("Unsupported or incomplete run envelope.")
        result = {key: value for key, value in stored.items() if key not in ("_run_file_version", "_integrity")}
        if hashlib.sha256(_canonical_json(result)).hexdigest() != stored["_integrity"].get("value"):
            raise ValueError("Run record checksum mismatch.")
    else:
        # Historical records remain readable; their original bytes are preserved.
        result = stored
    if not result.get("run_completed"):
        raise ValueError("Incomplete run record.")
    return result


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_once_json(path, value) -> None:
    """Publish immutable JSON atomically; fail if the destination already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create-if-absent on the same filesystem. No replace fallback:
        # filesystems without hard links must report their unsupported operation.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path, *, input_hashes=None, contents=None):
    """Load UTF-8 JSON from a file or already-read archive member bytes."""
    path = Path(path)
    content = path.read_bytes() if contents is None else contents
    if input_hashes is not None:
        input_hashes[path.name] = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode("utf-8"))


def _bundle_name(name):
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", name) is not None or record_temporary(name)


def record_temporary(name):
    """Identify unpublished write_once_json bytes retained after an interruption."""
    return re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9_.-]*\.json\.[A-Za-z0-9_-]{8}\.tmp", name) is not None


def read_record_bundle(archive):
    """Read a sealed group without extraction; verify every exact record byte."""
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError(f"Expected a regular record archive: {archive}")
    members = {}
    with gzip.open(archive, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as bundle:
        for entry in bundle:
            if not entry.isfile() or not _bundle_name(entry.name) or entry.name in members:
                raise ValueError(f"Unsafe or duplicate record archive member: {entry.name}")
            members[entry.name] = bundle.extractfile(entry).read()
        # tarfile may stop at its end marker without checking the gzip trailer.
        while stream.read(1 << 20):
            pass
    manifest = json.loads(members.pop("bundle-manifest.json", b"{}").decode("utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
            not isinstance(manifest.get("metadata"), dict) or not members or
            manifest.get("files") != {name: hashlib.sha256(value).hexdigest()
                                      for name, value in members.items()}):
        raise ValueError("Incomplete record archive or member checksum mismatch.")
    return manifest["metadata"], members


def _active_bundle_files(directory):
    """Review regular records and recognized unpublished bytes before compaction."""
    if directory.is_symlink():
        raise ValueError(f"Record directory cannot be a symlink: {directory}")
    members = {}
    if directory.exists():
        for path in directory.iterdir():
            if (path.is_symlink() or not path.is_file() or not _bundle_name(path.name) or
                    path.name == "bundle-manifest.json" or path.resolve().parent != directory.resolve()):
                raise ValueError(f"Unexpected active record path: {path}")
            members[path.name] = path.read_bytes()
    return members


def seal_record_bundle(directory, archive, metadata):
    """Seal a caller-validated complete group, then remove verified duplicates.

    Retain failures, orphan start receipts and unpublished temporary bytes.
    The caller establishes semantic completeness and excludes concurrent writers.
    Rerunning after an interrupted cleanup accepts an exact remaining subset.
    """
    directory, archive = Path(directory), Path(archive)
    if archive.resolve().is_relative_to(directory.resolve()):
        raise ValueError("The sealed archive must be outside its active directory.")
    if not isinstance(metadata, dict):
        raise ValueError("Record archive metadata must be a JSON object.")
    members = _active_bundle_files(directory)
    if not archive.exists():
        if not members:
            raise ValueError("Cannot seal an empty or missing record directory.")
        manifest = {"schema_version": 1, "metadata": metadata,
                    "files": {name: hashlib.sha256(value).hexdigest() for name, value in members.items()}}
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=archive.parent, prefix=f".{archive.name}.",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
                    for name, value in sorted({**members, "bundle-manifest.json": _canonical_json(manifest)}.items()):
                        entry = tarfile.TarInfo(name)
                        entry.size = len(value)
                        bundle.addfile(entry, io.BytesIO(value))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, archive)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    saved_metadata, saved_members = read_record_bundle(archive)
    if _canonical_json(saved_metadata) != _canonical_json(metadata):
        raise ValueError("Existing record archive metadata conflicts with this group.")
    if any(saved_members.get(name) != value for name, value in members.items()):
        raise ValueError("Active records conflict with the sealed archive; nothing was removed.")
    del members
    remaining = _active_bundle_files(directory)
    if any(saved_members.get(name) != value for name, value in remaining.items()):
        raise ValueError("Active records conflict with the sealed archive; nothing was removed.")
    for name, value in remaining.items():
        path = directory / name
        if path.is_symlink() or path.resolve().parent != directory.resolve() or path.read_bytes() != value:
            raise ValueError(f"Active record changed during compaction: {path}")
        path.unlink()
    if directory.exists():
        directory.rmdir()
    return saved_metadata


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
    write_once_json(directory / "config.json", library["config"])
    write_once_json(directory / "splits.json", splits)
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
    write_once_json(directory / "manifest.json", manifest)


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
    """Audit topology, stored cut witnesses, features and split consistency.

    A witness proves its reported cut value, not optimality of C_star. This
    read-only audit never solves MaxCut; independent solver tests validate the
    exact enumerator separately.
    """
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
