"""Create or verify a runtime snapshot or a complete, local B3 checkpoint.

The archive excludes Git history, notes, tests, credentials and environments.
Runtime snapshots include the selected frozen library under data/library.
B3 checkpoints retain the original relative paths of the selected run data;
canonical B1/B2 archives remain external, with explicit paths and hashes.
Creation/verification reads that library without recomputing exact answers,
features, QAOA values or diagnostics. Transfer and execution remain manual.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from qaoa_study.experiments import (
    _writer_lock, digest, load_attempts, read_batch, runtime_files, source_identity,
)
from qaoa_study.records import read_json, read_library


LIBRARY_FILES = ("graphs.parquet", "config.json", "generation.jsonl", "splits.json", "manifest.json")


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(output, payload, manifest, *, validate_inventory=None):
    """Write checked bytes once; paths are streamed so traces need not fit in RAM."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in sorted(payload.items()):
                if isinstance(value, Path):
                    archive.write(value, name)
                else:
                    archive.writestr(name, value)
            archive.writestr("snapshot.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            for name, expected in manifest["files"].items():
                with archive.open(name) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError(f"Files changed while capturing snapshot: {name}")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if validate_inventory is not None:
            validate_inventory()
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"snapshot_id": manifest["snapshot_id"], "source_id": manifest["source_id"],
            "library_id": manifest["library_id"], "files": len(payload),
            "archive_sha256": _file_hash(output)}


def _payload_path(root, name):
    """Resolve a manifest entry within its root, excluding traversal/symlinks."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Snapshot path must be relative and portable: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Snapshot path leaves its selected directory: {name}")
    return path


def create_snapshot(output, library_path, config_path, *, root=None):
    """Atomically publish one new ZIP with a manifest of captured file hashes."""
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output, library_path, config_path = Path(output), Path(library_path).resolve(), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    try:
        config_relative = config_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Configuration must be a runtime configs/*.json file inside the source root.") from error
    identity = source_identity(root)
    if config_relative not in identity["files"] or config_path.parent != root / "configs":
        raise ValueError("Configuration must be in the explicit runtime configs/*.json boundary.")
    read_json(config_path)
    library = read_library(library_path)
    payload = {}
    for path in runtime_files(root):
        name = path.relative_to(root).as_posix()
        payload[name] = _payload_path(root, name).read_bytes()
        if hashlib.sha256(payload[name]).hexdigest() != identity["files"][name]:
            raise ValueError("Working files changed while preparing the snapshot; retry after editing stops.")
    for name in LIBRARY_FILES:
        payload[f"data/library/{name}"] = _payload_path(library_path, name).read_bytes()
    # Confirm that the bytes captured after the read still match the frozen manifest.
    library_manifest = json.loads(payload["data/library/manifest.json"])
    if library_manifest["library_metadata"]["library_id"] != library["library_id"]:
        raise ValueError("Frozen library identity changed while preparing the snapshot.")
    for name in LIBRARY_FILES[:-1]:
        if hashlib.sha256(payload[f"data/library/{name}"]).hexdigest() != library_manifest["files"][name]:
            raise ValueError("Frozen library changed while preparing the snapshot.")
    manifest = {"snapshot_version": 1, "source_id": identity["source_id"],
                "source_files": identity["files"], "config_path": config_relative,
                "library_path": "data/library", "library_id": library["library_id"],
                "files": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(payload.items())}}
    manifest["snapshot_id"] = digest(manifest)
    return _publish(output, payload, manifest)


def _tree_files(root, directory):
    """Inventory the explicitly selected result tree, retaining unpublished bytes."""
    directory = _payload_path(root, directory)
    paths = sorted(directory.rglob("*")) if directory.is_dir() else [directory]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Checkpoint inputs cannot contain symlinks.")
    return {path.relative_to(root).as_posix(): path for path in paths
            if path.is_file() and path.name != "writer.lock"}


def _dependencies(value, external_root=None):
    """Validate portable external references; optionally verify the canonical bytes."""
    checkpoints, inputs = value["checkpoints"], value["comparison_inputs"]
    if value.get("dependency_version") != 1 or sorted(row["role"] for row in checkpoints) != ["B1", "B2"]:
        raise ValueError("Checkpoint dependencies require exactly canonical B1 and B2 entries.")
    if set(inputs) != {"b1_batch", "b1_attempts", "b2_cases"} or not inputs["b2_cases"]:
        raise ValueError("Checkpoint comparison inputs require B1 and explicit B2 cases.")
    paths = [inputs["b1_batch"], inputs["b1_attempts"]]
    for case in inputs["b2_cases"]:
        if len(case) != 2:
            raise ValueError("Each B2 case must give its batch and attempts paths.")
        paths.extend(case)
    for row in checkpoints:
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("Canonical checkpoint sha256 is invalid.")
        paths.append(row["path"])
    for name in paths:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("External dependency paths must be relative and portable.")
    if external_root is not None:
        base = Path(external_root).resolve()
        for row in checkpoints:
            if _file_hash(_payload_path(base, row["path"])) != row["sha256"]:
                raise ValueError(f"Canonical {row['role']} checkpoint checksum mismatch.")
    return inputs


def _checkpoint_state(root, manifest, external_root=None, *, dependencies=None):
    """Read only frozen B3 records; never initialize, optimize or rebuild references."""
    batch, library, tasks = read_batch(_payload_path(root, manifest["batch_path"]))
    if batch["config"].get("kind") != "b3" or batch["source_id"] != manifest["source_id"]:
        raise ValueError("Checkpoint requires a B3 batch with matching frozen runtime source.")
    if (library["library_id"] != manifest["library_id"] or
            (_payload_path(root, manifest["batch_path"]) / batch["library_path"]).resolve()
            != _payload_path(root, manifest["library_path"]).resolve()):
        raise ValueError("Checkpoint library path or identity disagrees with the batch.")
    config = read_json(_payload_path(root, manifest["config_path"]))
    if config.get("kind") != "b3" or any(batch["config"].get(key) != value for key, value in config.items()):
        raise ValueError("Checkpoint candidate configuration disagrees with the frozen batch.")
    audit = {}
    attempts, execution = load_attempts(batch, tasks, _payload_path(root, manifest["attempts_path"]),
                                        metadata_only=True, audit=audit)
    summary = read_json(_payload_path(root, manifest["summary_path"]))
    if (execution is None or summary.get("batch_id") != batch["batch_id"] or
            summary.get("execution_id") != execution["execution_id"] or
            summary.get("analysis_source") != batch["source"]):
        raise ValueError("Checkpoint summary, execution or analysis source mismatch.")
    if (any(summary.get(key) != batch["config"]["b3_inputs"][key]
            for key in ("initialization_set_id", "reference_binding_digest")) or
            summary["input_audits"]["B3"]["input_digest"] != digest(audit.get("file_hashes", {}))):
        raise ValueError("Checkpoint summary does not describe the frozen initialization/binding and retained attempts.")
    if dependencies is None:
        dependencies = read_json(_payload_path(root, "checkpoint/dependencies.json"))
    inputs = _dependencies(dependencies, external_root)
    if external_root is not None:
        external_root = Path(external_root).resolve()
        baseline, _, _ = read_batch(_payload_path(external_root, inputs["b1_batch"]))
        baseline_execution = read_json(_payload_path(external_root, inputs["b1_attempts"] + "/manifest.json"))
        cases = [(read_batch(_payload_path(external_root, batch_path))[0]["batch_id"],
                  read_json(_payload_path(external_root, attempts_path + "/manifest.json"))["execution_id"])
                 for batch_path, attempts_path in inputs["b2_cases"]]
        if (baseline["batch_id"] != summary["b1_batch_id"] or
                baseline_execution["execution_id"] != summary["b1_execution_id"] or
                sorted(cases) != sorted((row["batch_id"], row["execution_id"]) for row in summary["b2_cases"])):
            raise ValueError("External comparison batches disagree with the saved summary.")
    return {"batch_id": batch["batch_id"], "retained_completed_attempts": len(attempts),
            "summary_complete": summary.get("complete"), "external_dependencies_verified": external_root is not None}


REPRODUCE_SCRIPT = '''"""Rebuild the saved B3 summary read-only using separately retained B1/B2 data."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scripts.snapshot import verify_snapshot
from scripts.experiment import summarize_b3_batch
from qaoa_study.records import write_once_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--external-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
verify_snapshot(root, external_root=args.external_root)
manifest = json.loads((root / "snapshot.json").read_text())
inputs = json.loads((root / "checkpoint/dependencies.json").read_text())["comparison_inputs"]
external = args.external_root.resolve()
summary = summarize_b3_batch(root / manifest["batch_path"], root / manifest["attempts_path"],
    external / inputs["b1_batch"], external / inputs["b1_attempts"],
    [(external / batch, external / attempts) for batch, attempts in inputs["b2_cases"]])
write_once_json(args.output, summary)
print(json.dumps({"complete": summary["complete"], "batch_id": summary["batch_id"]}))
'''


def create_b3_checkpoint(output, library_path, config_path, batch, attempts_directory,
                         summary_path, dependencies_path, *, root=None, external_root=None):
    """Preserve complete B3 evidence; cite canonical B1/B2 archives without copying them.

    The dependency JSON contains dependency_version=1, checkpoints (one B1 and
    one B2 role/path/sha256), and comparison_inputs (b1_batch, b1_attempts,
    b2_cases pairs). All dependency paths are relative to external_root, default
    root. Run inputs must lie inside root so batch/library relative paths survive.
    """
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    external_root = Path(external_root or root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    locations = {name: Path(path).resolve().relative_to(root).as_posix() for name, path in
                 (("library_path", library_path), ("config_path", config_path), ("batch_path", batch),
                  ("attempts_path", attempts_directory), ("summary_path", summary_path))}
    summary_tree = locations["summary_path"]
    if _payload_path(root, summary_tree).is_dir():
        locations["summary_path"] += "/summary.json"
    trees = [locations["batch_path"], locations["attempts_path"], summary_tree]
    if any(name == "." for name in trees):
        raise ValueError("Select individual B3 result directories, not the whole source root.")
    if any(output.is_relative_to(_payload_path(root, name)) for name in trees):
        raise ValueError("Checkpoint output must be outside its selected input trees.")
    identity = source_identity(root)
    dependencies = read_json(dependencies_path)
    _dependencies(dependencies)
    with _writer_lock(_payload_path(root, locations["attempts_path"])):
        payload = {name: _payload_path(root, name) for name in identity["files"]}
        payload[locations["config_path"]] = _payload_path(root, locations["config_path"])
        payload.update({locations["library_path"] + "/" + name: _payload_path(root, locations["library_path"] + "/" + name)
                        for name in LIBRARY_FILES})
        tree_payload = {name: path for directory in trees for name, path in _tree_files(root, directory).items()}
        payload.update(tree_payload)
        canonical = {_payload_path(external_root, row["path"]).resolve() for row in dependencies["checkpoints"]}
        if any(path.resolve() in canonical for path in payload.values()):
            raise ValueError("Canonical B1/B2 archives must remain external to the B3 checkpoint.")
        payload["checkpoint/dependencies.json"] = (json.dumps(dependencies, indent=2, allow_nan=False) + "\n").encode()
        payload["checkpoint/reproduce.py"] = REPRODUCE_SCRIPT.encode()
        library = read_library(_payload_path(root, locations["library_path"]))
        manifest = {"snapshot_version": 2, "kind": "b3-checkpoint-v1", **locations,
                    "source_id": identity["source_id"], "source_files": identity["files"],
                    "library_id": library["library_id"], "result_trees": trees,
                    "files": {name: _file_hash(value) if isinstance(value, Path) else hashlib.sha256(value).hexdigest()
                              for name, value in sorted(payload.items())}}
        if any(manifest["files"][name] != value for name, value in identity["files"].items()):
            raise ValueError("Working source changed while preparing checkpoint.")
        _checkpoint_state(root, manifest, external_root, dependencies=dependencies)
        def validate_inventory():
            if tree_payload != {name: path for directory in trees for name, path in _tree_files(root, directory).items()}:
                raise ValueError("B3 result inventory changed while preparing checkpoint.")
        manifest["snapshot_id"] = digest(manifest)
        return _publish(output, payload, manifest, validate_inventory=validate_inventory)


def verify_snapshot(directory, *, external_root=None):
    """Verify extracted payload/identity and frozen library without recomputation.

    Hashes detect content changes, not the identity of a sender; compare the ZIP
    hash with the independently retained local value before extracting it.
    """
    directory = Path(directory).resolve()
    manifest = read_json(directory / "snapshot.json")
    if manifest.get("snapshot_version") not in (1, 2):
        raise ValueError("Unsupported snapshot version.")
    if digest({key: value for key, value in manifest.items() if key != "snapshot_id"}) != manifest["snapshot_id"]:
        raise ValueError("Snapshot manifest checksum mismatch.")
    expected_runtime = {path.relative_to(directory).as_posix() for path in runtime_files(directory)}
    expected_library = {manifest["library_path"] + "/" + name for name in LIBRARY_FILES}
    expected = expected_runtime | expected_library
    if manifest["snapshot_version"] == 2:
        if manifest.get("kind") != "b3-checkpoint-v1":
            raise ValueError("Unsupported result checkpoint kind.")
        expected |= {name for tree in manifest["result_trees"] for name in _tree_files(directory, tree)}
        expected |= {manifest["config_path"], "checkpoint/dependencies.json", "checkpoint/reproduce.py"}
    if set(manifest["files"]) != expected:
        raise ValueError("Snapshot payload does not match the runtime/library boundary.")
    if manifest["snapshot_version"] == 1 and (manifest["config_path"] not in expected_runtime or
            PurePosixPath(manifest["config_path"]).parent != PurePosixPath("configs")):
        raise ValueError("Snapshot config is not inside its runtime boundary.")
    if manifest["snapshot_version"] == 1 and manifest["library_path"] != "data/library":
        raise ValueError("Unexpected frozen library location.")
    for name, expected in manifest["files"].items():
        if _file_hash(_payload_path(directory, name)) != expected:
            raise ValueError(f"Snapshot payload checksum mismatch: {name}")
    actual_source = source_identity(directory)
    if actual_source != {"source_id": manifest["source_id"], "files": manifest["source_files"]}:
        raise ValueError("Snapshot working source identity mismatch.")
    library = read_library(_payload_path(directory, manifest["library_path"]))
    if library["library_id"] != manifest["library_id"]:
        raise ValueError("Snapshot library identity mismatch.")
    result = {"snapshot_id": manifest["snapshot_id"], "source_id": manifest["source_id"],
              "library_id": manifest["library_id"], "files_verified": len(manifest["files"])}
    if manifest["snapshot_version"] == 2:
        result.update(_checkpoint_state(directory, manifest, external_root))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an already extracted snapshot directory.")
    parser.add_argument("--checkpoint", action="store_true", help="Capture complete B3 results and external dependency declarations.")
    for name in ("batch", "attempts", "summary", "dependencies", "external-root"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        if args.checkpoint or any(value is not None for value in
                (args.output, args.library, args.config, args.batch, args.attempts, args.summary, args.dependencies)):
            parser.error("--verify is separate from snapshot creation options")
        result = verify_snapshot(args.verify, external_root=args.external_root)
    else:
        if any(value is None for value in (args.output, args.library, args.config)):
            parser.error("snapshot creation requires --output, --library and --config")
        if args.checkpoint:
            if any(value is None for value in (args.batch, args.attempts, args.summary, args.dependencies)):
                parser.error("B3 checkpoint requires --batch, --attempts, --summary and --dependencies")
            result = create_b3_checkpoint(args.output, args.library, args.config, args.batch, args.attempts,
                                         args.summary, args.dependencies, external_root=args.external_root)
        else:
            if any(value is not None for value in (args.batch, args.attempts, args.summary, args.dependencies, args.external_root)):
                parser.error("B3 result options require --checkpoint")
            result = create_snapshot(args.output, args.library, args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
