"""Create or verify a minimal runtime snapshot of the actual local working code.

The archive excludes Git history, notes, tests, credentials and environments.
Only the explicitly selected frozen library is included, under data/library.
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

from qaoa_study.experiments import digest, runtime_files, source_identity
from qaoa_study.records import read_json, read_library


LIBRARY_FILES = ("graphs.parquet", "config.json", "generation.jsonl", "splits.json", "manifest.json")


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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in sorted(payload.items()):
                archive.writestr(name, value)
            archive.writestr("snapshot.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"snapshot_id": manifest["snapshot_id"], "source_id": manifest["source_id"],
            "library_id": manifest["library_id"], "files": len(payload),
            "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def verify_snapshot(directory):
    """Verify extracted payload/identity and frozen library without recomputation.

    Hashes detect content changes, not the identity of a sender; compare the ZIP
    hash with the independently retained local value before extracting it.
    """
    directory = Path(directory).resolve()
    manifest = read_json(directory / "snapshot.json")
    if manifest.get("snapshot_version") != 1:
        raise ValueError("Unsupported snapshot version.")
    if digest({key: value for key, value in manifest.items() if key != "snapshot_id"}) != manifest["snapshot_id"]:
        raise ValueError("Snapshot manifest checksum mismatch.")
    expected_runtime = {path.relative_to(directory).as_posix() for path in runtime_files(directory)}
    expected_library = {f"data/library/{name}" for name in LIBRARY_FILES}
    if set(manifest["files"]) != expected_runtime | expected_library:
        raise ValueError("Snapshot payload does not match the runtime/library boundary.")
    if manifest["config_path"] not in expected_runtime or PurePosixPath(manifest["config_path"]).parent != PurePosixPath("configs"):
        raise ValueError("Snapshot config is not inside its runtime boundary.")
    if manifest["library_path"] != "data/library":
        raise ValueError("Unexpected frozen library location.")
    for name, expected in manifest["files"].items():
        if hashlib.sha256(_payload_path(directory, name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot payload checksum mismatch: {name}")
    actual_source = source_identity(directory)
    if actual_source != {"source_id": manifest["source_id"], "files": manifest["source_files"]}:
        raise ValueError("Snapshot working source identity mismatch.")
    library = read_library(directory / "data/library")
    if library["library_id"] != manifest["library_id"]:
        raise ValueError("Snapshot library identity mismatch.")
    return {"snapshot_id": manifest["snapshot_id"], "source_id": manifest["source_id"],
            "library_id": manifest["library_id"], "files_verified": len(manifest["files"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an already extracted snapshot directory.")
    args = parser.parse_args(argv)
    if args.verify is not None:
        if any(value is not None for value in (args.output, args.library, args.config)):
            parser.error("--verify is separate from snapshot creation options")
        result = verify_snapshot(args.verify)
    else:
        if any(value is None for value in (args.output, args.library, args.config)):
            parser.error("snapshot creation requires --output, --library and --config")
        result = create_snapshot(args.output, args.library, args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
