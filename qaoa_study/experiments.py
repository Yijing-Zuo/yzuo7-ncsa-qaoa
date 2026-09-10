"""Frozen task lists, independent pools and immutable whole-attempt execution.

Planning and aggregation never call a quantum device or exact solver. Execution
is explicit; each worker reuses a QNode per (graph, depth). No scheduler or
optimizer checkpoint is implemented here.
"""

from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distributions, version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from uuid import uuid4

import numpy as np
import pennylane as qml

from .graphs import SPLIT_VERSION, graph_from_record, graph_split_coverage
from .optimize import OptimizerSettings, failure_record, optimize_qaoa, random_angles
from .qaoa import make_qaoa, p1_expectation_grid, qaoa_loss_and_gradient
from .records import (execution_failed, read_json, read_library, read_run, save_run, write_once_json,
                      read_record_bundle, record_temporary, seal_record_bundle)


TASK_VERSION = "plan-a-tasks-v1"
ATTEMPT_RULE = "first nonfault attempt ordered by attempt_number,started_utc,attempt_id; else first fault"
ENTRY_FILES = ("build_library.py", "experiment.py", "verify_backend.py", "snapshot.py")
PARTITIONED_LAYOUT = "graph-role-v1"


def digest(value):
    """Portable content identity; it contains no execution order or device."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def runtime_files(root):
    """The code/configuration snapshot boundary; no notes, tests or prior data."""
    root = Path(root)
    paths = [*root.glob("qaoa_study/*.py"), *root.glob("configs/*.json")]
    paths += [root / "scripts" / name for name in ENTRY_FILES]
    paths += [root / name for name in ("pyproject.toml", "requirements-lock.txt", "README.md",
                                      "jobs/experiment.slurm")]
    return sorted(p for p in paths if p.is_file())


def source_identity(root=None):
    """Hash actual working files, including uncommitted changes, not just HEAD."""
    root = Path(root or Path(__file__).resolve().parents[1])
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime_files(root)}
    return {"source_id": digest(hashes), "files": hashes}


def environment(backend):
    """Read package/GPU metadata without creating a quantum or CUDA device.

    Device locations belong to individual attempts, not shared execution identity.
    Installed wheel versions do not prove which external CUDA libraries will load.
    """
    packages = {}
    gpu_packages = ("pennylane-lightning-gpu", "scipy-openblas32", "custatevec-cu12", "nvidia-nvjitlink-cu12",
                    "nvidia-cusparse-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12")
    for name in ("pennylane", "pennylane-lightning", "numpy", "scipy", "networkx",
                 "autograd", "pyarrow", *gpu_packages):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    for distribution in distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name.startswith(("nvidia-", "custatevec-")):
            packages[name] = distribution.version
    identity = {"hostname": platform.node(), "gpu_inventory": None,
                "visibility": {key: os.environ.get(key) for key in
                    ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
                     "SLURM_JOB_ID", "SLURM_STEP_GPUS")}}
    gpu = None
    if backend == "lightning.gpu":
        gpu = {"model": None, "driver_version": None, "unknown": []}
        try:
            query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, check=True, timeout=10)
            inventory = [dict(zip(("index", "uuid", "name", "driver_version"),
                                 (value.strip() for value in row), strict=True))
                         for row in csv.reader(query.stdout.splitlines()) if row]
            identity["gpu_inventory"] = inventory
            visible = identity["visibility"]["CUDA_VISIBLE_DEVICES"]
            selected = inventory
            if visible is not None:
                selected = []
                for token in visible.split(","):
                    token = token.strip()
                    if token.isdecimal() and len({(row["name"], row["driver_version"]) for row in inventory}) != 1:
                        raise ValueError("Numeric CUDA ordinals are ambiguous on heterogeneous GPUs; use GPU UUIDs.")
                    matches = [row for row in inventory if row["index"] == token
                               or (token.startswith("GPU-") and row["uuid"].startswith(token))]
                    if len(matches) != 1:
                        raise ValueError("CUDA_VISIBLE_DEVICES cannot be resolved from the queried inventory.")
                    selected.extend(matches)
            profiles = {(row["name"], row["driver_version"]) for row in selected}
            if len(profiles) != 1:
                raise ValueError("No single GPU model/driver profile is established for visible devices.")
            model, driver = profiles.pop()
            if not model or not driver or any(value in ("N/A", "[N/A]") for value in (model, driver)):
                raise ValueError("GPU model or driver version is unavailable.")
            gpu.update(model=model, driver_version=driver)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            gpu["unknown"].append("gpu_model_or_driver")
            identity["gpu_query_error"] = f"{type(error).__name__}: {error}"
        gpu["unknown"] += [name for name in gpu_packages if packages[name] is None]
    return {"environment_version": "compute-environment-v2",
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "backend": backend, "shots": None, "interface": "autograd",
            "diff_method": "adjoint", "device_vjp": True,
            "angle_dtype": "float64", "state_dtype": "complex128",
            "gpu": gpu, "device_identity": identity,
            "cuda_version_scope": "installed_distribution_metadata; loaded_libraries_unverified",
            "packages": packages, "thread_environment": {key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def validate_config(config):
    """Validate the small study configuration once at the planning boundary."""
    if config["depths"] != [1, 2]:
        raise ValueError("Plan A requires p=1 and p=2.")
    if config["epsilon"] != 0.5 or config["reference"]["p2_candidate"] != "best_seen_valid_execution":
        raise ValueError("Reference and success definitions must follow this protocol version.")
    for count in (config["tier1_restarts"], config["reference"]["p2_restarts"],
                  config["evaluation_restarts"], config["gradient_samples"], config["max_attempts"]):
        if type(count) is not int or count < 1:
            raise ValueError("Pool/sample/attempt counts must be positive integers.")
    for p in config["depths"]:
        OptimizerSettings(**config["optimizer"][str(p)])
    grid = config["reference"]["p1_grid"]
    if any(type(grid[key]) is not int or grid[key] < 2 for key in
           ("gamma_points", "beta_points", "refinement_factor")) or grid["chunk_size"] < 1:
        raise ValueError("Use positive grid dimensions and integer refinement >=2.")
    if not math.isfinite(grid["refinement_tolerance"]) or grid["refinement_tolerance"] < 0:
        raise ValueError("Grid refinement tolerance must be finite and nonnegative.")


def design_counts(graph_count, tier2_count, config):
    """Count the requested design without constructing graphs/tasks or circuits."""
    return {"tier1": graph_count * 2 * config["tier1_restarts"],
            "reference_grid": tier2_count,
            "reference_optimization": tier2_count * config["reference"]["p2_restarts"],
            "evaluation": tier2_count * 2 * config["evaluation_restarts"],
            "gradient_diagnostic": tier2_count * 2 * config["gradient_samples"]}


def build_tasks(library, config):
    """Prepare B1/grid/diagnostic tasks from a frozen library, never regenerate it."""
    validate_config(config)
    tasks = []
    for record in sorted(library["graphs"], key=lambda row: row["iso_class_id"]):
        for p in config["depths"]:
            specifications = [("tier1", "optimization", config["tier1_restarts"])]
            if record["tier"] == 2:
                specifications += [("reference", "p1_grid" if p == 1 else "optimization",
                                    1 if p == 1 else config["reference"]["p2_restarts"]),
                                   ("evaluation", "optimization", config["evaluation_restarts"]),
                                   ("gradient_diagnostic", "gradient", config["gradient_samples"])]
            for role, kind, count in specifications:
                for restart_id in range(count):
                    stream = {"namespace": config["seed_namespace"], "master_seed": config["seed"],
                              "library_id": library["library_id"], "iso_class_id": record["iso_class_id"],
                              "p": p, "experiment_role": role, "restart_id": restart_id}
                    seed = int(digest(stream)[:32], 16)
                    budget = (config["reference"]["p1_grid"] if kind == "p1_grid" else
                              {"max_evaluations": 1} if kind == "gradient" else config["optimizer"][str(p)])
                    task = {"task_version": TASK_VERSION, "library_id": library["library_id"],
                            "iso_class_id": record["iso_class_id"], "n": record["n"], "p": p,
                            "graph_split": record["graph_split"], "kind": kind, "experiment_role": role,
                            "pool": role if role in ("reference", "evaluation") else None,
                            "method": "analytic_grid" if kind == "p1_grid" else "uniform" if kind == "gradient" else "B1",
                            "restart_id": restart_id, "seed": seed,
                            "theta0": None if kind == "p1_grid" else random_angles(p, seed).tolist(),
                            "budget": budget, "protocol_version": config["protocol_version"]}
                    task["task_id"] = digest(task)
                    tasks.append(task)
    if len({task["seed"] for task in tasks}) != len(tasks):
        raise ValueError("Unexpected seed collision; do not run a non-disjoint plan.")
    return sorted(tasks, key=lambda task: task["task_id"])


def library_issues(library, config):
    """Check annotations/design presence, not exact optimality or QAOA scores."""
    issues = []
    for row in library["graphs"]:
        if row.get("exact_status") != "computed" or row.get("C_star") is None or row.get("optimal_cut_side") is None:
            issues.append({"iso_class_id": row["iso_class_id"], "missing": "exact_MaxCut_annotation"})
        if not row.get("features") or "feature_missing" not in row:
            issues.append({"iso_class_id": row["iso_class_id"], "missing": "features_annotation"})
    expected = config["expected_library"]
    actual = {"graphs": len(library["graphs"]), "tier2": sum(r["tier"] == 2 for r in library["graphs"]),
              "training": sum(r["graph_split"] == "training" for r in library["graphs"]),
              "evaluation": sum(r["graph_split"] == "evaluation" for r in library["graphs"])}
    for key, target in expected.items():
        if actual[key] != target:
            issues.append({"design": key, "expected": target, "actual": actual[key]})
    if config.get("require_full_design", False) or config.get("design"):
        design = config.get("design", {"sizes": [12, 16, 20, 24],
                            "families": ["regular", "er", "ba", "sbm"], "degree_bands": [3, 5]})
        cells = {f"n{n}:{family}:k{band}" for n in design["sizes"]
                 for family in design["families"] for band in design["degree_bands"]}
        present = {r["representative_cell"] for r in library["graphs"]}
        if cells - present:
            issues.append({"missing_cells": sorted(cells - present)})
        if present - cells:
            issues.append({"unexpected_cells": sorted(present - cells)})
        if library.get("split_summary", {}).get("split_version") != SPLIT_VERSION:
            issues.append({"missing": "seeded_v2_split_revision"})
    if config.get("require_full_design", False) or config.get("require_split_coverage", False):
        if graph_split_coverage(library["graphs"], library.get("cells"))["smoke_test"]:
            issues.append({"missing": "training_evaluation_cell_coverage", "status": "smoke_test_only"})
    if any(cell.get("shortfall", 0) for cell in library.get("cells", [])):
        issues.append({"missing": "generator_cell_quotas", "status": "population_unknown"})
    return issues


def save_batch(directory, library_path, config):
    """Freeze task JSONL and metadata. This operation performs no scientific computation."""
    directory, library_path = Path(directory), Path(library_path).resolve()
    library = read_library(library_path)
    tasks = build_tasks(library, config)
    content = "".join(json.dumps(task, sort_keys=True, allow_nan=False) + "\n" for task in tasks).encode()
    source = source_identity()
    identity = {"library_id": library["library_id"], "config": config, "source_id": source["source_id"],
                "tasks_sha256": hashlib.sha256(content).hexdigest()}
    manifest = {**identity, "batch_id": digest(identity), "source": source,
                "library_path": os.path.relpath(library_path, directory.resolve()),
                "library_manifest_sha256": hashlib.sha256((library_path / "manifest.json").read_bytes()).hexdigest(),
                "issues": library_issues(library, config), "task_count": len(tasks),
                "counts": dict(Counter(task["experiment_role"] for task in tasks)),
                "coverage": graph_split_coverage(library["graphs"], library.get("cells")),
                "attempt_selection_rule": ATTEMPT_RULE}
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tasks.jsonl").write_bytes(content)
    write_once_json(directory / "manifest.json", manifest)
    return manifest


def read_batch(directory):
    """Check the frozen plan and library hashes without annotation/re-optimization."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    content = (directory / "tasks.jsonl").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["tasks_sha256"]:
        raise ValueError("Task list checksum mismatch.")
    identity = {key: manifest[key] for key in ("library_id", "config", "source_id", "tasks_sha256")}
    if digest(identity) != manifest["batch_id"]:
        raise ValueError("Batch identity mismatch.")
    if digest(manifest["source"]["files"]) != manifest["source_id"] or manifest["source"]["source_id"] != manifest["source_id"]:
        raise ValueError("Actual source hashes do not match the batch source identity.")
    library_path = (directory / manifest["library_path"]).resolve()
    if hashlib.sha256((library_path / "manifest.json").read_bytes()).hexdigest() != manifest["library_manifest_sha256"]:
        raise ValueError("Frozen library manifest changed.")
    library = read_library(library_path)
    if library["library_id"] != manifest["library_id"]:
        raise ValueError("Library identity mismatch.")
    tasks = [json.loads(line) for line in content.splitlines()]
    if len(tasks) != manifest["task_count"] or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Task counts or uniqueness changed.")
    for task in tasks:
        if digest({k: v for k, v in task.items() if k != "task_id"}) != task["task_id"]:
            raise ValueError("Task identity mismatch.")
    derived = {"issues": library_issues(library, manifest["config"]),
               "coverage": graph_split_coverage(library["graphs"], library.get("cells")),
               "counts": dict(Counter(task["experiment_role"] for task in tasks)),
               "attempt_selection_rule": ATTEMPT_RULE}
    if any(manifest.get(key) != value for key, value in derived.items()):
        raise ValueError("Batch completeness, coverage, counts or attempt rule changed.")
    return manifest, library, tasks


def shard_tasks(tasks, index=0, count=1):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    return [task for task in tasks if int(task["task_id"], 16) % count == index]


def validate_attempt(task, attempt, manifest, execution):
    """Check task/protocol/settings provenance and trace consistency on resume."""
    required = {key: task[key] for key in ("task_id", "library_id", "iso_class_id", "p", "graph_split",
                                          "kind", "experiment_role", "pool", "method", "restart_id",
                                          "seed", "theta0", "protocol_version", "budget")}
    required.update(batch_id=manifest["batch_id"], execution_id=execution["execution_id"], task_digest=digest(task))
    if any(attempt.get(key) != value for key, value in required.items()):
        raise ValueError("Attempt task, protocol, execution settings or initial angles disagree.")
    if not attempt.get("run_completed") or not attempt.get("attempt_id"):
        raise ValueError("Incomplete attempt.")
    if "invalid_result" in attempt:
        counts, unavailable = _recoverable_counts(attempt["invalid_result"], record_version=attempt.get("record_version", 2))
        if (attempt["stop_reason"] != "program_error" or attempt.get("trace") != []
                or any(attempt.get(key) is not None for key in
                       ("C_final", "theta_final", "final_call_id", "best_seen", "best_theta", "best_call_id"))
                or attempt.get("counts") != counts or attempt.get("cost_unavailable") != unavailable
                or attempt.get("cost_status") != "partially_verified_after_record_validation_failure"):
            raise ValueError("Invalid-result failure evidence or cost accounting changed.")
        return
    if attempt.get("record_version") == 3:
        samples = attempt["trace"] if task["kind"] == "optimization" else attempt.get("gradient_samples", [])
        for name, raw_name in (("device_executions", "executions"),
                              ("device_derivatives", "derivatives"), ("device_vjps", "vjps")):
            work = [row.get("device_work", {}).get(raw_name, 0) for row in samples]
            if (any(type(value) is not int or value < 0 for value in work)
                    or type(attempt["counts"].get(name)) is not int
                    or attempt["counts"][name] != sum(work)):
                raise ValueError("Device work counters disagree with the preserved Tracker totals.")
    if task["kind"] == "optimization":
        trace = attempt["trace"]
        if (attempt["settings"] != task["budget"] or attempt["counts"]["objective_calls"] != len(trace)
                or len(trace) > task["budget"]["max_evaluations"]
                or attempt["counts"]["gradient_requests"] != len(trace)):
            raise ValueError("Optimization settings/call count mismatch.")
        if [row["call_id"] for row in trace] != list(range(1, len(trace) + 1)):
            raise ValueError("Trace call IDs are not contiguous.")
        accepted = [row for row in trace if row["accepted"] and row["valid"]]
        last = accepted[-1] if accepted else None
        if attempt["C_final"] != (last["C"] if last else None) or attempt["theta_final"] != (last["theta"] if last else None):
            raise ValueError("Final point is not the last accepted valid trace point.")
        if attempt["final_call_id"] != (last["call_id"] if last else None):
            raise ValueError("Final call identity does not match the accepted point.")
        finite = [row for row in trace if row["C"] is not None and math.isfinite(row["C"])]
        best = max(finite, key=lambda row: row["C"], default=None)
        if attempt["best_seen"] != (best["C"] if best else None) or attempt["best_call_id"] != (best["call_id"] if best else None):
            raise ValueError("best_seen disagrees with the raw trace.")
        if attempt["best_theta"] != (best["theta"] if best else None):
            raise ValueError("Best angles do not match the selected trace point.")


def _recoverable_counts(invalid_result, *, record_version=2):
    """Retain only raw counters independently consistent with their trace.

    A malformed result is evidence, not a valid optimization. Unknown work is
    null; even when some counters can be checked, overall reliability is marked.
    """
    names = ("objective_calls", "gradient_requests", "device_executions", "device_derivatives", "iterations")
    if record_version >= 3:
        names += ("device_vjps",)
    counts, unavailable = dict.fromkeys(names), {}
    trace = invalid_result.get("trace") if isinstance(invalid_result, dict) else None
    expected = {}
    if isinstance(trace, list) and all(isinstance(row, dict) for row in trace):
        expected.update(objective_calls=len(trace), gradient_requests=len(trace))
        for name, tracker_name in (("device_executions", "executions"),
                                   ("device_derivatives", "derivatives"), ("device_vjps", "vjps")):
            work = [row.get("device_work", {}).get(tracker_name, 0)
                    for row in trace if isinstance(row.get("device_work"), dict)]
            if len(work) == len(trace) and all(type(value) is int and value >= 0 for value in work):
                expected[name] = sum(work)
        if all(type(row.get("accepted")) is bool and type(row.get("valid")) is bool for row in trace):
            expected["iterations"] = max(0, sum(row["accepted"] and row["valid"] for row in trace) - 1)
    raw_counts = invalid_result.get("counts", {}) if isinstance(invalid_result, dict) else {}
    for name in names:
        value = raw_counts.get(name) if isinstance(raw_counts, dict) else None
        if type(value) is int and value >= 0 and expected.get(name) == value:
            counts[name] = value
        else:
            unavailable[name] = "Raw counter cannot be verified against the preserved invalid result trace."
    return counts, unavailable


def _invalid_result_failure(result, task, error, elapsed_seconds):
    """Preserve a post-computation program failure before returning to the CLI."""
    try:
        json.dumps(result, allow_nan=False)
        evidence = result
    except (TypeError, ValueError):
        evidence = {"serialization": "python_repr", "value": repr(result),
                    "limitation": "Nonstandard objects may abbreviate their own repr; counters are unavailable."}
    settings = OptimizerSettings(**task["budget"]) if task["kind"] == "optimization" else OptimizerSettings()
    failed = failure_record(task["theta0"] or [], settings, error, p=task["p"], stop_reason="program_error")
    counts, unavailable = _recoverable_counts(evidence, record_version=failed["record_version"])
    failed.update(invalid_result=evidence, counts=counts, elapsed_seconds=elapsed_seconds,
                  cost_status="partially_verified_after_record_validation_failure", cost_unavailable=unavailable)
    return failed


def _attempt_metadata(attempt):
    """Discard already-validated samples, including the preserved failed raw result."""
    return {key: value for key, value in attempt.items() if key not in ("trace", "gradient_samples", "invalid_result")}


def _validate_execution(manifest, execution):
    if execution["batch_id"] != manifest["batch_id"] or digest({k: v for k, v in execution.items()
            if k != "execution_id"}) != execution["execution_id"]:
        raise ValueError("Execution manifest identity mismatch.")
    if (execution["source"]["source_id"] != manifest["source_id"] or execution["config"] != manifest["config"]
            or digest(execution["source"]["files"]) != manifest["source_id"]):
        raise ValueError("Execution source/configuration does not match the frozen batch.")


def _record_directory(directory):
    """Support hashed record paths beyond Windows MAX_PATH; keep POSIX paths unchanged."""
    path = os.path.abspath(directory) if os.name == "nt" else str(directory)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\UNC\\" + path[2:] if path.startswith("\\\\") else "\\\\?\\" + path
    return Path(path)


def _load_flat_attempts(manifest, tasks, directory, *, metadata_only=False, paths=None, audit=None,
                        execution=None, contents=None):
    """Validate every completed record; corruption is an error, not a missing run."""
    directory = _record_directory(directory)
    if contents is None and not directory.exists():
        return [], None
    input_hashes = audit.setdefault("file_hashes", {}) if audit is not None else None
    if execution is None:
        execution = read_json(directory / "manifest.json", input_hashes=input_hashes,
                              contents=contents.get("manifest.json") if contents is not None else None)
    _validate_execution(manifest, execution)
    lookup = {task["task_id"]: task for task in tasks}
    if contents is not None:
        paths = [directory / name for name in sorted(contents) if name.endswith(".json")]
    else:
        paths = sorted(directory.glob("*.json")) if paths is None else sorted(paths)
    receipts = {}
    for path in (path for path in paths if path.name.endswith(".started.json")):
        receipt = read_json(path, input_hashes=input_hashes,
                            contents=contents[path.name] if contents is not None else None)
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if digest(body) != receipt.get("receipt_id") or receipt.get("task_id") not in lookup:
            raise ValueError("Interrupted/start receipt integrity or task identity mismatch.")
        expected = {**lookup[receipt["task_id"]], "batch_id": manifest["batch_id"],
                    "execution_id": execution["execution_id"], "task_digest": digest(lookup[receipt["task_id"]])}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("Start receipt protocol/initial angles/execution mismatch.")
        if (path.name != f"{receipt['task_id']}--{receipt['attempt_id']}.started.json"
                or type(receipt.get("attempt_number")) is not int or receipt["attempt_number"] < 1
                or not receipt.get("started_utc")):
            raise ValueError("Invalid attempt start identity.")
        receipts[receipt["attempt_id"]] = receipt
        if audit is not None:
            audit.setdefault("started_attempt_ids", []).append(receipt["attempt_id"])
    attempts = []
    for path in paths:
        if path.name == "manifest.json" or path.name.endswith((".started.json", ".io.json")):
            continue
        attempt = read_run(path, require_integrity=True, input_hashes=input_hashes,
                           contents=contents[path.name] if contents is not None else None)
        if attempt["task_id"] not in lookup:
            raise ValueError("An attempt belongs to a different task list.")
        if path.name != f"{attempt['task_id']}--{attempt['attempt_id']}.json":
            raise ValueError("Attempt filename does not match its identity.")
        validate_attempt(lookup[attempt["task_id"]], attempt, manifest, execution)
        receipt = receipts.get(attempt["attempt_id"])
        if receipt is None or any(attempt.get(key) != receipt[key] for key in
                                  ("receipt_id", "attempt_number", "started_utc")):
            raise ValueError("Completed attempt does not match its immutable start receipt.")
        if attempt.get("device_identity") != receipt.get("device_identity"):
            raise ValueError("Completed attempt device identity differs from its start receipt.")
        io_path = directory / f"{attempt['task_id']}--{attempt['attempt_id']}.io.json"
        if (io_path.name in contents) if contents is not None else io_path.exists():
            io = read_json(io_path, input_hashes=input_hashes,
                           contents=contents[io_path.name] if contents is not None else None)
            if io.get("attempt_id") != attempt["attempt_id"] or not math.isfinite(io["save_seconds"]) or io["save_seconds"] < 0:
                raise ValueError("I/O measurement provenance mismatch.")
            if audit is not None:
                audit.setdefault("save_seconds", {})[attempt["attempt_id"]] = io["save_seconds"]
        attempts.append(_attempt_metadata(attempt) if metadata_only else attempt)
    if len({r["attempt_id"] for r in attempts}) != len(attempts):
        raise ValueError("Duplicate attempt ID.")
    complete_names = {f"{row['task_id']}--{row['attempt_id']}.io.json" for row in attempts}
    if any(path.name.endswith(".io.json") and path.name not in complete_names for path in paths):
        raise ValueError("I/O measurement has no completed attempt.")
    return attempts, execution


def _task_groups(tasks):
    groups = defaultdict(list)
    for task in tasks:
        groups[task["iso_class_id"], task["experiment_role"]].append(task)
    return dict(sorted(groups.items()))


def _scope_metadata(manifest, execution, tasks):
    return {"batch_id": manifest["batch_id"], "execution_id": execution["execution_id"],
            "group_id": digest([tasks[0]["iso_class_id"], tasks[0]["experiment_role"]]),
            "task_ids": sorted(task["task_id"] for task in tasks)}


def _check_partition_paths(directory, groups):
    """Inspect names once; foreign scopes never silently disappear in a filtered run."""
    allowed = {digest(list(key)) for key in groups}
    for path in directory.iterdir():
        if path.is_symlink() or (path.name not in {"manifest.json", "active", "sealed", "writer.lock"}
                                  and not (path.is_file() and record_temporary(path.name))):
            raise ValueError(f"Unexpected partitioned output entry: {path.name}")
    for folder, suffix in (("active", ""), ("sealed", ".tgz")):
        parent = directory / folder
        if parent.exists():
            for path in parent.iterdir():
                if (folder == "sealed" and not path.is_symlink() and path.is_file()
                        and path.name.startswith(".") and path.name.endswith(".tmp")):
                    continue  # Unpublished interrupted archive; never completion evidence.
                identity = path.name.removesuffix(suffix) if suffix else path.name
                if (path.is_symlink() or identity not in allowed or
                    (folder == "active" and not path.is_dir()) or
                    (folder == "sealed" and (not path.is_file() or not path.name.endswith(suffix)))):
                    raise ValueError(f"Unexpected partition scope: {path}")


def _load_scope(manifest, execution, tasks, directory, *, metadata_only=False, audit=None):
    metadata = _scope_metadata(manifest, execution, tasks)
    identity = metadata["group_id"]
    active, archive = directory / "active" / identity, directory / "sealed" / f"{identity}.tgz"
    contents = None
    if active.exists():
        for path in active.iterdir():
            if path.is_symlink() or not path.is_file() or not (path.name.endswith(".json") or record_temporary(path.name)):
                raise ValueError(f"Unexpected active record entry: {path}")
    if archive.exists():
        stored, contents = read_record_bundle(archive)
        if stored != metadata or "manifest.json" not in contents:
            raise ValueError("Sealed scope task/execution identity mismatch.")
        if active.exists():
            for path in active.iterdir():
                if path.is_symlink() or not path.is_file() or path.name not in contents or path.read_bytes() != contents[path.name]:
                    raise ValueError("Active files conflict with the sealed scope.")
    elif not active.exists():
        return []
    local_audit = {} if audit is not None else None
    rows, actual = _load_flat_attempts(manifest, tasks, active, metadata_only=metadata_only,
                                       audit=local_audit, contents=contents)
    if actual != execution:
        raise ValueError("Scope and root execution identities differ.")
    if contents is not None:
        chosen = select_attempts(rows)
        if set(chosen) != set(metadata["task_ids"]) or any(execution_failed(row) for row in chosen.values()):
            raise ValueError("Sealed scope contains missing or failed tasks.")
    if audit is not None:
        opaque = contents if contents is not None else {
            path.name: path.read_bytes() for path in active.iterdir() if record_temporary(path.name)}
        local_audit.setdefault("file_hashes", {}).update({name: hashlib.sha256(value).hexdigest()
                                                        for name, value in opaque.items() if record_temporary(name)})
        audit.setdefault("file_hashes", {}).update({identity + "/" + name: value
                                                    for name, value in local_audit.get("file_hashes", {}).items()})
        audit.setdefault("started_attempt_ids", []).extend(local_audit.get("started_attempt_ids", []))
        audit.setdefault("save_seconds", {}).update(local_audit.get("save_seconds", {}))
    return rows


def iter_attempt_groups(manifest, tasks, directory, *, metadata_only=False, graph_ids=None, audit=None):
    """Read each flat graph or sealed graph/role once; never retain the whole trace set."""
    directory = _record_directory(directory)
    if not directory.exists():
        return
    execution = read_json(directory / "manifest.json",
                          input_hashes=audit.setdefault("file_hashes", {}) if audit is not None else None)
    _validate_execution(manifest, execution)
    groups = _task_groups(tasks)
    requested = set(graph_ids) if graph_ids is not None else {key[0] for key in groups}
    if requested - {key[0] for key in groups}:
        raise ValueError("Requested graph is absent from this batch.")
    layout = execution.get("storage_layout", "flat")
    if layout == PARTITIONED_LAYOUT:
        _check_partition_paths(directory, groups)
        for (identity, _), group in groups.items():
            if identity in requested:
                yield group, _load_scope(manifest, execution, group, directory,
                                         metadata_only=metadata_only, audit=audit), execution
    elif layout == "flat":
        by_graph, files = defaultdict(list), defaultdict(list)
        lookup = {task["task_id"]: task for task in tasks}
        for task in tasks:
            by_graph[task["iso_class_id"]].append(task)
        for path in sorted(directory.glob("*.json")):
            if path.name != "manifest.json":
                task = lookup.get(path.name.partition("--")[0])
                if task is None:
                    raise ValueError("Attempt directory includes an unplanned task.")
                files[task["iso_class_id"]].append(path)
        for identity, group in sorted(by_graph.items()):
            if identity in requested:
                rows, _ = _load_flat_attempts(manifest, group, directory, metadata_only=metadata_only,
                                              paths=files[identity], audit=audit, execution=execution)
                yield group, rows, execution
    else:
        raise ValueError("Unsupported attempt storage layout.")


def load_attempts(manifest, tasks, directory, *, metadata_only=False, paths=None, audit=None,
                  execution=None, graph_ids=None):
    """Compatibility reader; full audits visit every scope, filtered reads state their scope."""
    if paths is not None or execution is not None or not tasks:
        return _load_flat_attempts(manifest, tasks, directory, metadata_only=metadata_only,
                                   paths=paths, audit=audit, execution=execution)
    attempts, actual = [], None
    for _, rows, actual in iter_attempt_groups(manifest, tasks, directory, metadata_only=metadata_only,
                                               graph_ids=graph_ids, audit=audit):
        attempts.extend(rows)
    if len({row["attempt_id"] for row in attempts}) != len(attempts):
        raise ValueError("Duplicate attempt ID across groups.")
    return attempts, actual


def select_attempts(attempts):
    """One sample/task; choose first nonfault execution, never the highest score."""
    groups = defaultdict(list)
    for attempt in attempts:
        groups[attempt["task_id"]].append(attempt)
    selected = {}
    for task_id, group in groups.items():
        ordered = sorted(group, key=lambda row: (row["attempt_number"], row["started_utc"], row["attempt_id"]))
        selected[task_id] = next((row for row in ordered if not execution_failed(row)), ordered[0])
    return selected


class ReferenceGridError(RuntimeError):
    """Keep work completed/requested before an analytic grid failure."""

    def __init__(self, error, statistics):
        super().__init__(f"{type(error).__name__}: {error}")
        self.statistics = statistics


def p1_grid_reference(graph, settings):
    """Chunked half-open grids and fixed refinement; closeness is not a bound."""
    start = perf_counter()
    levels, spent, requested, evaluated = [], 0, 0, 0
    for factor in (1, settings["refinement_factor"]):
        ng, nb = settings["gamma_points"] * factor, settings["beta_points"] * factor
        best = None
        value_hash = hashlib.sha256()
        for offset in range(0, ng * nb, settings["chunk_size"]):
            indices = np.arange(offset, min(offset + settings["chunk_size"], ng * nb))
            gamma, beta = (indices // nb) * (2 * np.pi / ng), (indices % nb) * (np.pi / nb)
            requested += len(indices)
            try:
                values = np.asarray(p1_expectation_grid(graph, gamma, beta), dtype=np.float64)
                evaluated += len(indices)
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError("Nonfinite p1 grid values.")
            except Exception as error:
                raise ReferenceGridError(error, {"settings": settings, "levels": levels,
                    "analytic_evaluations": evaluated, "analytic_points_requested": requested,
                    "elapsed_seconds": perf_counter() - start}) from error
            value_hash.update(values.astype("<f8").tobytes())
            i = int(np.argmax(values))
            candidate = {"C": float(values[i]), "theta": [float(gamma[i]), float(beta[i])],
                         "call_id": spent + offset + i + 1}
            if best is None or candidate["C"] > best["C"]:
                best = candidate
        levels.append({"gamma_points": ng, "beta_points": nb, "evaluations": ng * nb,
                       "best": best, "values_sha256": value_hash.hexdigest()})
        spent += ng * nb
    best = max((level["best"] for level in levels), key=lambda row: row["C"])
    delta = abs(levels[1]["best"]["C"] - levels[0]["best"]["C"])
    return {"grid_version": "half-open-row-major-v1", "periods": [2 * np.pi, np.pi],
            "origin": [0.0, 0.0], "endpoint": False, "settings": settings, "levels": levels,
            "C_ref_candidate": best["C"], "theta": best["theta"], "best_call_id": best["call_id"],
            "analytic_evaluations": spent, "analytic_points_requested": requested, "refinement_difference": delta,
            "refinement_passed": delta <= settings["refinement_tolerance"],
            "strict_error_bound": False, "elapsed_seconds": perf_counter() - start}


def _execute_task(task, graph, circuit):
    settings = OptimizerSettings(**task["budget"]) if task["kind"] == "optimization" else OptimizerSettings()
    if task["kind"] == "optimization":
        return optimize_qaoa(graph, task["p"], task["theta0"], settings, circuit=circuit)
    result = failure_record(task["theta0"] or [], settings, "", p=task["p"])
    result.update(stop_reason="completed", error=None)
    if task["kind"] == "p1_grid":
        try:
            result["grid"] = p1_grid_reference(graph, task["budget"])
        except ReferenceGridError as error:
            result.update(stop_reason="numerical_error" if isinstance(error.__cause__, FloatingPointError) else "evaluation_error",
                          error=str(error), grid=error.statistics)
        result["elapsed_seconds"] = result["grid"]["elapsed_seconds"]
    else:
        start = perf_counter()
        tracker = qml.Tracker(circuit.device)
        sample = {"call_id": 1, "seed": task["seed"], "theta": task["theta0"],
                  "score": None, "gradient": None, "device_work": {}}
        try:
            with tracker:
                loss, gradient = qaoa_loss_and_gradient(task["theta0"], circuit)
            if np.isfinite(loss):
                sample["score"] = -loss
            if np.shape(gradient) == (2 * task["p"],) and np.all(np.isfinite(gradient)):
                sample["gradient"] = (-gradient).tolist()
            if sample["score"] is None or sample["gradient"] is None:
                raise FloatingPointError("Nonfinite diagnostic score/gradient.")
        except Exception as error:
            result.update(stop_reason="numerical_error" if isinstance(error, FloatingPointError) else "evaluation_error",
                          error=f"{type(error).__name__}: {error}")
        sample["device_work"] = dict(tracker.totals)
        result["gradient_samples"] = [sample]
        result["counts"].update(objective_calls=1, gradient_requests=1,
                                device_executions=int(tracker.totals.get("executions", 0)),
                                device_derivatives=int(tracker.totals.get("derivatives", 0)),
                                device_vjps=int(tracker.totals.get("vjps", 0)))
        result["elapsed_seconds"] = perf_counter() - start
    return result


def _execute_work(manifest, library, work, output, execution, attempts, report, *, backend,
                  device_identity, max_tasks):
    """Run the same numerical loop for legacy directories and active scopes."""
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists():
        write_once_json(output / "manifest.json", execution)
    elif read_json(output / "manifest.json") != execution:
        raise ValueError("Concurrent worker has incompatible execution settings.")
    selected = select_attempts(attempts)
    graphs = {row["iso_class_id"]: row for row in library["graphs"]}
    attempt_groups = defaultdict(list)
    for row in attempts:
        attempt_groups[row["task_id"]].append(row)
    circuits, prepared = {}, {}
    active_key = None
    for task in work:
        prior = attempt_groups[task["task_id"]]
        chosen = selected.get(task["task_id"])
        if chosen is not None and not execution_failed(chosen):
            continue
        if max_tasks is not None and report["executed"] >= max_tasks:
            break
        # A fixed cap applies to completed execution faults, never normal misses.
        for number in range(len(prior) + 1, manifest["config"]["max_attempts"] + 1):
            if max_tasks is not None and report["executed"] >= max_tasks:
                break
            attempt_id, started = uuid4().hex, datetime.now(timezone.utc).isoformat()
            identity = {**task, "batch_id": manifest["batch_id"], "execution_id": execution["execution_id"],
                        "task_digest": digest(task), "attempt_id": attempt_id, "attempt_number": number,
                        "started_utc": started, "device_identity": device_identity}
            identity["receipt_id"] = digest(identity)
            write_once_json(output / f"{task['task_id']}--{attempt_id}.started.json", identity)
            key = task["iso_class_id"], task["p"]
            start = perf_counter()
            prep_seconds = 0.0
            constructing = False
            try:
                if task["iso_class_id"] not in prepared:
                    prepared.clear()
                    prepared[task["iso_class_id"]] = graph_from_record(graphs[task["iso_class_id"]])
                graph = prepared[task["iso_class_id"]]
                if key != active_key:
                    circuits.clear()
                    active_key = key
                if task["kind"] != "p1_grid" and key not in circuits:
                    prep_start = perf_counter()
                    constructing = True
                    circuits[key] = make_qaoa(graph, task["p"], backend=backend)
                    constructing = False
                    prep_seconds = perf_counter() - prep_start
                result = _execute_task(task, graph, circuits.get(key))
            except Exception as error:
                settings = OptimizerSettings(**task["budget"]) if task["kind"] == "optimization" else OptimizerSettings()
                result = failure_record(task["theta0"] or [], settings, error, p=task["p"],
                                        stop_reason="numerical_error" if isinstance(error, FloatingPointError)
                                        else "evaluation_error" if constructing else "program_error")
                result["elapsed_seconds"] = perf_counter() - start
            try:
                result.pop("edges", None)
                result.update(identity, preparation_seconds=prep_seconds,
                              completed_utc=datetime.now(timezone.utc).isoformat())
                validate_attempt(task, result, manifest, execution)
                # Serialization itself can expose a program error (e.g. NaN in
                # metadata); retain that failure instead of only a start receipt.
                json.dumps(result, allow_nan=False)
            except Exception as error:
                result = _invalid_result_failure(result, task, error, perf_counter() - start)
                result.update(identity, preparation_seconds=prep_seconds,
                              completed_utc=datetime.now(timezone.utc).isoformat())
                validate_attempt(task, result, manifest, execution)
            io_start = perf_counter()
            save_run(output / f"{task['task_id']}--{attempt_id}.json", result)
            write_once_json(output / f"{task['task_id']}--{attempt_id}.io.json",
                            {"attempt_id": attempt_id, "save_seconds": perf_counter() - io_start})
            metadata = _attempt_metadata(result)
            attempts.append(metadata)
            prior.append(metadata)
            report["compute_started"] = True
            report["executed"] += 1
            report["execution_faults"] += int(execution_failed(result))
            if not execution_failed(result):
                break
    _completion(report, work, attempts)
    return report


@contextmanager
def _writer_lock(directory):
    """One writer per partitioned output; the OS releases the lock on process exit."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "writer.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise ValueError("Another writer owns this output; use one worker per directory.") from error
        try:
            yield
        finally:
            release()


def _completion(report, work, attempts):
    chosen = select_attempts(attempts)
    report.update(completed=len(chosen), valid_completed=sum(not execution_failed(row) for row in chosen.values()),
                  unresolved_execution_faults=sum(task["task_id"] in chosen and execution_failed(chosen[task["task_id"]]) for task in work),
                  missing_tasks=sum(task["task_id"] not in chosen for task in work))


def run_worker(batch_directory, output, *, backend="default.qubit", execute=False,
               shard_index=0, shard_count=1, max_tasks=None, roles=None, partitioned=False, graph_ids=None):
    """Resume verified attempts; optionally seal complete graph/role scopes losslessly.

    Partitioned execution has one writer. Graph selection limits record reads,
    not task identity, budgets or seeds; reports explicitly describe that scope.
    """
    manifest, library, tasks = read_batch(batch_directory)
    groups = _task_groups(tasks)
    if roles is not None and not set(roles) <= {"tier1", "reference", "evaluation", "gradient_diagnostic"}:
        raise ValueError("Unknown experiment role.")
    if graph_ids is not None and not set(graph_ids) <= {key[0] for key in groups}:
        raise ValueError("Requested graph is absent from this batch.")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive when provided.")
    if partitioned and (shard_index != 0 or shard_count != 1):
        raise ValueError("Partitioned output uses graph selection and one writer, not task shards.")
    work = sorted((task for task in shard_tasks(tasks, shard_index, shard_count)
                   if (roles is None or task["experiment_role"] in roles)
                   and (graph_ids is None or task["iso_class_id"] in graph_ids)),
                  key=lambda row: (row["iso_class_id"], row["p"], row["experiment_role"], row["restart_id"]))
    source, runtime = source_identity(), environment(backend)
    device_identity = runtime.pop("device_identity")
    execution = {"batch_id": manifest["batch_id"], "source": source, "environment": runtime,
                 "protocol_version": manifest["config"]["protocol_version"], "config": manifest["config"],
                 "attempt_selection_rule": ATTEMPT_RULE}
    if partitioned:
        execution["storage_layout"] = PARTITIONED_LAYOUT
    execution["execution_id"] = digest(execution)
    issues = list(manifest["issues"])
    if runtime["gpu"] is not None and runtime["gpu"]["unknown"]:
        issues.append({"unknown_gpu_environment": runtime["gpu"]["unknown"],
                       "action": "verify_package_metadata_and_device_visibility_before_execution"})
    if source["source_id"] != manifest["source_id"]:
        issues.append({"mismatch": "source_changed_since_planning"})
    report = {"dry_run": not execute, "batch_id": manifest["batch_id"], "planned": len(tasks),
              "shard_tasks": len(work), "issues": issues,
              "completed_scope": "selected_groups" if partitioned else "entire_batch",
              "work_scope": {"roles": sorted(roles) if roles is not None else "all",
                             "graph_ids": sorted(graph_ids) if graph_ids is not None else "all",
                             "shard_index": shard_index, "shard_count": shard_count},
              "coverage": manifest["coverage"], "executed": 0, "execution_faults": 0, "compute_started": False}
    output = _record_directory(output)
    with _writer_lock(output) if execute and partitioned else nullcontext():
        old = read_json(output / "manifest.json") if (output / "manifest.json").exists() else None
        if old is not None:
            _validate_execution(manifest, old)
            if old != execution:
                issues.append({"mismatch": "execution_environment_or_settings_changed", "action": "use_a_new_output_directory"})
        if execute and issues:
            raise ValueError(f"Incomplete or incompatible batch: {issues}")
        if not partitioned:
            attempts, _ = (_load_flat_attempts(manifest, tasks, output, metadata_only=True, execution=old)
                            if old is None or old.get("storage_layout", "flat") == "flat" else
                            load_attempts(manifest, tasks, output, metadata_only=True))
            _completion(report, work, attempts)
            return (_execute_work(manifest, library, work, output, execution, attempts, report,
                                  backend=backend, device_identity=device_identity, max_tasks=max_tasks)
                    if execute else report)
        if output.exists():
            _check_partition_paths(output, groups)
            if old is None and any((output / name).exists() for name in ("active", "sealed")):
                raise ValueError("Partitioned records have no root execution manifest.")
        if execute and old is None:
            write_once_json(output / "manifest.json", execution)
        report.update(completed=0, valid_completed=0, unresolved_execution_faults=0, missing_tasks=0)
        for group in _task_groups(work).values():
            attempts = _load_scope(manifest, old or execution, group, output, metadata_only=True)
            metadata = _scope_metadata(manifest, execution, group)
            active, archive = output / "active" / metadata["group_id"], output / "sealed" / (metadata["group_id"] + ".tgz")
            scope = {"executed": 0, "execution_faults": 0, "compute_started": False}
            if execute:
                remaining = None if max_tasks is None else max_tasks - report["executed"]
                if not archive.exists() and (remaining is None or remaining > 0):
                    _execute_work(manifest, library, group, active, execution, attempts, scope,
                                  backend=backend, device_identity=device_identity, max_tasks=remaining)
                chosen = select_attempts(attempts)
                if (len(chosen) == len(group) and all(not execution_failed(row) for row in chosen.values())
                        and (not archive.exists() or active.exists())):
                    seal_record_bundle(active, archive, metadata)
                report["executed"] += scope["executed"]
                report["execution_faults"] += scope["execution_faults"]
                report["compute_started"] |= scope["compute_started"]
            _completion(scope, group, attempts)
            for name in ("completed", "valid_completed", "unresolved_execution_faults", "missing_tasks"):
                report[name] += scope[name]
        return report


def freeze_references(manifest, tasks, selected):
    """Use reference tasks only; freeze even incomplete candidates as provisional."""
    groups = defaultdict(list)
    for task in tasks:
        if task["experiment_role"] == "reference":
            groups[task["iso_class_id"], task["p"]].append(task)
    references = {}
    for (graph_id, p), group in groups.items():
        candidates, valid = [], []
        for task in sorted(group, key=lambda row: row["restart_id"]):
            attempt = selected.get(task["task_id"])
            if attempt is None or execution_failed(attempt):
                continue
            if p == 1:
                grid = attempt["grid"]
                candidate = {"C": grid["C_ref_candidate"], "theta": grid["theta"],
                             "call_id": grid["best_call_id"], "grid": grid}
                if grid["refinement_passed"]:
                    valid.append(task["task_id"])
            else:
                candidate = {"C": attempt["best_seen"], "theta": attempt["best_theta"],
                             "call_id": attempt["best_call_id"]}
                if candidate["C"] is not None and math.isfinite(candidate["C"]):
                    valid.append(task["task_id"])
            if candidate["C"] is not None and math.isfinite(candidate["C"]):
                candidates.append({**candidate, "task_id": task["task_id"], "attempt_id": attempt["attempt_id"]})
        best = max(candidates, key=lambda row: row["C"], default=None)
        ref = {"library_id": manifest["library_id"], "batch_id": manifest["batch_id"],
               "iso_class_id": graph_id, "p": p, "protocol_version": manifest["config"]["protocol_version"],
               "candidate_rule": "p1_grid_fixed_refinement" if p == 1 else "best_seen_valid_execution",
               "planned_task_ids": sorted(t["task_id"] for t in group), "valid_task_ids": sorted(valid),
               "selected_attempt_ids": [c["attempt_id"] for c in candidates],
               "complete": len(valid) == len(group), "C_ref": best["C"] if best else None,
               "source": best, "strict_global_optimum": False}
        ref["reference_id"] = digest(ref)
        references[graph_id, p] = ref
    return references


def save_references(directory, references):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = []
    for ref in references.values():
        name = ref["reference_id"] + ".json"
        write_once_json(directory / name, ref)
        files.append(name)
    write_once_json(directory / "manifest.json", {"files": sorted(files), "reference_set_id": digest(sorted(files))})


def read_references(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if digest(manifest["files"]) != manifest["reference_set_id"]:
        raise ValueError("Reference set manifest changed.")
    references = {}
    for name in manifest["files"]:
        ref = read_json(directory / name)
        if digest({k: v for k, v in ref.items() if k != "reference_id"}) != ref["reference_id"] or name != ref["reference_id"] + ".json":
            raise ValueError("Reference content identity mismatch.")
        key = ref["iso_class_id"], ref["p"]
        if key in references:
            raise ValueError("Multiple reference versions for one graph/depth in a frozen set.")
        references[key] = ref
    return references
