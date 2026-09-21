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


def save_batch(directory, library_path, config, *, b2_fit=None, reference_binding=None, b3_initializations=None,
               b4_prepared=None):
    """Freeze task JSONL and metadata. This operation performs no scientific computation."""
    directory, library_path = Path(directory), Path(library_path).resolve()
    library = read_library(library_path)
    if b4_prepared is not None:
        config = {**config, "b4_inputs": {"preparation_id": b4_prepared["preparation_id"],
                  "prepared_digest": digest(b4_prepared), "reference_binding_digest": digest(reference_binding)}}
        tasks = build_b4_tasks(library, config, b4_prepared, reference_binding)
    elif b3_initializations is not None:
        config = {**config, "b3_inputs": {
            "initialization_set_id": b3_initializations["initialization_set_id"],
            "initializations_digest": digest(b3_initializations),
            "initializations_sha256": hashlib.sha256((json.dumps(b3_initializations, indent=2,
                allow_nan=False) + "\n").encode()).hexdigest(),
            "reference_binding_digest": digest(reference_binding)}}
        tasks = build_b3_tasks(library, config, b3_initializations, reference_binding)
    elif b2_fit is not None:
        config = {**config, "b2_inputs": {"fit_id": b2_fit["fit_id"],
                  "fit_digest": digest(b2_fit), "reference_binding_digest": digest(reference_binding)}}
        tasks = build_b2_tasks(library, config, b2_fit, reference_binding)
    else:
        if config.get("kind") in ("b2", "b3", "b4"):
            raise ValueError("Warm planning requires frozen initialization inputs and audited reference binding.")
        tasks = build_tasks(library, config)
    content = "".join(json.dumps(task, sort_keys=True, allow_nan=False) + "\n" for task in tasks).encode()
    source = source_identity()
    if b3_initializations is not None and b3_initializations["source"] != source:
        raise ValueError("B3 source changed since preparation; prepare again with the final source.")
    if b4_prepared is not None and b4_prepared["source"] != source:
        raise ValueError("B4 source changed since preparation; use the final frozen source.")
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
    if b2_fit is not None:
        write_once_json(directory / "fit.json", b2_fit)
    if b3_initializations is not None:
        write_once_json(directory / "initializations.json", b3_initializations)
    if b4_prepared is not None:
        write_once_json(directory / "prepared.json", b4_prepared)
    if reference_binding is not None:
        write_once_json(directory / "reference_binding.json", reference_binding)
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
    if manifest["config"].get("kind") == "b2":
        fit, binding = read_b2_inputs(directory, manifest)
        if build_b2_tasks(library, manifest["config"], fit, binding) != tasks:
            raise ValueError("B2 tasks disagree with frozen fit/reference inputs.")
    elif manifest["config"].get("kind") == "b3":
        prepared, binding = read_b3_inputs(directory, manifest)
        if build_b3_tasks(library, manifest["config"], prepared, binding) != tasks:
            raise ValueError("B3 tasks disagree with frozen initialization/reference inputs.")
    elif manifest["config"].get("kind") == "b4":
        prepared, binding = read_b4_inputs(directory, manifest)
        if build_b4_tasks(library, manifest["config"], prepared, binding) != tasks:
            raise ValueError("B4 tasks disagree with frozen prediction/reference inputs.")
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
    required.update({key: task[key] for key in ("fit_id", "reference_id", "regime", "fold", "variant",
        "rule_id", "initialization_id", "initialization_set_id") if key in task})
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
    if roles is not None and not set(roles) <= {"tier1", "reference", "evaluation", "gradient_diagnostic", "warm_start"}:
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
    if manifest["config"].get("kind") == "b3":
        prepared, _ = read_b3_inputs(batch_directory, manifest)
        if prepared["environment"] != runtime:
            issues.append({"mismatch": "B3_preparation_environment_changed"})
    if manifest["config"].get("kind") == "b4":
        prepared, _ = read_b4_inputs(batch_directory, manifest)
        if prepared["compute_environment"] != runtime:
            issues.append({"mismatch": "B4_bound_compute_environment_changed"})
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


def attempt_cost_totals(attempts, *, save_seconds=None):
    """Account all retained attempts; unavailable work remains unknown."""
    from .analysis import _costs

    save_seconds = save_seconds or {}
    missing = [row["attempt_id"] for row in attempts if save_seconds.get(row["attempt_id"]) is None]
    measured = math.fsum(save_seconds[row["attempt_id"]] for row in attempts
                         if save_seconds.get(row["attempt_id"]) is not None)
    return {**_costs(attempts), "attempts": len(attempts),
            "record_save_seconds": None if missing else measured,
            "record_save_seconds_measured": measured, "record_save_measurements_missing": len(missing),
            "record_save_missing_attempt_ids": missing,
            "execution_faults": sum(execution_failed(r) for r in attempts),
            "population": "all_retained_attempts_including_execution_faults"}


def audit_b2_references(batch, references_directory, attempts_directory, graph_ids, *, depths=(1, 2)):
    """Re-derive required frozen references from validated original attempts.

    This is a read contract, never permission to resume the source batch.
    Only score-defining source files must match; optimizer compatibility is a
    separate requirement when comparing B1 traces.
    """
    manifest, library, tasks = read_batch(batch)
    if manifest["config"].get("kind") not in (None, "b1"):
        raise ValueError("Warm initialization requires an original B1 reference batch.")
    if not depths or len(set(depths)) != len(depths) or not set(depths) <= {1, 2}:
        raise ValueError("Reference audit needs a nonempty subset of p=1/2.")
    current_source = source_identity()
    for name in ("qaoa_study/qaoa.py", "qaoa_study/exact.py"):
        if manifest["source"]["files"].get(name) != current_source["files"].get(name):
            raise ValueError("Reference score convention/source requires independent validation.")
    ref_manifest = read_json(Path(references_directory) / "manifest.json")
    if digest(ref_manifest["files"]) != ref_manifest["reference_set_id"]:
        raise ValueError("Reference set manifest changed.")
    audit = {}
    attempts, execution = load_attempts(manifest, tasks, attempts_directory, metadata_only=True,
                                        graph_ids=graph_ids, audit=audit)
    expected = freeze_references(manifest, tasks, select_attempts(attempts))
    required = {(identity, p) for identity in graph_ids for p in depths}
    references = {}
    for key in required:
        candidate = expected.get(key)
        name = candidate["reference_id"] + ".json" if candidate else None
        if not candidate or not candidate["complete"] or name not in ref_manifest["files"]:
            raise ValueError(f"Complete, reproducible reference required for {key}.")
        ref = read_json(Path(references_directory) / name)
        if ref != candidate:
            raise ValueError(f"Complete, reproducible reference required for {key}.")
        references[key] = ref
        if not np.all(np.isfinite(ref["source"]["theta"])) or len(ref["source"]["theta"]) != 2 * key[1]:
            raise ValueError("Invalid reference angle dimensions or values.")
    binding = {"version": "b2-reference-binding-v1", "source_batch_id": manifest["batch_id"],
        "source_id": manifest["source_id"], "source": manifest["source"],
        "source_config": manifest["config"], "library_id": library["library_id"],
        "reference_set_id": ref_manifest["reference_set_id"],
        "source_batch_manifest_sha256": hashlib.sha256((Path(batch) / "manifest.json").read_bytes()).hexdigest(),
        "reference_manifest_sha256": hashlib.sha256((Path(references_directory) / "manifest.json").read_bytes()).hexdigest(),
        "execution_id": execution["execution_id"] if execution else None,
        "attempt_input_digest": digest(audit.get("file_hashes", {})),
        "interrupted_attempt_ids_in_read_scopes": sorted(set(audit.get("started_attempt_ids", [])) -
                                                          {row["attempt_id"] for row in attempts}),
        "references": [references[key] for key in sorted(required)]}
    costs = attempt_cost_totals([r for r in attempts if r["experiment_role"] == "reference"
                                and (r["iso_class_id"], r["p"]) in required],
                               save_seconds=audit.get("save_seconds"))
    costs.update(interrupted_attempt_ids_in_read_scopes=binding["interrupted_attempt_ids_in_read_scopes"],
                 scope="Completed reference attempts on required graph/depth pairs; interrupted work in read scopes has unknown cost.")
    return manifest, library, binding, costs


def save_b2_fit(output, batch, references_directory, attempts_directory, *, regime="random", fold=None):
    """Fit both depths from every predeclared training reference, with no QAOA calls."""
    from .learning import B2_RULES, fixed_angles, select_b2_graphs

    _, library, _ = read_batch(batch)
    training = select_b2_graphs(library, regime, fold, "training")
    ids = [r["iso_class_id"] for r in training]
    if not ids:
        raise ValueError("The declared B2 training set is empty.")
    _, _, binding, costs = audit_b2_references(batch, references_directory, attempts_directory, ids)
    start = perf_counter()
    fits = {str(p): fixed_angles([{"iso_class_id": ref["iso_class_id"], "theta": ref["source"]["theta"]}
                                  for ref in binding["references"] if ref["p"] == p]) for p in (1, 2)}
    fit = {"version": "b2-fit-v1", "rules": B2_RULES, "library_id": library["library_id"],
           "split_digest": digest(library["split_summary"]), "regime": regime, "fold": fold,
           "training_graph_ids": ids, "fits": fits, "reference_binding": binding,
           "reference_costs": costs, "fit_seconds": perf_counter() - start, "source": source_identity()}
    fit["fit_id"] = digest(fit)
    write_once_json(output, fit)
    return fit


def read_b2_fit(path):
    from .learning import B2_RULES

    fit = read_json(path)
    if (fit.get("version") != "b2-fit-v1" or fit.get("rules") != B2_RULES or
            digest({k: v for k, v in fit.items() if k != "fit_id"}) != fit.get("fit_id")):
        raise ValueError("B2 fit identity or construction rules changed.")
    return fit


def read_b2_inputs(directory, manifest):
    fit = read_b2_fit(Path(directory) / "fit.json")
    binding = read_json(Path(directory) / "reference_binding.json")
    expected = {"fit_id": fit["fit_id"], "fit_digest": digest(fit), "reference_binding_digest": digest(binding)}
    if manifest["config"].get("b2_inputs") != expected:
        raise ValueError("B2 fit or external reference binding changed.")
    return fit, binding


def build_b2_tasks(library, config, fit, binding):
    """One independent frozen B2 variant per evaluation graph/depth."""
    from .learning import B2_RULES, select_b2_graphs

    if digest({k: v for k, v in fit.items() if k != "fit_id"}) != fit.get("fit_id"):
        raise ValueError("B2 fit identity changed.")
    if (config.get("kind") != "b2" or config["depths"] != [1, 2] or config["epsilon"] != 0.5 or
            config["variants"] != ["medoid", "aligned_median"] or
            type(config["max_attempts"]) is not int or config["max_attempts"] < 1):
        raise ValueError("B2 requires both depths, both frozen variants and the adopted single-start budget.")
    training = select_b2_graphs(library, fit["regime"], fit["fold"], "training")
    evaluation = select_b2_graphs(library, fit["regime"], fit["fold"], "evaluation")
    ids = [r["iso_class_id"] for r in training]
    if (not ids or not evaluation or fit["training_graph_ids"] != ids or fit["rules"] != B2_RULES or
            fit["split_digest"] != digest(library["split_summary"]) or
            fit["library_id"] != library["library_id"] or binding["library_id"] != library["library_id"] or
            binding["source_batch_id"] != fit["reference_binding"]["source_batch_id"]):
        raise ValueError("B2 fit/evaluation membership or source library is incompatible.")
    refs = {(r["iso_class_id"], r["p"]): r for r in binding["references"]}
    if len(refs) != len(binding["references"]):
        raise ValueError("Duplicate bound reference.")
    for ref in fit["reference_binding"]["references"]:
        if refs.get((ref["iso_class_id"], ref["p"])) != ref:
            raise ValueError("Training reference changed after B2 fitting.")
    for ref in refs.values():
        if (not ref["complete"] or ref["library_id"] != library["library_id"] or
                ref["batch_id"] != binding["source_batch_id"] or
                digest({k: v for k, v in ref.items() if k != "reference_id"}) != ref["reference_id"]):
            raise ValueError("Invalid external reference identity.")
    tasks = []
    for p in (1, 2):
        budget = config["optimizer"][str(p)]
        OptimizerSettings(**budget)
        if budget != binding["source_config"]["optimizer"][str(p)]:
            raise ValueError("B2 must use the same single-run optimizer budget as B1.")
        for graph in evaluation:
            ref = refs.get((graph["iso_class_id"], p))
            if ref is None:
                raise ValueError("Evaluation reference is missing from the frozen binding.")
            for variant in config["variants"]:
                theta = fit["fits"][str(p)]["medoid" if variant == "medoid" else "median"]["theta"]
                stream = {"namespace": config["seed_namespace"], "seed": config["seed"],
                          "fit_id": fit["fit_id"], "iso_class_id": graph["iso_class_id"], "p": p, "variant": variant}
                task = {"task_version": "plan-a-b2-tasks-v1", "library_id": library["library_id"],
                    "iso_class_id": graph["iso_class_id"], "n": graph["n"], "p": p,
                    "graph_split": "evaluation", "kind": "optimization", "experiment_role": "warm_start",
                    "pool": None, "method": "B2", "restart_id": 0, "seed": int(digest(stream)[:32], 16),
                    "theta0": theta, "budget": budget, "protocol_version": config["protocol_version"],
                    "regime": fit["regime"], "fold": fit["fold"], "variant": variant,
                    "fit_id": fit["fit_id"], "reference_id": ref["reference_id"]}
                task["task_id"] = digest(task)
                tasks.append(task)
    return sorted(tasks, key=lambda row: row["task_id"])


def save_b2_batch(output, fit_path, batch, references_directory, attempts_directory, settings):
    """Audit external references once, then embed immutable fit/binding inputs."""
    from .learning import select_b2_graphs

    fit = read_b2_fit(fit_path)
    source, library, _ = read_batch(batch)
    evaluation = select_b2_graphs(library, fit["regime"], fit["fold"], "evaluation")
    ids = sorted(set(fit["training_graph_ids"]) | {r["iso_class_id"] for r in evaluation})
    _, _, binding, _ = audit_b2_references(batch, references_directory, attempts_directory, ids)
    inherited = {k: source["config"][k] for k in ("optimizer", "max_attempts", "expected_library",
                 "require_full_design", "design", "require_split_coverage") if k in source["config"]}
    if set(settings) & set(inherited):
        raise ValueError("B2 settings cannot override the source B1 optimizer or library contract.")
    return save_batch(output, (Path(batch) / source["library_path"]).resolve(), {**inherited, **settings},
                      b2_fit=fit, reference_binding=binding)


def validate_b1_comparison(warm, baseline, warm_execution, baseline_execution):
    """Compare existing B1 traces only under the declared identical numerical contract."""
    if warm["library_id"] != baseline["library_id"]:
        raise ValueError("B1 comparison library mismatch.")
    for name in ("qaoa_study/qaoa.py", "qaoa_study/exact.py", "qaoa_study/optimize.py"):
        if warm["source"]["files"].get(name) != baseline["source"]["files"].get(name):
            raise ValueError("B1 optimizer/objective source is not comparable.")
    depths = warm["config"]["depths"]
    if (not depths or not set(depths) <= set(baseline["config"]["depths"])
            or warm["config"]["epsilon"] != baseline["config"]["epsilon"]
            or any(warm["config"]["optimizer"][str(p)] != baseline["config"]["optimizer"][str(p)] for p in depths)):
        raise ValueError("B1 single-start comparison settings differ.")
    if warm_execution and baseline_execution and warm_execution["environment"] != baseline_execution["environment"]:
        raise ValueError("Warm comparison numerical environments differ; comparable runs are required.")


def _validate_b3_settings(settings):
    depths = settings["depths"]
    if (settings.get("kind") != "b3" or not depths or depths != sorted(set(depths))
            or any(type(p) is not int or p not in (1, 2) for p in depths) or settings["epsilon"] != 0.5):
        raise ValueError("B3 requires a nonempty ordered subset of p=1/2 and epsilon=0.5.")
    cases = settings["comparison_cases"]
    keys = [(row["regime"], row["fold"]) for row in cases]
    if (len(set(keys)) != len(keys) or ("random", None) not in keys or
            any(key != ("random", None) and key not in
                [("lofo", family) for family in ("regular", "er", "ba", "sbm")] for key in keys)):
        raise ValueError("B3 comparison cases require random and unique declared LOFO folds.")


def save_b3_initializations(output, library_path, settings, *, backend="default.qubit"):
    """Explicit graph-only preparation; no reference, fit, QNode or optimizer calls.

    Timings are separate from scientific initialization identities. The measured
    initializer interval includes graph statistics and angle arithmetic together;
    graph decoding/I/O and the entire preparation interval are reported separately.
    """
    from .learning import b3_initialization, select_b2_graphs

    _validate_b3_settings(settings)
    if Path(output).exists():
        raise FileExistsError(output)
    source, runtime = source_identity(), environment(backend)
    runtime.pop("device_identity")
    start = perf_counter()
    library_path = Path(library_path)
    library = read_library(library_path)
    graphs = select_b2_graphs(library, "random", None, "evaluation")
    if not graphs:
        raise ValueError("B3 evaluation set is empty.")
    load_seconds = perf_counter() - start
    entries, graph_timings, initializer_timings = [], [], []
    for record in graphs:
        before = perf_counter()
        graph = graph_from_record(record)
        graph_timings.append({"iso_class_id": record["iso_class_id"], "seconds": perf_counter() - before})
        for p in settings["depths"]:
            before = perf_counter()
            result = b3_initialization(graph, p)
            elapsed = perf_counter() - before
            entry = {"library_id": library["library_id"], "source_id": source["source_id"],
                     "iso_class_id": record["iso_class_id"], "p": p,
                     "topology_hash": digest({"n": record["n"], "edges": record["edges"]}), **result}
            entry["initialization_id"] = digest(entry)
            entries.append(entry)
            initializer_timings.append({"initialization_id": entry["initialization_id"], "p": p, "seconds": elapsed})
    prepared = {"version": "b3-initializations-v1", "library_id": library["library_id"],
                "library_manifest_sha256": hashlib.sha256((library_path / "manifest.json").read_bytes()).hexdigest(),
                "split_digest": digest(library["split_summary"]), "depths": settings["depths"],
                "source": source, "environment": runtime, "initializations": entries,
                "initialization_set_id": digest([row["initialization_id"] for row in entries]),
                "preparation": {"library_read_seconds": load_seconds, "graph_decode": graph_timings,
                    "initializers": initializer_timings, "total_seconds": perf_counter() - start,
                    "timing_scope": "initializer includes topology statistics and angle arithmetic; shared library read/decode charged once",
                    "external_p2_constant_optimization_seconds": None,
                    "external_cost_status": "unmeasured_published_theoretical_proxy_optimization"}}
    prepared["artifact_id"] = digest(prepared)
    write_once_json(output, prepared)
    return prepared


def read_b3_inputs(directory, manifest):
    """Read complete frozen preparation, including measured cost; never redo it."""
    path = Path(directory) / "initializations.json"
    prepared = read_json(path)
    binding = read_json(Path(directory) / "reference_binding.json")
    expected = {"initialization_set_id": prepared["initialization_set_id"],
                "initializations_digest": digest(prepared),
                "initializations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "reference_binding_digest": digest(binding)}
    if (manifest["config"].get("b3_inputs") != expected or prepared["source"] != manifest["source"]):
        raise ValueError("B3 preparation, source or external reference binding changed.")
    return prepared, binding


def build_b3_tasks(library, config, prepared, binding):
    """Rebuild one task per graph/depth using only frozen initial angles."""
    from .learning import B3_RULES, select_b2_graphs

    _validate_b3_settings(config)
    if (prepared.get("version") != "b3-initializations-v1" or
            digest({k: v for k, v in prepared.items() if k != "artifact_id"}) != prepared.get("artifact_id") or
            prepared["library_id"] != library["library_id"] or binding["library_id"] != library["library_id"] or
            prepared["split_digest"] != digest(library["split_summary"]) or prepared["depths"] != config["depths"]):
        raise ValueError("B3 preparation identity, library, split or depths disagree.")
    evaluation = select_b2_graphs(library, "random", None, "evaluation")
    expected = {(graph["iso_class_id"], p) for graph in evaluation for p in config["depths"]}
    entries = {(row["iso_class_id"], row["p"]): row for row in prepared["initializations"]}
    refs = {(r["iso_class_id"], r["p"]): r for r in binding["references"]}
    if (not expected or set(entries) != expected or len(entries) != len(prepared["initializations"])
            or set(refs) != expected or len(refs) != len(binding["references"])
            or prepared["initialization_set_id"] != digest([row["initialization_id"] for row in prepared["initializations"]])):
        raise ValueError("B3 requires exact initialization/reference coverage for every evaluation graph/depth.")
    if binding["source_config"].get("kind") not in (None, "b1"):
        raise ValueError("B3 scoring references must come from the original B1 batch.")
    if type(config["max_attempts"]) is not int or config["max_attempts"] < 1:
        raise ValueError("B3 requires a positive fixed attempt cap.")
    tasks = []
    for graph in evaluation:
        for p in config["depths"]:
            row, ref = entries[graph["iso_class_id"], p], refs[graph["iso_class_id"], p]
            if (row["rule_id"] != B3_RULES[str(p)]["rule_id"] or row["library_id"] != library["library_id"] or
                    row["source_id"] != prepared["source"]["source_id"] or
                    row["topology_hash"] != digest({"n": graph["n"], "edges": graph["edges"]}) or
                    row["initialization_id"] != digest({k: v for k, v in row.items() if k != "initialization_id"}) or
                    np.shape(row["theta"]) != (2 * p,) or not np.all(np.isfinite(row["theta"]))):
                raise ValueError("B3 initialization rule, angle or graph identity changed.")
            if (not ref["complete"] or ref["library_id"] != library["library_id"] or
                    ref["batch_id"] != binding["source_batch_id"] or
                    digest({k: v for k, v in ref.items() if k != "reference_id"}) != ref["reference_id"]):
                raise ValueError("B3 scoring reference is incomplete or changed.")
            budget = config["optimizer"][str(p)]
            OptimizerSettings(**budget)
            if budget != binding["source_config"]["optimizer"][str(p)]:
                raise ValueError("B3 must use the same optimizer budget as B1 at each depth.")
            seed = int(digest({"namespace": config["seed_namespace"], "seed": config["seed"],
                "library_id": library["library_id"], "iso_class_id": graph["iso_class_id"], "p": p,
                "rule_id": row["rule_id"]})[:32], 16)
            task = {"task_version": "plan-a-b3-tasks-v1", "library_id": library["library_id"],
                "iso_class_id": graph["iso_class_id"], "n": graph["n"], "p": p, "graph_split": "evaluation",
                "kind": "optimization", "experiment_role": "warm_start", "pool": None, "method": "B3",
                "restart_id": 0, "seed": seed, "theta0": row["theta"], "budget": budget,
                "protocol_version": config["protocol_version"], "rule_id": row["rule_id"],
                "initialization_id": row["initialization_id"], "initialization_set_id": prepared["initialization_set_id"],
                "reference_id": ref["reference_id"]}
            task["task_id"] = digest(task)
            tasks.append(task)
    return sorted(tasks, key=lambda row: row["task_id"])


def save_b3_batch(output, initializations_path, batch, references_directory, attempts_directory, settings):
    """Bind already prepared graph-only starts to audited scoring references."""
    from .learning import select_b2_graphs

    _validate_b3_settings(settings)
    prepared = read_json(initializations_path)
    source, library, _ = read_batch(batch)
    library_path = (Path(batch) / source["library_path"]).resolve()
    if prepared["library_manifest_sha256"] != source["library_manifest_sha256"]:
        raise ValueError("B3 preparation used a different frozen library.")
    ids = [g["iso_class_id"] for g in select_b2_graphs(library, "random", None, "evaluation")]
    _, _, binding, _ = audit_b2_references(batch, references_directory, attempts_directory, ids, depths=settings["depths"])
    inherited = {k: source["config"][k] for k in ("optimizer", "max_attempts", "expected_library",
                 "require_full_design", "design", "require_split_coverage") if k in source["config"]}
    if set(settings) & set(inherited):
        raise ValueError("B3 settings cannot override the source B1 optimizer or library contract.")
    validate_b1_comparison({"library_id": library["library_id"], "source": prepared["source"],
                           "config": {**inherited, **settings}}, source,
                          {"environment": prepared["environment"]},
                          read_json(Path(attempts_directory) / "manifest.json"))
    return save_batch(output, library_path, {**inherited, **settings},
                      b3_initializations=prepared, reference_binding=binding)


def b4_specifications(settings):
    """The complete predeclared design; development subsets never become production."""
    from .features import B4_GROUPS

    _validate_b3_settings({**settings, "kind": "b3"})
    if settings.get("kind") != "b4" or settings["depths"] != [1, 2]:
        raise ValueError("B4 requires both depths and its own study kind.")
    groups, seeds = settings["feature_groups"], settings["shuffle_seeds"]
    if (not groups or len(set(groups)) != len(groups) or not set(groups) <= set(B4_GROUPS)
            or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds)):
        raise ValueError("Invalid or repeated B4 feature groups/shuffle seeds.")
    learning = settings.get("learning", {})
    if set(learning) - {"development", "random_n_splits"}:
        raise ValueError("B4 learning settings cannot override the adopted rules.")
    if not learning.get("development", False):
        cases = [{"regime": "random", "fold": None}] + [
            {"regime": "lofo", "fold": family} for family in ("regular", "er", "ba", "sbm")]
        if (groups != list(B4_GROUPS) or seeds != [20260922, 20260923, 20260924]
                or settings["comparison_cases"] != cases or learning.get("random_n_splits", 5) != 5):
            raise ValueError("Production B4 requires the complete approved five-group/five-scope design.")
    specs = [{**case, "group": group, "shuffle_seed": None, "p": p}
             for case in settings["comparison_cases"] for group in groups for p in (1, 2)]
    if seeds and "F" not in groups:
        raise ValueError("The shuffled control belongs to the F feature group.")
    specs += [{"regime": "random", "fold": None, "group": "F", "shuffle_seed": seed, "p": p}
              for seed in seeds for p in (1, 2)]
    return specs


def _b4_check_identity(value, key):
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"B4 {key} checksum mismatch.")


def _b4_training_labels(manifest, tasks, attempts_directory, binding, ids):
    """Use complete original evaluation pools on training graphs, never held-out labels."""
    if manifest["config"]["evaluation_restarts"] != 50:
        raise ValueError("B4 success labels require all 50 original evaluation restarts.")
    refs = {(r["iso_class_id"], r["p"]): r for r in binding["references"]}
    audit = {}
    attempts, execution = load_attempts(manifest, tasks, attempts_directory, metadata_only=True, graph_ids=ids, audit=audit)
    selected = select_attempts(attempts)
    labels = []
    for identity in ids:
        for p in (1, 2):
            planned = [t for t in tasks if t["iso_class_id"] == identity and t["p"] == p
                       and t["experiment_role"] == "evaluation"]
            rows = [selected.get(t["task_id"]) for t in planned]
            if (len(planned) != 50 or {t["restart_id"] for t in planned} != set(range(50))
                    or any(r is None or execution_failed(r) or not r.get("run_completed") for r in rows)):
                raise ValueError(f"B4 needs every successful execution receipt in the training pool: {identity}, p={p}.")
            reference = refs[identity, p]
            successes = sum(r["C_final"] >= reference["C_ref"] - .5 for r in rows)
            labels.append({"iso_class_id": identity, "p": p, "successes": successes, "trials": 50,
                           "reference_id": reference["reference_id"],
                           "attempt_ids": [r["attempt_id"] for r in rows]})
    used = [r for r in attempts if r["experiment_role"] == "evaluation" and r["iso_class_id"] in ids]
    costs = {**attempt_cost_totals(used, save_seconds=audit.get("save_seconds")),
             "attempt_ids": sorted(r["attempt_id"] for r in used),
             "scope": "Original training-graph evaluation pools, shared once across both heads/models; no additional QAOA."}
    return labels, execution, costs


def save_b4_fits(output, batch, references_directory, attempts_directory, settings):
    """Fit the declared CPU models, resuming only complete immutable model artifacts.

    Features are topology-only; only original global-training reference/outcome
    rows are loaded here. An interrupted fit restarts from scratch, not optimizer
    state. No quantum circuit is evaluated by this function.
    """
    from threadpoolctl import threadpool_limits
    from .features import B4_FEATURE_VERSION, b4_feature_row
    from .learning import fit_b4_model, select_b2_graphs, validate_b4_model

    specs = b4_specifications(settings)
    output = Path(output)
    artifact_writes = {}

    def save_artifact(path, value):
        start = perf_counter()
        write_once_json(path, value)
        artifact_writes[Path(path).relative_to(output).as_posix()] = perf_counter() - start

    manifest, library, tasks = read_batch(batch)
    source = source_identity()
    training = select_b2_graphs(library, "random", None, "training")
    ids = [g["iso_class_id"] for g in training]
    if not ids:
        raise ValueError("B4 training set is empty.")
    setup = {"version": "b4-fitting-v1", "source": source, "settings": settings,
             "library_id": library["library_id"], "library_manifest_sha256": manifest["library_manifest_sha256"],
             "split_digest": digest(library["split_summary"]), "source_batch_id": manifest["batch_id"],
             "reference_manifest_sha256": hashlib.sha256((Path(references_directory) / "manifest.json").read_bytes()).hexdigest(),
             "execution_manifest_sha256": hashlib.sha256((Path(attempts_directory) / "manifest.json").read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output):
        if (output / "setup.json").exists():
            if read_json(output / "setup.json") != setup:
                raise ValueError("B4 fit resume source/library/configuration changed.")
        else:
            if any(output.iterdir()):
                # The writer lock may create its own advisory lock file.
                unexpected = [f for f in output.iterdir() if f.name != "writer.lock"]
                if unexpected:
                    raise ValueError("B4 fitting directory contains unbound files.")
            save_artifact(output / "setup.json", setup)
        if (output / "manifest.json").exists():
            return read_b4_fits(output)
        feature_path = output / "features.json"
        if feature_path.exists():
            features = read_json(feature_path)
            _b4_check_identity(features, "feature_artifact_id")
        else:
            start = perf_counter()
            rows = []
            for graph in library["graphs"]:
                if graph["tier"] != 2:
                    continue
                row = b4_feature_row(graph_from_record(graph), stored_features={
                    "values": graph["features"], "missing": graph["feature_missing"]})
                rows.append({**row, "iso_class_id": graph["iso_class_id"],
                             "topology_hash": digest({"n": graph["n"], "edges": graph["edges"]})})
            features = {"version": B4_FEATURE_VERSION, "library_id": library["library_id"],
                        "split_digest": setup["split_digest"], "rows": sorted(rows, key=lambda r: r["iso_class_id"]),
                        "construction_seconds": perf_counter() - start,
                        "historical_feature_seconds": None, "source": source}
            features["feature_artifact_id"] = digest(features)
            save_artifact(feature_path, features)
        _, _, binding, costs = audit_b2_references(batch, references_directory, attempts_directory, ids)
        labels, execution, label_costs = _b4_training_labels(manifest, tasks, attempts_directory, binding, ids)
        training_data = {"reference_binding": binding, "reference_costs": costs, "success_labels": labels,
                         "success_label_costs": label_costs}
        training_path = output / "training.json"
        if training_path.exists():
            saved = read_json(training_path)
            if saved != training_data:
                raise ValueError("Original B4 training references/outcomes changed on resume.")
        else:
            save_artifact(training_path, training_data)
        learning_environment = {"python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
            "blas_threads": 1, "thread_limit": "threadpoolctl context during fitting"}
        feature_lookup = {r["iso_class_id"]: r for r in features["rows"]}
        model_paths = []
        (output / "models").mkdir(exist_ok=True)
        for spec in specs:
            graphs = select_b2_graphs(library, spec["regime"], spec["fold"], "training")
            wanted = {g["iso_class_id"] for g in graphs}
            candidates = [{"iso_class_id": r["iso_class_id"], "theta": r["source"]["theta"],
                           "reference_id": r["reference_id"], "source": r["source"]}
                          for r in binding["references"] if r["p"] == spec["p"] and r["iso_class_id"] in wanted]
            outcomes = [r for r in labels if r["p"] == spec["p"] and r["iso_class_id"] in wanted]
            for head in ("angles", "success"):
                name = "models/" + digest({**spec, "head": head}) + ".json"
                model_paths.append(name)
                path = output / name
                if path.exists():
                    model = read_json(path)
                    _b4_check_identity(model, "fit_id")
                    validate_b4_model(model)
                    if (model["source"] != source or model["training_binding_digest"] != digest(training_data)
                            or model["learning_environment"] != learning_environment
                            or any(model.get(k) != v for k, v in {**spec, "head": head}.items())):
                        raise ValueError("B4 saved model no longer matches its fitting inputs.")
                    continue
                start = perf_counter()
                with threadpool_limits(limits=1):
                    model = fit_b4_model(graphs, [feature_lookup[g["iso_class_id"]] for g in graphs],
                        candidates, outcomes, **spec, head=head, settings=settings.get("learning"))
                model.update(source=source, library_id=library["library_id"], split_digest=setup["split_digest"],
                    training_binding_digest=digest(training_data), feature_artifact_id=features["feature_artifact_id"],
                    learning_environment=learning_environment, fit_seconds=perf_counter() - start)
                model["fit_id"] = digest(model)
                validate_b4_model(model)
                save_artifact(path, model)
        if source_identity() != source:
            raise ValueError("Source changed while fitting; do not finalize this B4 fit set.")
        names = ["setup.json", "features.json", "training.json", *model_paths]
        result = {**setup, "model_files": model_paths, "compute_environment": execution["environment"],
                  "artifact_io": {"write_seconds_by_file": artifact_writes,
                      "unmeasured_preexisting_files": sorted(set(names) - set(artifact_writes)),
                      "read_and_manifest_write_seconds": None,
                      "scope": "Measured immutable artifact writes in the final fitting invocation only; "
                               "reads, final manifest and interrupted earlier invocations are unmeasured."},
                  "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}}
        result["fit_set_id"] = digest(result)
        write_once_json(output / "manifest.json", result)
    return read_b4_fits(output)


def _b4_fit_manifest(manifest):
    """Require a complete portable inventory, including every declared model."""
    _b4_check_identity(manifest, "fit_set_id")
    specs = [{**s, "head": head} for s in b4_specifications(manifest["settings"])
             for head in ("angles", "success")]
    names = ["models/" + digest(s) + ".json" for s in specs]
    if (manifest.get("version") != "b4-fitting-v1" or manifest["model_files"] != names
            or set(manifest["files"]) != {"setup.json", "features.json", "training.json", *names}):
        raise ValueError("B4 fit manifest must bind every planned artifact exactly once.")
    return specs


def read_b4_fits(directory):
    """Read immutable numeric models with both byte and semantic validation."""
    from .features import B4_FEATURE_VERSION, b4_feature_matrix
    from .learning import _b4_settings, validate_b4_model

    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    expected_specs = _b4_fit_manifest(manifest)
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("B4 fitting file checksum/path mismatch.")
    models = [read_json(directory / name) for name in manifest["model_files"]]
    features, training = read_json(directory / "features.json"), read_json(directory / "training.json")
    _b4_check_identity(features, "feature_artifact_id")
    setup = read_json(directory / "setup.json")
    if (any(manifest.get(k) != v for k, v in setup.items())
            or features.get("version") != B4_FEATURE_VERSION
            or any(features.get(k) != manifest[k] for k in ("source", "library_id", "split_digest"))
            or len({r["iso_class_id"] for r in features["rows"]}) != len(features["rows"])
            or training["reference_binding"]["library_id"] != manifest["library_id"]
            or training["reference_binding"]["source_batch_id"] != manifest["source_batch_id"]):
        raise ValueError("B4 feature/training sidecar provenance changed.")
    b4_feature_matrix(features["rows"], "F")
    for model, spec in zip(models, expected_specs, strict=True):
        _b4_check_identity(model, "fit_id")
        validate_b4_model(model)
        if (any(model.get(k) != v for k, v in spec.items()) or model["source"] != manifest["source"]
                or model["library_id"] != manifest["library_id"] or model["split_digest"] != manifest["split_digest"]
                or model["settings"] != _b4_settings(manifest["settings"].get("learning"))
                or model["training_binding_digest"] != digest(training)
                or model["feature_artifact_id"] != features["feature_artifact_id"]):
            raise ValueError("B4 model identity/design/provenance changed.")
    return {"manifest": manifest, "features": features, "training": training, "models": models}


def save_b4_predictions(output, fits_directory, library_path):
    """Freeze one prediction per target before binding any evaluation references."""
    from .learning import predict_b4_model, select_b2_graphs

    if Path(output).exists():
        raise FileExistsError(output)
    loaded = read_b4_fits(fits_directory)
    manifest, models = loaded["manifest"], loaded["models"]
    library = read_library(library_path)
    if (library["library_id"] != manifest["library_id"] or digest(library["split_summary"]) != manifest["split_digest"]
            or hashlib.sha256((Path(library_path) / "manifest.json").read_bytes()).hexdigest() != manifest["library_manifest_sha256"]
            or source_identity() != manifest["source"]):
        raise ValueError("B4 prediction source/library changed since fitting.")
    rows = {r["iso_class_id"]: r for r in loaded["features"]["rows"]}
    topology = {g["iso_class_id"]: digest({"n": g["n"], "edges": g["edges"]})
                for g in library["graphs"] if g["tier"] == 2}
    if set(rows) != set(topology) or any(rows[k]["topology_hash"] != v for k, v in topology.items()):
        raise ValueError("B4 feature sidecar must cover exactly the frozen Tier2 topologies.")
    predictions, timings = [], []
    for index in range(0, len(models), 2):
        angle, success = models[index:index + 2]
        targets = select_b2_graphs(library, angle["regime"], angle["fold"], "evaluation")
        if not targets:
            raise ValueError("B4 evaluation scope is empty.")
        features = [rows[g["iso_class_id"]] for g in targets]
        start = perf_counter()
        initial = predict_b4_model(angle, features)
        probabilities = predict_b4_model(success, features)
        timings.append({"fit_id": angle["fit_id"], "success_fit_id": success["fit_id"],
                        "graphs": len(targets), "prediction_seconds": perf_counter() - start})
        for graph, point, probability in zip(targets, initial, probabilities, strict=True):
            row = {**point, "success_probability": probability["success_probability"],
                   "raw_success_prediction": probability["raw_prediction"],
                   "iso_class_id": graph["iso_class_id"], "p": angle["p"], "regime": angle["regime"],
                   "fold": angle["fold"], "feature_group": angle["group"], "shuffle_seed": angle["shuffle_seed"],
                   "fit_id": angle["fit_id"], "success_fit_id": success["fit_id"],
                   "feature_artifact_id": loaded["features"]["feature_artifact_id"],
                   "topology_hash": rows[graph["iso_class_id"]]["topology_hash"]}
            row["prediction_id"] = digest(row)
            predictions.append(row)
    prepared = {"version": "b4-prepared-v1", "library_id": library["library_id"],
        "library_manifest_sha256": manifest["library_manifest_sha256"], "split_digest": manifest["split_digest"],
        "source": manifest["source"], "settings": manifest["settings"], "fit_set_id": manifest["fit_set_id"],
        "fit_set_manifest": manifest, "compute_environment": manifest["compute_environment"],
        "feature_artifact_id": loaded["features"]["feature_artifact_id"], "models": models,
        "training_binding": loaded["training"]["reference_binding"],
        "training_reference_costs": loaded["training"]["reference_costs"],
        "success_label_costs": loaded["training"]["success_label_costs"],
        "feature_costs": {k: loaded["features"][k] for k in ("construction_seconds", "historical_feature_seconds")},
        "artifact_io": {"fitting": manifest["artifact_io"], "prediction_file_write_seconds": None},
        "predictions": predictions, "prediction_timings": timings}
    prepared["preparation_id"] = digest(prepared)
    write_once_json(output, prepared)
    return prepared


def read_b4_inputs(directory, manifest):
    prepared = read_json(Path(directory) / "prepared.json")
    binding = read_json(Path(directory) / "reference_binding.json")
    _b4_check_identity(prepared, "preparation_id")
    expected = {"preparation_id": prepared["preparation_id"], "prepared_digest": digest(prepared),
                "reference_binding_digest": digest(binding)}
    if manifest["config"].get("b4_inputs") != expected or prepared["source"] != manifest["source"]:
        raise ValueError("B4 preparation/source/reference binding changed.")
    return prepared, binding


def build_b4_tasks(library, config, prepared, binding):
    """Plan from frozen predictions only; no refit, inference or score-based selection."""
    from .learning import _b4_settings, b4_decode_angles, select_b2_graphs, validate_b4_model

    specs = b4_specifications(config)
    _b4_check_identity(prepared, "preparation_id")
    fit_manifest = prepared["fit_set_manifest"]
    _b4_fit_manifest(fit_manifest)
    if (any(fit_manifest.get(k) != prepared[k] for k in ("fit_set_id", "source", "library_id",
            "library_manifest_sha256", "split_digest", "settings", "compute_environment"))
            or fit_manifest["source_batch_id"] != prepared["training_binding"]["source_batch_id"]):
        raise ValueError("B4 embedded fit manifest/provenance changed.")
    if (prepared["version"] != "b4-prepared-v1" or prepared["library_id"] != library["library_id"]
            or prepared["split_digest"] != digest(library["split_summary"])
            or any(config.get(k) != v for k, v in prepared["settings"].items())
            or binding["library_id"] != library["library_id"]
            or binding["source_batch_id"] != prepared["training_binding"]["source_batch_id"]):
        raise ValueError("B4 preparation/settings/library mismatch.")
    models = prepared["models"]
    if len(models) != 2 * len(specs) or len({m["fit_id"] for m in models}) != len(models):
        raise ValueError("B4 requires the complete unique model design.")
    lookup = {}
    for model, spec, name in zip(models, [{**s, "head": h} for s in specs for h in ("angles", "success")],
                                 fit_manifest["model_files"], strict=True):
        _b4_check_identity(model, "fit_id")
        validate_b4_model(model)
        ids = [g["iso_class_id"] for g in select_b2_graphs(library, spec["regime"], spec["fold"], "training")]
        if (any(model.get(k) != v for k, v in spec.items()) or model["training_graph_ids"] != ids
                or model["source"] != prepared["source"] or model["library_id"] != library["library_id"]
                or model["split_digest"] != prepared["split_digest"]
                or model["settings"] != _b4_settings(config.get("learning"))
                or hashlib.sha256((json.dumps(model, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
                   != fit_manifest["files"][name]
                or model["feature_artifact_id"] != prepared["feature_artifact_id"]):
            raise ValueError("B4 model training membership/source/design mismatch.")
        lookup[model["fit_id"]] = model
    refs = {(r["iso_class_id"], r["p"]): r for r in binding["references"]}
    if len(refs) != len(binding["references"]):
        raise ValueError("Duplicate B4 scoring reference.")
    for ref in refs.values():
        _b4_check_identity(ref, "reference_id")
        if not ref["complete"] or ref["library_id"] != library["library_id"] or ref["batch_id"] != binding["source_batch_id"]:
            raise ValueError("B4 requires complete original scoring references.")
    for ref in prepared["training_binding"]["references"]:
        if refs.get((ref["iso_class_id"], ref["p"])) != ref:
            raise ValueError("B4 training reference changed after fitting.")
    predictions = prepared["predictions"]
    expected = {(m["fit_id"], g["iso_class_id"]): g for m in models if m["head"] == "angles"
                for g in select_b2_graphs(library, m["regime"], m["fold"], "evaluation")}
    if len(predictions) != len(expected) or {(r["fit_id"], r["iso_class_id"]) for r in predictions} != set(expected):
        raise ValueError("B4 requires every predeclared frozen prediction exactly once.")
    tasks = []
    for row in predictions:
        _b4_check_identity(row, "prediction_id")
        model, success = lookup[row["fit_id"]], lookup.get(row["success_fit_id"])
        graph = expected[row["fit_id"], row["iso_class_id"]]
        p = model["p"]
        if (success is None or success["head"] != "success" or any(success[k] != model[k] for k in ("group", "p", "regime", "fold", "shuffle_seed"))
                or any(row[k] != model[k] for k in ("p", "regime", "fold", "shuffle_seed"))
                or row["feature_group"] != model["group"] or row["feature_artifact_id"] != prepared["feature_artifact_id"]
                or row["topology_hash"] != digest({"n": graph["n"], "edges": graph["edges"]})
                or np.shape(row["theta0"]) != (2 * p,) or not np.all(np.isfinite(row["theta0"]))
                or not 0 <= row["success_probability"] <= 1):
            raise ValueError("B4 prediction geometry/model/schema changed.")
        decoded = b4_decode_angles(row["raw_prediction"], model["model"]["anchor"]["theta"])
        # The bound JSON bytes stay exact; recomputed libm values may differ by
        # an ulp across CPU platforms when reading a returned checkpoint.
        if (any(row.get(k) != decoded[k] for k in ("fallback", "trigger_coordinates"))
                or np.shape(row["rho"]) != (2 * p,)
                or not np.allclose(row["rho"], decoded["rho"], rtol=1e-12, atol=1e-15)
                or not np.allclose(row["theta0"], decoded["theta0"], rtol=0., atol=1e-12)
                or not math.isfinite(row["raw_success_prediction"])
                or row["success_probability"] != float(np.clip(row["raw_success_prediction"], 0., 1.))
                or type(row["unseen_type_count"]) is not int or not 0 <= row["unseen_type_count"] <= 2300
                or not math.isfinite(row["unseen_type_fraction"]) or not 0 <= row["unseen_type_fraction"] <= 1):
            raise ValueError("B4 decoded angle/fallback/success diagnostics disagree with frozen raw predictions.")
        budget = config["optimizer"][str(p)]
        OptimizerSettings(**budget)
        if budget != binding["source_config"]["optimizer"][str(p)]:
            raise ValueError("B4 cannot change the original B1 optimization budget.")
        ref = refs.get((graph["iso_class_id"], p))
        if ref is None:
            raise ValueError("B4 evaluation scoring reference missing.")
        task = {"task_version": "plan-a-b4-tasks-v1", "library_id": library["library_id"],
            "iso_class_id": graph["iso_class_id"], "n": graph["n"], "p": p, "graph_split": "evaluation",
            "kind": "optimization", "experiment_role": "warm_start", "pool": None, "method": "B4",
            "restart_id": 0, "seed": int(digest({"namespace": config["seed_namespace"], "seed": config["seed"],
                "prediction_id": row["prediction_id"]})[:32], 16), "theta0": row["theta0"], "budget": budget,
            "protocol_version": config["protocol_version"], "regime": row["regime"], "fold": row["fold"],
            "feature_group": row["feature_group"], "shuffle_seed": row["shuffle_seed"], "fit_id": row["fit_id"],
            "prediction_id": row["prediction_id"], "reference_id": ref["reference_id"]}
        task["task_id"] = digest(task)
        tasks.append(task)
    return sorted(tasks, key=lambda r: r["task_id"])


def save_b4_batch(output, predictions_path, batch, references_directory, attempts_directory, settings):
    prepared = read_json(predictions_path)
    source, library, _ = read_batch(batch)
    if prepared["library_manifest_sha256"] != source["library_manifest_sha256"]:
        raise ValueError("B4 predictions used a different frozen library.")
    ids = sorted({r["iso_class_id"] for r in prepared["training_binding"]["references"]}
                 | {r["iso_class_id"] for r in prepared["predictions"]})
    _, _, binding, _ = audit_b2_references(batch, references_directory, attempts_directory, ids)
    inherited = {k: source["config"][k] for k in ("optimizer", "max_attempts", "expected_library",
                 "require_full_design", "design", "require_split_coverage") if k in source["config"]}
    if set(settings) & set(inherited):
        raise ValueError("B4 settings cannot override the original optimizer/library contract.")
    validate_b1_comparison({"library_id": library["library_id"], "source": prepared["source"],
        "config": {**inherited, **settings}}, source, {"environment": prepared["compute_environment"]},
        read_json(Path(attempts_directory) / "manifest.json"))
    return save_batch(output, (Path(batch) / source["library_path"]).resolve(), {**inherited, **settings},
                      b4_prepared=prepared, reference_binding=binding)
