"""Plan, run and summarize the finite-shot B1–B5 comparison using immutable records.

This entry point reuses the library, B2-fit and record contracts.
It reuses frozen angle fits. Reference execution uses the existing worker.
"""

import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qaoa_study.experiments import (digest, environment, prepare_shot_starts, read_batch,
                                    read_references, save_batch, source_identity)
from qaoa_study.graphs import graph_from_record
from qaoa_study.optimize import ShotOptimizerSettings, optimize_qaoa_shots, random_angles
from qaoa_study.records import (EXECUTION_FAILURES, read_json, read_library, read_run,
                                resolve_success_metric, save_run, success_threshold, write_once_json)


def _seed(config, *parts):
    return int(digest([config["seed_namespace"], config["seed"], *parts])[:32], 16)


def plan_benchmark(b1_batch, b2_fits, b4_fits, b5_fits, output, *,
                   libraries=(), references=(), depths=(1, 2), repetitions=None,
                   settings=None, methods=("COBYLA", "SPSA"),
                   modes=("finite_shots", "exact_control"), backend="default.qubit",
                   seed=20261001, graph_ids=None, initializations=("B1", "B2", "B3", "B4", "B5"),
                   b3_rules=None):
    """Freeze selected evaluation graphs and initialization rules from intact sources."""
    baseline, original, baseline_tasks = read_batch(b1_batch)
    loaded = [original, *[read_library(path) for path in libraries]]
    depths, methods, modes = list(depths), list(methods), list(modes)
    if (not depths or len(set(depths)) != len(depths) or any(type(p) is not int or p < 1 for p in depths)
            or not methods or len(set(methods)) != len(methods) or not set(methods) <= {"COBYLA", "COBYQA", "SPSA"}
            or not modes or len(set(modes)) != len(modes) or not set(modes) <= {"finite_shots", "exact_control"}):
        raise ValueError("Use distinct supported depths, optimizers and evaluation modes.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    initializations = list(initializations)
    if (not initializations or len(set(initializations)) != len(initializations)
            or not set(initializations) <= {"B1", "B2", "B3", "B4", "B5"}):
        raise ValueError("Use distinct supported initialization methods.")
    selected_ids = None
    if graph_ids is not None:
        if not isinstance(graph_ids, (list, tuple)) or not graph_ids or len(set(graph_ids)) != len(graph_ids):
            raise ValueError("Graph selection must contain distinct held-out evaluation graph IDs.")
        selected_ids = set(graph_ids)
    if backend not in ("default.qubit", "lightning.qubit", "lightning.gpu"):
        raise ValueError("Unsupported finite-shot backend.")
    settings = settings or ShotOptimizerSettings()
    required = max(4 * p + 1 if method == "COBYQA" else 2 * p + 2 if method == "COBYLA" else 2
                   for p in depths for method in methods)
    if settings.total_shots // settings.shots_per_evaluation < required:
        raise ValueError("Shot budget cannot initialize every declared optimizer/depth.")
    prior, graphs, source_library = set(), [], {}
    for library in loaded:
        identity = library["library_id"]
        if identity in prior or not prior <= set(library.get("excluded_library_ids", [])):
            raise ValueError("Additions must exclude all prior libraries, including the original B2 training library.")
        prior.add(identity)
        if library is not original and any(g["graph_split"] != "evaluation" for g in library["graphs"]):
            raise ValueError("New-family transfer libraries must remain wholly outside training.")
        for graph in library["graphs"]:
            if graph["tier"] != 2 or graph["graph_split"] != "evaluation":
                continue
            if selected_ids is not None and graph["iso_class_id"] not in selected_ids:
                continue
            if graph["iso_class_id"] in source_library:
                raise ValueError("Duplicate graph identity.")
            if graph.get("exact_status") != "computed" or not graph.get("C_star"):
                raise ValueError("Exact MaxCut annotations are required for complete comparison.")
            source_library[graph["iso_class_id"]] = identity
            graphs.append(graph)
    if selected_ids is not None and selected_ids != set(source_library):
        raise ValueError("Graph selection contains IDs outside the held-out evaluation libraries.")
    if not graphs:
        raise ValueError("No evaluation graphs.")
    frozen_refs = {}
    for directory in references:
        for key, ref in read_references(directory).items():
            if key[0] in source_library and key[1] in depths:
                if ref["library_id"] != source_library[key[0]] or not ref["complete"]:
                    raise ValueError("Scoring references must be complete and belong to the target library.")
                if key in frozen_refs and frozen_refs[key] != ref:
                    raise ValueError("Conflicting reference versions.")
                frozen_refs[key] = ref
    missing = sorted((g["iso_class_id"], p) for g in graphs for p in depths
                     if (g["iso_class_id"], p) not in frozen_refs)
    if missing:
        raise ValueError(f"Missing {len(missing)} scoring references; use reference-plan and the existing reference worker first.")
    rule_options = {"b3_rules": b3_rules} if b3_rules is not None else {}
    prepared = prepare_shot_starts(loaded, b2_fits, b4_fits, b5_fits, depths=depths,
        graph_ids=graph_ids, initializations=[method for method in initializations if method != "B1"], **rule_options)
    starts = prepared["starts"]
    pools = defaultdict(dict)
    for task in baseline_tasks:
        if task["experiment_role"] == "evaluation" and task["iso_class_id"] in source_library:
            pools[task["iso_class_id"], task["p"]][task["restart_id"]] = {
                "theta0": task["theta0"], "seed": task["seed"], "source_task_id": task["task_id"]}
    for graph in (graphs if "B1" in initializations else []):
        identity = graph["iso_class_id"]
        scope = "original_evaluation" if source_library[identity] == original["library_id"] else "external_family_transfer"
        for p in depths:
            starts.append({"iso_class_id": identity, "p": p, "initialization": "B1",
                "variant": "random", "regime": "random", "fold": None, "feature_group": None,
                "shuffle_seed": None, "theta0": None, "source_id": baseline["batch_id"],
                "target_scope": scope,
                "random_pool": {str(i): row for i, row in sorted(pools[identity, p].items()) if i < repetitions}})
    for start in starts:
        if start["iso_class_id"] not in source_library or start["p"] not in depths:
            raise ValueError("Initialization outside the declared evaluation scope.")
        if start["initialization"] != "B1" and (
                np.shape(start["theta0"]) != (2 * start["p"],) or not np.all(np.isfinite(start["theta0"]))):
            raise ValueError("Invalid frozen initial angles.")
        if "start_id" not in start:
            start["start_id"] = digest(start)
    if len({s["start_id"] for s in starts}) != len(starts):
        raise ValueError("Duplicate initialization conditions.")
    if graph_ids is not None:
        order = {identity: index for index, identity in enumerate(graph_ids)}
        starts.sort(key=lambda start: (order[start["iso_class_id"]], start["p"], start["initialization"],
                    start.get("variant") or "", start["start_id"]))
        graphs.sort(key=lambda graph: order[graph["iso_class_id"]])
    coverage = defaultdict(set)
    for start in starts:
        coverage[start["iso_class_id"], start["p"]].add(start["initialization"])
    if any(coverage[g["iso_class_id"], p] != set(initializations) for g in graphs for p in depths):
        raise ValueError("Every evaluation graph/depth must include exactly the declared initializations.")
    settings_dict = asdict(settings)
    config = {"depths": depths, "optimizers": methods, "evaluation_modes": modes,
              "repetitions": repetitions, "backend": backend, "optimizer": settings_dict,
              "initializations": initializations,
              "seed": seed, "seed_namespace": "finite-shot-b1-b5-v2",
              "max_attempts": baseline["config"]["max_attempts"],
              "study_role": "frozen_model_comparison", "success_metric": resolve_success_metric()}
    manifest = {"version": "shot-benchmark-v2", "config": config, "success_metric": config["success_metric"],
                "baseline_batch_id": baseline["batch_id"], "libraries": [lib["library_id"] for lib in loaded],
                "graphs": graphs, "starts": starts, "preparation": {k: v for k, v in prepared.items() if k != "starts"},
                "references": [frozen_refs[key] for key in sorted(frozen_refs)], "source": source_identity()}
    optimized = len(starts) * repetitions * len(methods)
    fixed = sum(s["initialization"] != "B1" for s in starts) * repetitions
    tasks = optimized * len(modes) + fixed
    manifest["budget"] = {"graphs": len(graphs), "starts": len(starts), "tasks": tasks,
        "by_initialization": dict(Counter(s["initialization"] for s in starts)),
        "optimization_shots_cap": optimized * settings.total_shots * int("finite_shots" in modes),
        "validation_shots_cap": tasks * settings.validation_shots,
        "scope": "Full declared evaluation graphs; historical references/models and new reference construction are separate costs."}
    manifest["plan_id"] = digest(manifest)
    write_once_json(output, manifest)
    return manifest


def replan_benchmark(parent_path, output, *, depth=2, sizes=(12, 16), per_cell=4,
                     selection_seed=20261026):
    """Create an independent COBYQA plan from archived B1-B3 inputs and references.

    Frozen starts and the seed policy are inherited. Only the new plan binds
    current execution code; the parent's source identity remains in provenance.
    """
    parent = _read_plan(parent_path)
    if depth not in parent["config"]["depths"] or type(per_cell) is not int or per_cell < 1:
        raise ValueError("Use an archived depth and a positive per-cell graph count.")
    families = ("regular", "er", "ba", "sbm", "ws", "rgg")
    pools = defaultdict(list)
    for graph in parent["graphs"]:
        family = graph["provenance"][0]["family"]
        if family in families and graph["n"] in sizes:
            pools[graph["representative_cell"]].append(graph)
    cells = Counter((rows[0]["provenance"][0]["family"], rows[0]["n"]) for rows in pools.values())
    if cells != Counter({(family, n): 2 for family in families for n in sizes}):
        raise ValueError("Archived inputs must cover both settings of every requested family and size.")
    selected = []
    for cell, rows in sorted(pools.items()):
        if len(rows) < per_cell:
            raise ValueError(f"{cell}: need {per_cell} archived evaluation graphs; found {len(rows)}.")
        selected.append(sorted(rows, key=lambda g: digest([selection_seed, cell, g["iso_class_id"]]))[:per_cell])
    graphs = [rows[i] for i in range(per_cell) for rows in selected]
    order = {g["iso_class_id"]: i for i, g in enumerate(graphs)}
    starts = [s for s in parent["starts"] if s["iso_class_id"] in order and s["p"] == depth
              and s["regime"] == "random" and s["initialization"] in ("B1", "B2", "B3")
              and (s["initialization"] != "B2" or s["variant"] in ("medoid", "aligned_median"))]
    for graph_id in order:
        kinds = Counter((s["initialization"], s["variant"] if s["initialization"] == "B2" else None)
                        for s in starts if s["iso_class_id"] == graph_id)
        if kinds != Counter({("B1", None): 1, ("B2", "medoid"): 1,
                             ("B2", "aligned_median"): 1, ("B3", None): 1}):
            raise ValueError(f"Archived B1-B3 starts are incomplete for {graph_id}.")
    starts.sort(key=lambda s: (order[s["iso_class_id"]], s["initialization"], s["variant"]))
    refs = [r for r in parent["references"] if r["iso_class_id"] in order and r["p"] == depth]
    if len(refs) != len(graphs) or not all(r["complete"] for r in refs):
        raise ValueError("Selected graphs require complete archived scoring references.")
    manifest = deepcopy(parent)
    manifest.pop("plan_id")
    manifest.update(graphs=graphs, starts=starts, references=refs, source=source_identity(),
                    parent_plan={"plan_id": parent["plan_id"], "source": parent["source"]},
                    selection={"sizes": list(sizes), "per_cell": per_cell, "seed": selection_seed})
    config = manifest["config"]
    config.update(depths=[depth], optimizers=["COBYQA"], initializations=["B1", "B2", "B3"],
                  evaluation_modes=["finite_shots", "exact_control"], study_role="archived_start_cobyqa_comparison")
    settings = ShotOptimizerSettings(**config["optimizer"])
    if settings.total_shots // settings.shots_per_evaluation < 4 * depth + 1:
        raise ValueError("Archived query budget is too small to initialize COBYQA.")
    optimized = len(starts) * config["repetitions"]
    tasks = optimized * 2 + 3 * len(graphs) * config["repetitions"]
    manifest["budget"] = {"graphs": len(graphs), "starts": len(starts), "tasks": tasks,
        "by_initialization": dict(Counter(s["initialization"] for s in starts)),
        "optimization_shots_cap": optimized * settings.total_shots,
        "validation_shots_cap": tasks * settings.validation_shots,
        "scope": "New COBYQA runs; all selected references and starting angles are inherited."}
    manifest["plan_id"] = digest(manifest)
    if Path(output).exists():
        if _read_plan(output) != manifest:
            raise ValueError("A different plan already exists at the output path.")
    else:
        write_once_json(output, manifest)
    return manifest


def iter_tasks(plan, graph_ids=None):
    """Generate tasks lazily; large repetition counts do not expand the plan file."""
    config = plan["config"]
    known = {g["iso_class_id"] for g in plan["graphs"]}
    if graph_ids is not None and not set(graph_ids) <= known:
        raise ValueError("Requested graph is absent from this plan.")
    for start in plan["starts"]:
        if graph_ids is not None and start["iso_class_id"] not in graph_ids:
            continue
        for repetition in range(config["repetitions"]):
            initial_seed, theta, old_task = None, start["theta0"], None
            if start["initialization"] == "B1":
                original = start["random_pool"].get(str(repetition))
                initial_seed = original["seed"] if original else _seed(config, start["iso_class_id"], start["p"], repetition, "initialization")
                theta = original["theta0"] if original else random_angles(start["p"], initial_seed).tolist()
                old_task = original["source_task_id"] if original else None
            choices = [(method, mode) for mode in config["evaluation_modes"] for method in config["optimizers"]]
            if start["initialization"] != "B1":
                choices.append(("fixed", "finite_shots"))
            for method, mode in choices:
                key = [start["start_id"], repetition, method, mode]
                task = {k: start.get(k) for k in ("iso_class_id", "p", "initialization", "variant",
                        "regime", "fold", "feature_group", "shuffle_seed", "target_scope", "source_id", "start_id")}
                task.update(repetition=repetition, method=method, evaluation_mode=mode, theta0=theta,
                            initialization_seed=initial_seed, original_B1_task_id=old_task,
                            seeds={role: _seed(config, *key, role) for role in ("measurement", "perturbation", "validation")})
                task["seeds"]["perturbation"] = _seed(config, start["start_id"], repetition, method, "perturbation")
                task["task_id"] = digest(task)
                yield task


def plan_references(b1_batch, library_path, output):
    """Reuse the original reference algorithm on a separate new-family library."""
    baseline, original, _ = read_batch(b1_batch)
    library = read_library(library_path)
    if original["library_id"] not in library.get("excluded_library_ids", []):
        raise ValueError("Reference extension must exclude the original library.")
    config = deepcopy(baseline["config"])
    config.update(purpose="new_family_reference_completion", require_full_design=False,
                  require_split_coverage=False, tier1_restarts=1, evaluation_restarts=1, gradient_samples=1,
                  expected_library={"graphs": len(library["graphs"]),
                    "tier2": sum(g["tier"] == 2 for g in library["graphs"]),
                    "training": sum(g["graph_split"] == "training" for g in library["graphs"]),
                    "evaluation": sum(g["graph_split"] == "evaluation" for g in library["graphs"])})
    config.pop("design", None)
    config.pop("kind", None)
    # The mature worker's --roles reference selects only the missing reference pool.
    return save_batch(output, library_path, config)


def _read_plan(path):
    plan = read_json(path)
    if plan.get("version") != "shot-benchmark-v2" or digest(
            {k: v for k, v in plan.items() if k != "plan_id"}) != plan.get("plan_id"):
        raise ValueError("Shot plan identity changed.")
    return plan


def run_path(output, task):
    return Path(output) / digest(task["iso_class_id"]) / (task["task_id"] + ".json")


def _completed(output, task, execution):
    path = run_path(output, task)
    if not path.exists():
        return None
    record = read_run(path, require_integrity=True)
    if record.get("task") != task or record.get("execution_id") != execution["execution_id"]:
        raise ValueError("Saved run belongs to different task inputs or execution provenance.")
    return record


def _shot_failed(result):
    return bool(result.get("error") or result.get("validation_error")
                or result.get("exact_audit_error") or result["stop_reason"] in EXECUTION_FAILURES)


def _resource_failure(result):
    """Retry allocation/busy-device faults, never an optimizer's ordinary stop."""
    messages = [result.get(key) or "" for key in ("error", "validation_error", "exact_audit_error")]
    messages += [row.get("error") or "" for row in result.get("trace", [])]
    text = "".join(character for character in " ".join(messages).lower() if character.isalnum())
    return any(token in text for token in ("outofmemory", "memoryallocation", "allocfailed",
                                           "badalloc", "devicesarebusy", "deviceisbusy"))


def _attempt_paths(output, task, attempt):
    path = run_path(output, task)
    suffix = "" if attempt == 1 else f".attempt-{attempt}"
    return (path.parent / "started" / f'{task["task_id"]}{suffix}.json',
            path.parent / "resource-faults" / f'{task["task_id"]}.{attempt}.json')


def _execute_shot_task(item):
    """One process owns one task and writes its complete result before returning."""
    task, config = item["task"], item["config"]
    result = optimize_qaoa_shots(graph_from_record(item["graph"]), task["p"], task["theta0"],
        ShotOptimizerSettings(**config["optimizer"]), method=task["method"],
        evaluation_mode=task["evaluation_mode"], measurement_seed=task["seeds"]["measurement"],
        perturbation_seed=task["seeds"]["perturbation"], validation_seed=task["seeds"]["validation"],
        backend=config["backend"], audit_exact=True)
    result.update(task=task, execution_id=item["execution_id"], run_completed=True,
        attempt_number=item["attempt_number"], previous_interrupted_attempts=item.get("previous_interrupted_attempts", 0),
        previous_resource_failures=item.get("previous_resource_failures", 0),
        device_identity=item["device_identity"], worker_pid=os.getpid(),
        execution_resources={"workers_requested": item["parallel_workers"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "timing_scope": "Per-task elapsed time under this run's resource sharing; use shots for algorithm cost."})
    resource_error = _resource_failure(result)
    terminal = not resource_error or item["attempt_number"] >= config["max_attempts"]
    if resource_error:
        _, fault = _attempt_paths(item["output"], task, item["attempt_number"])
        save_run(fault, result)
    if terminal:
        save_run(run_path(item["output"], task), result)
    return {"failed": _shot_failed(result), "resource_error": resource_error, "terminal": terminal}


@contextmanager
def run_lock(path):
    """The operating system releases this lock if the owning process exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def run_benchmark(plan_path, output, *, execute=False, **options):
    with run_lock(Path(output) / ".run.lock") if execute else nullcontext():
        return _run_benchmark(plan_path, output, execute=execute, **options)


def _run_benchmark(plan_path, output, *, execute=False, max_tasks=None, graph_ids=None,
                  retry_interrupted=False, progress=None, should_stop=None, parallel=None):
    """Resume immutable records; optional callbacks report progress and stop between tasks."""
    plan, output = _read_plan(plan_path), Path(output)
    if source_identity() != plan["source"]:
        raise ValueError("Working source/configuration changed since planning; create a new plan.")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer.")
    config = plan["config"]
    env = environment(config["backend"])
    device_identity = env.pop("device_identity", None)
    env.update(shots="per-task fixed batches", interface=None, diff_method=None, device_vjp=False)
    execution = {"plan_id": plan["plan_id"], "source_id": plan["source"]["source_id"], "environment": env}
    execution["execution_id"] = digest(execution)
    execution_path = output / "execution.json"
    if execution_path.exists():
        if read_json(execution_path) != execution:
            raise ValueError("Output belongs to another plan, source or compute environment.")
    elif execute:
        if output.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Refusing an unrecognized nonempty output directory.")
        write_once_json(execution_path, execution)
    graphs = {g["iso_class_id"]: g for g in plan["graphs"]}
    selected = sum(config["repetitions"] * (len(config["optimizers"]) * len(config["evaluation_modes"])
                   + int(start["initialization"] != "B1")) for start in plan["starts"]
                   if graph_ids is None or start["iso_class_id"] in graph_ids)
    report = {"planned": plan["budget"]["tasks"], "selected": selected, "completed": 0,
              "executed": 0, "interrupted": 0, "retried_interrupted": 0, "execution_failures": 0,
              "resource_failures": 0, "resource_retries": 0,
              "scanned": 0, "stopped": False, "active_task": None}
    from qaoa_study.parallel import ParallelSettings, run_tasks, estimate_statevector_bytes
    parallel = parallel or ParallelSettings()
    started = perf_counter()
    admitted = 0
    active = {}
    retry_queue = deque()

    def pending():
        for task in iter_tasks(plan, graph_ids):
            if should_stop is not None and should_stop():
                report["stopped"] = True
                break
            report["scanned"] += 1
            completed = _completed(output, task, execution)
            if completed is not None:
                report["completed"] += 1
                report["execution_failures"] += int(_shot_failed(completed))
                continue
            interrupted, resource_failures = 0, 0
            for attempt_number in range(1, config["max_attempts"] + 1):
                marker, fault = _attempt_paths(output, task, attempt_number)
                if not marker.exists():
                    break
                if fault.exists():
                    failed = read_run(fault, require_integrity=True)
                    if failed["task"] != task or failed["execution_id"] != execution["execution_id"]:
                        raise ValueError("Saved resource attempt has different task inputs or execution provenance.")
                    resource_failures += 1
                    if attempt_number == config["max_attempts"]:
                        # Complete publication after a crash between fault and canonical writes.
                        if execute:
                            save_run(run_path(output, task), failed)
                            report["completed"] += 1
                            report["execution_failures"] += 1
                        marker = None
                        break
                else:
                    interrupted += 1
                    if not retry_interrupted or attempt_number == config["max_attempts"]:
                        report["interrupted"] += 1
                        marker = None
                        break
            if marker is None:
                continue
            if not execute or (max_tasks is not None and admitted >= max_tasks):
                continue
            yield {"task": task, "graph": graphs[task["iso_class_id"]], "config": config,
                   "execution_id": execution["execution_id"], "device_identity": device_identity,
                   "attempt_number": attempt_number, "marker": str(marker), "output": str(output),
                   "previous_interrupted_attempts": interrupted, "previous_resource_failures": resource_failures,
                   "parallel_workers": parallel.workers}

    def submitted(item):
        nonlocal admitted
        if max_tasks is not None and admitted >= max_tasks:
            return False
        task = item["task"]
        write_once_json(item["marker"], {"task_id": task["task_id"],
            "execution_id": execution["execution_id"], "attempt_number": item["attempt_number"],
            "started_utc": datetime.now(timezone.utc).isoformat()})
        admitted += 1
        active[task["task_id"]] = {k: task[k] for k in
            ("task_id", "iso_class_id", "p", "initialization", "method", "evaluation_mode", "repetition")}
        report.update(active_task=active[task["task_id"]], active_tasks=list(active.values()))
        if progress is not None:
            progress(dict(report))

    def resources(snapshot):
        report["resources"] = snapshot
        if progress is not None and (active or snapshot["waiting_for_memory"] or snapshot["waiting_for_utilization"]):
            progress(dict(report))

    for item, result in run_tasks(_execute_shot_task, pending(), settings=parallel,
            backend=config["backend"] if execute else "default.qubit", should_stop=should_stop, on_submit=submitted,
            on_resources=resources, retry_queue=retry_queue,
            resource_history=lambda item: item["previous_resource_failures"],
            memory_bytes=lambda item: estimate_statevector_bytes(
                item["graph"]["n"], state_copies=parallel.state_copies,
                worker_overhead_mib=parallel.worker_overhead_mib)):
        report["executed"] += 1
        report["resource_failures"] += int(result["resource_error"])
        report["resource_retries"] += int(item["previous_resource_failures"] > 0)
        report["retried_interrupted"] += int(item["previous_interrupted_attempts"] > 0)
        if result["terminal"]:
            report["completed"] += 1
            report["execution_failures"] += int(result["failed"])
        elif max_tasks is None or admitted < max_tasks:
            next_attempt = item["attempt_number"] + 1
            marker, _ = _attempt_paths(output, item["task"], next_attempt)
            retry_queue.append({**item, "attempt_number": next_attempt, "marker": str(marker),
                                "previous_resource_failures": item["previous_resource_failures"] + 1})
        active.pop(item["task"]["task_id"], None)
        report.update(active_task=next(iter(active.values()), None), active_tasks=list(active.values()))
        if progress is not None:
            progress(dict(report))
    report["stopped"] |= bool(should_stop is not None and should_stop())
    report["session_seconds"] = perf_counter() - started
    report["workers_requested"] = parallel.workers
    report["complete"] = report["completed"] == report["planned"]
    report["selected_complete"] = report["completed"] == report["selected"]
    report["missing"] = report["scanned"] - report["completed"] - report["interrupted"]
    report["unscanned"] = report["selected"] - report["scanned"]
    if progress is not None:
        progress(dict(report))
    return report


SUMMARY_KEYS = ("p", "initialization", "variant", "regime", "fold", "feature_group",
                "shuffle_seed", "method", "evaluation_mode", "target_scope", "family")
METRICS = ("C_initial_exact", "C_final_exact", "quality", "quality_gain", "terminal_success")


def summarize_benchmark(plan_path, output):
    """Stream attempts into graph means, retaining every planned denominator."""
    plan, output = _read_plan(plan_path), Path(output)
    execution = read_json(output / "execution.json")
    if execution["plan_id"] != plan["plan_id"] or digest(
            {k: v for k, v in execution.items() if k != "execution_id"}) != execution["execution_id"]:
        raise ValueError("Execution identity does not match the plan.")
    refs = {(r["iso_class_id"], r["p"]): r for r in plan["references"]}
    graphs = {g["iso_class_id"]: g for g in plan["graphs"]}
    per_graph, total_counts = {}, Counter()
    retry_counts, retry_attempts, retry_seconds = Counter(), 0, 0.0
    resource_paths = defaultdict(list)
    for identity in graphs:
        for path in (output / digest(identity) / "resource-faults").glob("*.json"):
            task_id, attempt = path.stem.rsplit(".", 1)
            resource_paths[task_id].append((int(attempt), path))
    failures, completed = 0, 0
    for task in iter_tasks(plan):
        graph = graphs[task["iso_class_id"]]
        context = {k: task.get(k) for k in SUMMARY_KEYS}
        context["family"] = graph["provenance"][0]["family"]
        key = (task["iso_class_id"], *context.values())
        row = per_graph.setdefault(key, {**context, "iso_class_id": task["iso_class_id"], "n": graph["n"],
            "planned": 0, "completed": 0, "optimization_failures": 0, "validation_failures": 0,
            "exact_audit_failures": 0, "error_examples": [], "counts": Counter(),
            "elapsed_seconds": 0.0, "sums": Counter(), "available": Counter()})
        row["planned"] += 1
        result = _completed(output, task, execution)
        # Preserve resource-fault cost separately from the selected terminal result.
        for attempt, fault in resource_paths.get(task["task_id"], ()):
            if result is None or attempt < result.get("attempt_number", 1):
                failed_attempt = read_run(fault, require_integrity=True)
                if failed_attempt["task"] != task or failed_attempt["execution_id"] != execution["execution_id"]:
                    raise ValueError("Resource-attempt costs belong to another task or execution.")
                retry_counts.update(failed_attempt["counts"])
                retry_seconds += failed_attempt["elapsed_seconds"]
                retry_attempts += 1
        if result is None:
            continue
        completed += 1
        row["completed"] += 1
        opt_failed = bool(result.get("error") or result["stop_reason"] in EXECUTION_FAILURES)
        phase_errors = (result.get("error"), result.get("validation_error"), result.get("exact_audit_error"))
        failed = opt_failed or any(phase_errors)
        failures += int(failed)
        row["optimization_failures"] += int(opt_failed)
        row["validation_failures"] += int(bool(phase_errors[1]))
        row["exact_audit_failures"] += int(bool(phase_errors[2]))
        if failed and len(row["error_examples"]) < 3:
            row["error_examples"].append({"task_id": task["task_id"], "stop_reason": result["stop_reason"],
                                         "optimization": phase_errors[0], "validation": phase_errors[1], "exact_audit": phase_errors[2]})
        row["counts"].update(result["counts"])
        total_counts.update(result["counts"])
        row["elapsed_seconds"] += result["elapsed_seconds"]
        exact = result.get("exact_audit")
        if exact is not None and not opt_failed:
            initial, final = exact["C_initial"], exact["C_final"]
            optimum = graph["C_star"]
            ref = refs[task["iso_class_id"], task["p"]]
            values = (initial, final, final / optimum, (final - initial) / optimum,
                      float(final >= success_threshold(ref["C_ref"], optimum, plan["success_metric"])))
            for name, value in zip(METRICS, values, strict=True):
                row["sums"][name] += value
                row["available"][name] += 1
    grouped = defaultdict(list)
    for row in per_graph.values():
        for name in METRICS:
            row[name] = row["sums"][name] / row["planned"] if row["available"][name] == row["planned"] else None
        row.pop("sums")
        row.pop("available")
        row["counts"] = dict(row["counts"])
        grouped[tuple(row[k] for k in SUMMARY_KEYS)].append(row)
    # Reuse graph-level B1/B3 outcomes on each LOFO's actual target set; a
    # family-wide mean can include multi-family graphs excluded by that LOFO.
    lofo_targets = defaultdict(set)
    for start in plan["starts"]:
        if start["regime"] == "lofo":
            lofo_targets[start["p"], start["fold"]].add(start["iso_class_id"])
    for row in per_graph.values():
        if row["initialization"] in ("B1", "B3"):
            for (p, fold), identities in lofo_targets.items():
                if row["p"] == p and row["iso_class_id"] in identities:
                    view = {**row, "regime": "lofo", "fold": fold}
                    grouped[tuple(view[k] for k in SUMMARY_KEYS)].append(view)
    comparisons = []
    for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        comparisons.append({**dict(zip(SUMMARY_KEYS, key, strict=True)),
            "reuses_random_baseline": rows[0]["regime"] == "lofo" and rows[0]["initialization"] in ("B1", "B3"),
            "graphs": len(rows), "planned": sum(r["planned"] for r in rows),
            "completed": sum(r["completed"] for r in rows),
            **{name: float(np.mean([r[name] for r in rows])) if all(r[name] is not None for r in rows) else None
               for name in METRICS}})
    return {"plan_id": plan["plan_id"], "study_role": plan["config"]["study_role"],
            "complete": completed == plan["budget"]["tasks"],
            "assessments_complete": completed == plan["budget"]["tasks"] and failures == 0,
            "planned": plan["budget"]["tasks"], "completed": completed, "execution_failures": failures,
            "counts": dict(total_counts), "comparisons": comparisons, "graph_results": list(per_graph.values()),
            "resource_retry_attempts": retry_attempts, "resource_retry_counts": dict(retry_counts),
            "resource_retry_elapsed_seconds": retry_seconds,
            "counts_including_resource_retries": dict(total_counts + retry_counts),
            "weighting": "Repetitions within graph, then equal graph weights; any missing/faulted exact endpoint leaves its group metric null.",
            "scope": "Original evaluation and external-family transfer are separate; LOFO baselines reuse matched graph outcomes. Counts cover completed records; work before an interrupted record is unknown."}


def _status(root):
    path = root / "status.json"
    state = read_json(path) if path.exists() else {"state": "not_started"}
    try:
        with run_lock(root / "runner.lock"):
            active = False
    except OSError:
        active = True
    if state["state"] == "running" and not active:
        state["state"] = "interrupted"
    return {**state, "active_lock": active}


def _stamp(root, **values):
    import json
    path = root / "status.json"
    state = read_json(path) if path.exists() else {}
    state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+"\n", encoding="utf8")
    temporary.replace(path)


def _background_work(root, lock_fd):
    from qaoa_study.parallel import NvmlMemory, ParallelSettings, _worker_environment
    from scripts.verify_backend import verify_shots
    with os.fdopen(lock_fd, "a+b"):
        options = read_json(root / "runner.json")
        plan = _read_plan(options["plan"])
        _stamp(root, state="running", phase="verify", pid=os.getpid(), error=None,
               planned=plan["budget"]["tasks"], completed=0, active_tasks=[])
        try:
            if (root / "stop.request").exists():
                _stamp(root, state="paused", phase="work", stopped=True)
                return
            if source_identity() != plan["source"]:
                raise ValueError("The checkout differs from this plan. Create a new plan for updated code.")
            parallel = ParallelSettings(**options["parallel"])
            with ExitStack() as stack:
                if plan["config"]["backend"] == "lightning.gpu":
                    probe = NvmlMemory(parallel.gpu_device)
                    stack.callback(probe.close)
                    stack.enter_context(_worker_environment(probe.uuid, False))
                receipt = root / "backend-verification.json"
                expected = {"source_id": plan["source"]["source_id"], "backend": plan["config"]["backend"],
                            "depths": plan["config"]["depths"]}
                if receipt.exists():
                    saved = read_json(receipt)
                    if any(saved[k] != v for k,v in expected.items()) or not saved["verification"]["passed"]:
                        raise ValueError("Backend verification belongs to a different plan configuration.")
                else:
                    if (root / "stop.request").exists():
                        _stamp(root, state="paused", phase="work", stopped=True)
                        return
                    checked = verify_shots(expected["backend"], expected["depths"])
                    if not checked["passed"]:
                        raise RuntimeError(f"Backend verification failed: {checked}")
                    write_once_json(receipt, {**expected, "verification": checked})
                result = run_benchmark(options["plan"], root / "attempts", execute=True,
                    retry_interrupted=True, parallel=parallel,
                    should_stop=lambda: (root / "stop.request").exists(),
                    progress=lambda report: _stamp(root, phase="work", **report))
            if result["complete"] and not result["execution_failures"]:
                _stamp(root, phase="summarize")
                summary = summarize_benchmark(options["plan"], root / "attempts")
                path = root / "summary.json"
                if not path.exists():
                    write_once_json(path, summary)
                _stamp(root, state="complete", phase="complete", **result)
            else:
                _stamp(root, state="paused", phase="work", **result)
        except BaseException as error:
            _stamp(root, state="paused", active_tasks=[], error=f"{type(error).__name__}: {error}")
            raise


def _lifecycle(args):
    import json
    from qaoa_study.parallel import ParallelSettings
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        while True:
            state = _status(root)
            if state["state"] == "not_started" and not state["active_lock"]:
                print(json.dumps(state, indent=2) if args.json else
                      f"Not started: no run status exists in {root}. "
                      "Run start first; status only monitors an existing run.", flush=True)
                return
            resources = state.get("resources", {})
            waiting = [name for name in ("memory", "utilization") if resources.get("waiting_for_" + name)]
            print(json.dumps(state,indent=2) if args.json else
                f'{state.get("completed",0)}/{state.get("planned","?")}  {state["state"]}  '
                f'{state.get("phase","")}  active={len(state.get("active_tasks",[]))}  '
                f'faults={state.get("execution_failures",0)}'
                + (f'  limit={resources["worker_limit"]}' if "worker_limit" in resources else "")
                + (f'  waiting={"+".join(waiting)}' if waiting else "")
                + (f'  {state["error"]}' if state.get("error") else ""), flush=True)
            if args.watch <= 0 or not state["active_lock"]:
                if args.watch > 0 and not args.json and state["state"] != "complete":
                    print("Monitoring ended: no active runner. Use start to launch or resume.", flush=True)
                return
            try:
                time.sleep(args.watch)
            except KeyboardInterrupt:
                return
    if args.command == "stop":
        if _status(root)["active_lock"]:
            (root / "stop.request").touch()
            print("Stop requested. Active tasks will finish and save; no new tasks will start.")
        else:
            print("Already stopped.")
        return
    if sys.platform != "linux":
        raise ValueError("Background start runs on Linux; use run --execute for foreground CPU checks.")
    if args.command == "_background":
        return _background_work(root,args.lock_fd)
    if _status(root)["active_lock"]:
        print(json.dumps(_status(root),indent=2))
        return
    path = root / "runner.json"
    saved = read_json(path) if path.exists() else None
    plan_path = str(Path(args.plan).resolve()) if args.plan else saved["plan"] if saved else None
    if not plan_path:
        raise ValueError("First start requires --plan PATH; later starts resume the same root.")
    plan = _read_plan(plan_path)
    if saved and saved["plan_id"] != plan["plan_id"]:
        raise ValueError("Use a new root for a different scientific plan.")
    options = {"plan":plan_path,"plan_id":plan["plan_id"], "parallel":asdict(
        _parallel_settings(args, saved["parallel"] if saved else None, default_workers=4))}
    log_path = root / "console.log"
    with run_lock(root / "runner.lock") as guard, log_path.open("ab",buffering=0) as log:
        path.write_text(json.dumps(options,indent=2)+"\n",encoding="utf8")
        (root / "stop.request").unlink(missing_ok=True)
        child = subprocess.Popen([sys.executable,"-B","-u",str(Path(__file__).resolve()),
            "_background","--root",str(root),"--lock-fd",str(guard.fileno())],
            cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(guard.fileno(),))
    print(json.dumps({"pid":child.pid,"log":str(log_path),"root":str(root)}))


def _parallel_settings(args, saved=None, *, default_workers=1):
    from qaoa_study.parallel import ParallelSettings
    values = asdict(ParallelSettings(workers=default_workers))
    values.update(saved or {})
    values.update({name: getattr(args, name) for name in values if hasattr(args, name)})
    return ParallelSettings(**values)


def _parallel_arguments(parser, workers=1, *, inherit=False):
    from qaoa_study.parallel import ParallelSettings
    for name, field in ParallelSettings.__dataclass_fields__.items():
        default = workers if name == "workers" else field.default
        parser.add_argument("--"+name.replace("_","-"), default=argparse.SUPPRESS if inherit else default,
                            type=str if default is None else type(default))


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "status", "stop", "_background"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, help="Run directory containing status and immutable attempts")
        if name == "start":
            command.add_argument("--plan", help="Frozen plan; optional when resuming the same root")
            _parallel_arguments(command, workers=4, inherit=True)
        elif name == "status":
            command.add_argument("--watch",type=float,default=0)
            command.add_argument("--json",action="store_true")
        elif name == "_background":
            command.add_argument("--lock-fd",type=int,required=True)
    reference = commands.add_parser("reference-plan", help="Prepare only the new library's inherited reference recipe")
    for option in ("b1-batch", "library", "output"):
        reference.add_argument("--" + option, required=True)
    replan = commands.add_parser("replan", help="Plan independent COBYQA runs using archived B1-B3 starts and references")
    replan.add_argument("--from-plan", required=True)
    replan.add_argument("--output", required=True)
    replan.add_argument("--depth", type=int, default=2)
    replan.add_argument("--sizes", nargs="+", type=int, default=[12, 16])
    replan.add_argument("--per-cell", type=int, default=4)
    replan.add_argument("--selection-seed", type=int, default=20261026)
    plan = commands.add_parser("plan", help="Freeze declared graphs and B1–B5 initialization conditions")
    for option in ("b1-batch", "output"):
        plan.add_argument("--" + option, required=True)
    for option in ("b4-fits", "b5-fits"):
        plan.add_argument("--" + option)
    plan.add_argument("--b2-fit", action="append", default=[], help="Repeat for the original random and LOFO fits")
    plan.add_argument("--library", action="append", default=[], help="Additional held-out libraries; original comes from B1 batch")
    plan.add_argument("--references", action="append", required=True)
    plan.add_argument("--depths", nargs="+", type=int, default=[1, 2])
    plan.add_argument("--b3-rules", help="JSON mapping from depth to explicit B3 coefficients and provenance")
    plan.add_argument("--repetitions", type=int, required=True, help="Explicit stochastic repeats per initialization condition")
    plan.add_argument("--graph-ids-file", help="JSON list of exact evaluation graph IDs; source libraries remain intact")
    plan.add_argument("--initializations", nargs="+", choices=("B1", "B2", "B3", "B4", "B5"),
                      default=["B1", "B2", "B3", "B4", "B5"])
    plan.add_argument("--methods", nargs="+", choices=("COBYLA", "COBYQA", "SPSA"), default=["COBYLA", "SPSA"])
    plan.add_argument("--modes", nargs="+", choices=("finite_shots", "exact_control"), default=["finite_shots", "exact_control"])
    plan.add_argument("--backend", choices=("default.qubit", "lightning.qubit", "lightning.gpu"), default="default.qubit")
    plan.add_argument("--seed", type=int, default=20261001)
    for name, field in ShotOptimizerSettings.__dataclass_fields__.items():
        plan.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    for name in ("run", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--output", required=True, help="Immutable attempt directory")
        if name == "run":
            _parallel_arguments(sub)
            sub.add_argument("--execute", action="store_true")
            sub.add_argument("--retry-interrupted", action="store_true",
                             help="After the previous process has stopped, retry interrupted tasks within the frozen attempt cap")
            sub.add_argument("--max-tasks", type=int)
            sub.add_argument("--graph-ids", nargs="+", help="Execute/resume only these graph identities")
        else:
            sub.add_argument("--save", required=True, help="New summary JSON path")
    args = parser.parse_args(argv)
    if args.command in ("start","status","stop","_background"):
        return _lifecycle(args)
    if args.command == "reference-plan":
        manifest = plan_references(args.b1_batch, args.library, args.output)
        result = {k: manifest[k] for k in ("batch_id", "counts", "issues")}
    elif args.command == "replan":
        result = replan_benchmark(args.from_plan, args.output, depth=args.depth, sizes=args.sizes,
                                 per_cell=args.per_cell, selection_seed=args.selection_seed)["budget"]
    elif args.command == "plan":
        settings = ShotOptimizerSettings(**{name: getattr(args, name) for name in ShotOptimizerSettings.__dataclass_fields__})
        result = plan_benchmark(args.b1_batch, args.b2_fit, args.b4_fits, args.b5_fits, args.output,
            libraries=args.library, references=args.references, depths=args.depths, repetitions=args.repetitions,
            settings=settings, methods=args.methods, modes=args.modes, backend=args.backend, seed=args.seed,
            graph_ids=read_json(args.graph_ids_file) if args.graph_ids_file else None,
            initializations=args.initializations, b3_rules=read_json(args.b3_rules) if args.b3_rules else None)["budget"]
    elif args.command == "run":
        result = run_benchmark(args.plan, args.output, execute=args.execute, max_tasks=args.max_tasks,
                               graph_ids=args.graph_ids, retry_interrupted=args.retry_interrupted,
                               parallel=_parallel_settings(args))
    else:
        summary = summarize_benchmark(args.plan, args.output)
        write_once_json(args.save, summary)
        result = {k: summary[k] for k in ("complete", "assessments_complete", "planned", "completed", "execution_failures")}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
