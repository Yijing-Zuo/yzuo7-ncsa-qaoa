"""Plan, run and summarize the finite-shot B1–B5 comparison using immutable records.

This entry point reuses the library, B2-fit and record contracts.
It never fits angles or schedules workers. Reference execution uses the existing worker.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

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
                   seed=20261001):
    """Freeze all evaluation graphs and every existing B1–B5 angle condition."""
    baseline, original, baseline_tasks = read_batch(b1_batch)
    loaded = [original, *[read_library(path) for path in libraries]]
    depths, methods, modes = list(depths), list(methods), list(modes)
    if (not depths or len(set(depths)) != len(depths) or not set(depths) <= {1, 2}
            or not methods or len(set(methods)) != len(methods) or not set(methods) <= {"COBYLA", "SPSA"}
            or not modes or len(set(modes)) != len(modes) or not set(modes) <= {"finite_shots", "exact_control"}):
        raise ValueError("Use distinct supported depths, optimizers and evaluation modes.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    if backend not in ("default.qubit", "lightning.qubit"):
        raise ValueError("Finite shots currently supports the validated CPU backends.")
    settings = settings or ShotOptimizerSettings()
    required = max(2 * p + 2 if method == "COBYLA" else 2 for p in depths for method in methods)
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
            if graph["iso_class_id"] in source_library:
                raise ValueError("Duplicate graph identity.")
            if graph.get("exact_status") != "computed" or not graph.get("C_star"):
                raise ValueError("Exact MaxCut annotations are required for complete comparison.")
            source_library[graph["iso_class_id"]] = identity
            graphs.append(graph)
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
    prepared = prepare_shot_starts(loaded, b2_fits, b4_fits, b5_fits, depths=depths)
    starts = prepared["starts"]
    pools = defaultdict(dict)
    for task in baseline_tasks:
        if task["experiment_role"] == "evaluation" and task["iso_class_id"] in source_library:
            pools[task["iso_class_id"], task["p"]][task["restart_id"]] = {
                "theta0": task["theta0"], "seed": task["seed"], "source_task_id": task["task_id"]}
    for graph in graphs:
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
    coverage = defaultdict(set)
    for start in starts:
        coverage[start["iso_class_id"], start["p"]].add(start["initialization"])
    if any(coverage[g["iso_class_id"], p] != {"B1", "B2", "B3", "B4", "B5"} for g in graphs for p in depths):
        raise ValueError("Every evaluation graph/depth must include B1 through B5.")
    settings_dict = asdict(settings)
    config = {"depths": depths, "optimizers": methods, "evaluation_modes": modes,
              "repetitions": repetitions, "backend": backend, "optimizer": settings_dict,
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


def run_benchmark(plan_path, output, *, execute=False, max_tasks=None, graph_ids=None,
                  retry_interrupted=False):
    """Skip complete records; explicitly retry interrupted work within its attempt cap."""
    plan, output = _read_plan(plan_path), Path(output)
    if source_identity() != plan["source"]:
        raise ValueError("Working source/configuration changed since planning; create a new plan.")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer.")
    config = plan["config"]
    env = environment(config["backend"])
    env.update(shots="per-task fixed batches", interface=None, diff_method=None, device_vjp=False)
    execution = {"plan_id": plan["plan_id"], "source_id": plan["source"]["source_id"], "environment": env}
    execution["execution_id"] = digest(execution)
    execution_path = output / "execution.json"
    if execution_path.exists():
        if read_json(execution_path) != execution:
            raise ValueError("Output belongs to another plan, source or compute environment.")
    elif execute:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing an unrecognized nonempty output directory.")
        write_once_json(execution_path, execution)
    graphs = {g["iso_class_id"]: g for g in plan["graphs"]}
    report = {"planned": plan["budget"]["tasks"], "selected": 0, "completed": 0,
              "executed": 0, "interrupted": 0, "retried_interrupted": 0, "execution_failures": 0}
    for task in iter_tasks(plan, graph_ids):
        report["selected"] += 1
        completed = _completed(output, task, execution)
        if completed is not None:
            report["completed"] += 1
            report["execution_failures"] += int(bool(completed.get("error") or completed.get("validation_error")
                or completed.get("exact_audit_error") or completed["stop_reason"] in EXECUTION_FAILURES))
            continue
        marker = run_path(output, task).parent / "started" / (task["task_id"] + ".json")
        attempt_number = 1
        if marker.exists():
            if not retry_interrupted or not execute or (max_tasks is not None and report["executed"] >= max_tasks):
                report["interrupted"] += 1
                continue
            while marker.exists() and attempt_number < config["max_attempts"]:
                attempt_number += 1
                marker = marker.with_name(task["task_id"] + f".attempt-{attempt_number}.json")
            if marker.exists():
                report["interrupted"] += 1
                continue
        if not execute or (max_tasks is not None and report["executed"] >= max_tasks):
            continue
        write_once_json(marker, {"task_id": task["task_id"], "execution_id": execution["execution_id"],
                                 "attempt_number": attempt_number,
                                 "started_utc": datetime.now(timezone.utc).isoformat()})
        result = optimize_qaoa_shots(graph_from_record(graphs[task["iso_class_id"]]), task["p"],
            task["theta0"], ShotOptimizerSettings(**config["optimizer"]), method=task["method"],
            evaluation_mode=task["evaluation_mode"], measurement_seed=task["seeds"]["measurement"],
            perturbation_seed=task["seeds"]["perturbation"], validation_seed=task["seeds"]["validation"],
            backend=config["backend"], audit_exact=True)
        result.update(task=task, execution_id=execution["execution_id"], run_completed=True,
                      attempt_number=attempt_number, previous_interrupted_attempts=attempt_number - 1)
        save_run(run_path(output, task), result)
        report["executed"] += 1
        report["completed"] += 1
        report["retried_interrupted"] += int(attempt_number > 1)
        report["execution_failures"] += int(bool(result.get("error") or result.get("validation_error")
            or result.get("exact_audit_error") or result["stop_reason"] in EXECUTION_FAILURES))
    report["complete"] = report["completed"] == report["planned"]
    report["selected_complete"] = report["completed"] == report["selected"]
    report["missing"] = report["selected"] - report["completed"] - report["interrupted"]
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
            "weighting": "Repetitions within graph, then equal graph weights; any missing/faulted exact endpoint leaves its group metric null.",
            "scope": "Original evaluation and external-family transfer are separate; LOFO baselines reuse matched graph outcomes. Counts cover completed records; work before an interrupted record is unknown."}


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference-plan", help="Prepare only the new library's inherited reference recipe")
    for option in ("b1-batch", "library", "output"):
        reference.add_argument("--" + option, required=True)
    plan = commands.add_parser("plan", help="Freeze all B1–B5 comparisons, without a pilot subset")
    for option in ("b1-batch", "b4-fits", "b5-fits", "output"):
        plan.add_argument("--" + option, required=True)
    plan.add_argument("--b2-fit", action="append", required=True, help="Repeat for the original random and LOFO fits")
    plan.add_argument("--library", action="append", default=[], help="Additional held-out libraries; original comes from B1 batch")
    plan.add_argument("--references", action="append", required=True)
    plan.add_argument("--depths", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    plan.add_argument("--repetitions", type=int, required=True, help="Explicit stochastic repeats per initialization condition")
    plan.add_argument("--methods", nargs="+", choices=("COBYLA", "SPSA"), default=["COBYLA", "SPSA"])
    plan.add_argument("--modes", nargs="+", choices=("finite_shots", "exact_control"), default=["finite_shots", "exact_control"])
    plan.add_argument("--backend", choices=("default.qubit", "lightning.qubit"), default="default.qubit")
    plan.add_argument("--seed", type=int, default=20261001)
    for name, field in ShotOptimizerSettings.__dataclass_fields__.items():
        plan.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    for name in ("run", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--output", required=True, help="Immutable attempt directory")
        if name == "run":
            sub.add_argument("--execute", action="store_true")
            sub.add_argument("--retry-interrupted", action="store_true",
                             help="After the previous process has stopped, retry interrupted tasks within the frozen attempt cap")
            sub.add_argument("--max-tasks", type=int)
            sub.add_argument("--graph-ids", nargs="+", help="Execute/resume only these graph identities")
        else:
            sub.add_argument("--save", required=True, help="New summary JSON path")
    args = parser.parse_args(argv)
    if args.command == "reference-plan":
        manifest = plan_references(args.b1_batch, args.library, args.output)
        result = {k: manifest[k] for k in ("batch_id", "counts", "issues")}
    elif args.command == "plan":
        settings = ShotOptimizerSettings(**{name: getattr(args, name) for name in ShotOptimizerSettings.__dataclass_fields__})
        result = plan_benchmark(args.b1_batch, args.b2_fit, args.b4_fits, args.b5_fits, args.output,
            libraries=args.library, references=args.references, depths=args.depths, repetitions=args.repetitions,
            settings=settings, methods=args.methods, modes=args.modes, backend=args.backend, seed=args.seed)["budget"]
    elif args.command == "run":
        result = run_benchmark(args.plan, args.output, execute=args.execute, max_tasks=args.max_tasks,
                               graph_ids=args.graph_ids, retry_interrupted=args.retry_interrupted)
    else:
        summary = summarize_benchmark(args.plan, args.output)
        write_once_json(args.save, summary)
        result = {k: summary[k] for k in ("complete", "assessments_complete", "planned", "completed", "execution_failures")}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
