"""Plan, explicitly execute, freeze and summarize the same Plan A task list."""

import argparse
from contextlib import nullcontext
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys
import zipfile

from qaoa_study.analysis import build_summary, build_warm_summary, build_b3_summary
from qaoa_study.experiments import (
    _record_directory, digest, freeze_references, iter_attempt_groups, load_attempts, read_batch, read_references,
    run_worker, save_batch, save_references, select_attempts, source_identity,
    attempt_cost_totals, read_b2_inputs, save_b2_fit, save_b2_batch, validate_b1_comparison,
    read_b3_inputs, save_b3_initializations, save_b3_batch,
    read_b4_inputs, save_b4_fits, save_b4_predictions, save_b4_batch,
    read_b5_inputs, read_b5_explanations, save_b5_fits, save_b5_predictions, save_b5_batch,
)
from qaoa_study.records import (evaluate_trace, execution_failed, read_json, write_once_json,
                                LEGACY_SUCCESS_METRIC, resolve_success_metric)


def save_summary(output, summary):
    """Save portable numeric columns plus lossless JSON for nested diagnostics."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_once_json(output / "summary.json", summary)
    for name in ("graph_qaoa", "restarts", "diagnostics"):
        rows = []
        for item in summary[name]:
            row = {}
            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    row[key + "_json"] = json.dumps(value, sort_keys=True, allow_nan=False)
                else:
                    # Task seeds are 128-bit identities; preserve decimal digits,
                    # never truncate them into Arrow's signed 64-bit integers.
                    row[key] = str(value) if key == "seed" and value is not None else value
            rows.append(row)
        keys = sorted({key for row in rows for key in row})
        table = pa.Table.from_pylist([{key: row.get(key) for key in keys} for row in rows])
        pq.write_table(table, output / f"{name}.parquet", compression="zstd")
    names = ["summary.json", "graph_qaoa.parquet", "restarts.parquet", "diagnostics.parquet"]
    write_once_json(output / "manifest.json", {"summary_id": digest(summary),
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names},
        "parquet_encoding": "Scalar columns native; seed decimal string; nested list/dict columns suffixed_json."})


def _summary_inputs(batch, references_directory):
    manifest, library, tasks = read_batch(batch)
    references = read_references(references_directory)
    if any(ref["batch_id"] != manifest["batch_id"] or ref["protocol_version"] != manifest["config"]["protocol_version"]
           for ref in references.values()):
        raise ValueError("Reference set does not belong to this batch/protocol.")
    return manifest, library, tasks, references


def _merge_audit(target, update):
    for key, values in update.items():
        if isinstance(values, dict):
            target.setdefault(key, {}).update(values)
        elif isinstance(values, list):
            target.setdefault(key, []).extend(values)


def _iter_graph_results(manifest, tasks, attempts_directory):
    """Release validated traces and I/O evidence after each graph is consumed."""
    attempts_directory = _record_directory(attempts_directory)
    if not attempts_directory.exists():
        for identity, group in groupby(sorted(tasks, key=lambda row: row["iso_class_id"]),
                                      key=lambda row: row["iso_class_id"]):
            yield identity, list(group), [], {}, None
        return
    audit, graph_audit, graph_tasks, attempts = {}, {}, [], []
    identity, execution = None, None
    for group, rows, current_execution in iter_attempt_groups(manifest, tasks, attempts_directory, audit=audit):
        current_audit = dict(audit)
        audit.clear()
        current_identity = group[0]["iso_class_id"]
        if identity is not None and current_identity != identity:
            graph_tasks.sort(key=lambda row: row["task_id"])
            attempts.sort(key=lambda row: (row["task_id"], row["attempt_id"]))
            yield identity, graph_tasks, attempts, graph_audit, execution
            graph_audit, graph_tasks, attempts = {}, [], []
        identity, execution = current_identity, current_execution
        graph_tasks.extend(group)
        attempts.extend(rows)
        _merge_audit(graph_audit, current_audit)
    if identity is not None:
        graph_tasks.sort(key=lambda row: row["task_id"])
        attempts.sort(key=lambda row: (row["task_id"], row["attempt_id"]))
        yield identity, graph_tasks, attempts, graph_audit, execution


def _scoring_settings(manifest, override=None):
    settings = dict(manifest["config"].get("analysis", {}))
    if not any(k in settings for k in ("epsilon", "success_metric")):
        settings["success_metric"] = resolve_success_metric(manifest["config"])
    if override is not None:
        if any(key in override for key in ("success_metric", "epsilon")):
            settings.pop("epsilon", None)
            settings.pop("success_metric", None)
        settings.update(override)
    return settings


def _empty_summary(library, manifest, analysis_settings=None):
    summary = build_summary(library, [], {}, {}, _scoring_settings(manifest, analysis_settings))
    summary.update(all_attempt_costs_by_role={}, selected_attempts={})
    return summary


def _append_graph_summary(summary, library, graph, manifest, tasks, attempts, references, audit, completed_ids, analysis_settings=None):
    selected = select_attempts(attempts)
    part = build_summary({**library, "graphs": [graph]}, tasks, selected, references,
                         _scoring_settings(manifest, analysis_settings))
    for key in ("graph_qaoa", "restarts", "diagnostics"):
        summary[key].extend(part[key])
    summary["selected_attempts"].update({key: row["attempt_id"] for key, row in selected.items()})
    for row in attempts:
        if row["attempt_id"] in completed_ids:
            raise ValueError("Duplicate attempt ID across graphs.")
        completed_ids.add(row["attempt_id"])
        role = row["experiment_role"]
        costs = summary["all_attempt_costs_by_role"].setdefault(role, {
            "attempts": 0, "selected_attempts": 0,
            "counts": {name: 0 for name in ("objective_calls", "gradient_requests", "device_executions",
                                            "device_derivatives", "device_vjps", "iterations")},
            "analytic_evaluations": 0, "analytic_points_requested": 0,
            "compute_elapsed_seconds": 0.0, "preparation_seconds": 0.0,
            "record_save_seconds_measured": 0.0, "record_save_measurements_missing": 0, "attempt_ids": [],
            "known_subtotals": {}, "unknown_attempt_ids_by_metric": {}, "cost_provenance": []})
        costs["attempts"] += 1
        costs["selected_attempts"] += int(row["attempt_id"] == selected[row["task_id"]]["attempt_id"])
        costs["attempt_ids"].append(row["attempt_id"])
        measurements = {name: (row.get("counts") or {}).get(name) for name in costs["counts"]}
        measurements.update({name: (row.get("grid") or {}).get(name, None if row["kind"] == "p1_grid" else 0)
                             for name in ("analytic_evaluations", "analytic_points_requested")})
        measurements.update(compute_elapsed_seconds=row.get("elapsed_seconds"),
                            preparation_seconds=row.get("preparation_seconds"))
        unknown = []
        for name, value in measurements.items():
            unavailable = row.get("cost_unavailable") or {}
            original_name = "elapsed_seconds" if name == "compute_elapsed_seconds" else name
            unverified = row.get("cost_status") == "unverified_invalid_result" and name in costs["counts"]
            target = costs["counts"] if name in costs["counts"] else costs
            costs["known_subtotals"].setdefault(name, 0)
            if value is None or not math.isfinite(value) or original_name in unavailable or unverified:
                target[name] = None
                costs["unknown_attempt_ids_by_metric"].setdefault(name, []).append(row["attempt_id"])
                unknown.append(name)
            else:
                costs["known_subtotals"][name] += value
                if target[name] is not None:
                    target[name] += value
        if unknown:
            costs["cost_provenance"].append({"task_id": row["task_id"], "attempt_id": row["attempt_id"],
                                              "unknown_metrics": unknown, "cost_status": row.get("cost_status"),
                                              "cost_unavailable": row.get("cost_unavailable")})
        measured = audit.get("save_seconds", {}).get(row["attempt_id"])
        if measured is not None:
            costs["record_save_seconds_measured"] += measured
        else:
            costs["record_save_measurements_missing"] += 1
    required_references = {(task["iso_class_id"], task["p"]) for task in tasks
                           if task["experiment_role"] == "reference"}
    return {"planned_tasks": len(tasks), "valid_tasks": sum(not execution_failed(row) for row in selected.values()),
            "planned_references": len(required_references),
            "complete_references": sum(bool((ref := references.get(key)) and ref.get("complete")
                                               and ref.get("C_ref") is not None and math.isfinite(ref["C_ref"]))
                                       for key in required_references),
            "scientific_labels": sum(row["evaluation"]["scientific_label_complete"] for row in part["graph_qaoa"])}


def _finish_summary(summary, manifest, execution, audit, completed_ids, source):
    input_digest = hashlib.sha256()
    hashes = audit.get("file_hashes", {})
    for name, checksum in sorted(hashes.items()):
        input_digest.update((name + "\0" + checksum + "\n").encode())
    summary.update(batch_id=manifest["batch_id"], execution_id=execution["execution_id"] if execution else None,
                   analysis_source=source, coverage=manifest["coverage"],
                   attempt_selection_rule=manifest["attempt_selection_rule"],
                   interrupted_attempt_ids=sorted(identity for identity in audit.get("started_attempt_ids", [])
                                                  if identity not in completed_ids),
                   interrupted_cost="Unknown incomplete work is not recorded as zero or added to hit samples.",
                   attempt_input_files=len(hashes), attempt_input_digest=input_digest.hexdigest(),
                   input_digest_rule="SHA256 of sorted basename + NUL + SHA256(file) + newline" if
                       not any("/" in name for name in hashes) else
                       "SHA256 of sorted logical relative path + NUL + SHA256(file) + newline")


def summarize_batch(batch, attempts_directory, references_directory, *, analysis_settings=None):
    """Reconstruct one graph at a time; return the legacy combined tables."""
    manifest, library, tasks, references = _summary_inputs(batch, references_directory)
    graphs = {graph["iso_class_id"]: graph for graph in library["graphs"]}
    summary = _empty_summary(library, manifest, analysis_settings)
    audit, completed_ids, execution = {}, set(), None
    for identity, graph_tasks, attempts, graph_audit, execution in _iter_graph_results(manifest, tasks, attempts_directory):
        _append_graph_summary(summary, library, graphs[identity], manifest, graph_tasks, attempts,
                              references, graph_audit, completed_ids, analysis_settings)
        _merge_audit(audit, graph_audit)
    _finish_summary(summary, manifest, execution, audit, completed_ids, source_identity())
    return summary


def save_partitioned_summary(output, batch, attempts_directory, references_directory, *, analysis_settings=None):
    """Export graph-sized JSON/Parquet tables; publish the index only after all writes succeed."""
    manifest, library, tasks, references = _summary_inputs(batch, references_directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    graphs = {graph["iso_class_id"]: graph for graph in library["graphs"]}
    totals = {key: 0 for key in ("planned_tasks", "valid_tasks", "planned_references", "complete_references", "scientific_labels")}
    row_counts = {key: 0 for key in ("graph_qaoa", "restarts", "diagnostics")}
    entries, completed_ids, execution = [], set(), None
    for identity, graph_tasks, attempts, audit, execution in _iter_graph_results(manifest, tasks, attempts_directory):
        summary = _empty_summary(library, manifest, analysis_settings)
        counts = _append_graph_summary(summary, library, graphs[identity], manifest, graph_tasks, attempts,
                                       references, audit, completed_ids, analysis_settings)
        _finish_summary(summary, manifest, execution, audit, completed_ids, source)
        name = "graphs/" + digest(identity)
        save_summary(output / name, summary)
        checksum = hashlib.sha256((output / name / "manifest.json").read_bytes()).hexdigest()
        entries.append({"iso_class_id": identity, "path": name, "manifest_sha256": checksum,
                        "attempt_input_digest": summary["attempt_input_digest"], **counts})
        for key in totals:
            totals[key] += counts[key]
        for key in row_counts:
            row_counts[key] += len(summary[key])
    complete = (len(entries) == len(graphs) and totals["planned_tasks"] == len(tasks)
                and totals["valid_tasks"] == len(tasks)
                and totals["complete_references"] == totals["planned_references"] and not manifest["issues"])
    index = {"format": "partitioned-summary-v1", "batch_id": manifest["batch_id"], "library_id": library["library_id"],
             "execution_id": execution["execution_id"] if execution else None, "analysis_source": source,
             "analysis_version": summary["analysis_version"] if entries else None,
             "settings": summary["settings"] if entries else _scoring_settings(manifest, analysis_settings), "coverage": manifest["coverage"],
             "attempt_selection_rule": manifest["attempt_selection_rule"], "complete": complete,
             "completion_scope": "All configured tasks execution-valid and all required frozen references complete; scientific label counts reported separately.",
             "counts": totals, "table_rows": row_counts, "graphs": entries}
    write_once_json(output / "manifest.json", index)
    return index


def _warm_attempt_views(manifest, tasks, directory, references, requested, *, reference_pairs=None, by_depth=False,
                        graphs=None, metric=None):
    """Project validated graph-sized traces onto events needed by warm analysis.

    These in-memory views retain the first call, first threshold hit and maximum
    score event. They are not run records and are never written as attempts.
    Full calls/counts and provenance were validated before projection.
    """
    metric = resolve_success_metric(metric if metric is not None else manifest["config"])
    selected, costs, reference_costs, audit, execution, completed_ids = {}, [], [], {}, None, set()
    verified_references = set()
    lookup = {t["task_id"]: t for t in requested}
    graph_ids = sorted({t["iso_class_id"] for t in requested})
    for group, rows, execution in iter_attempt_groups(manifest, tasks, directory, graph_ids=graph_ids, audit=audit):
        completed_ids.update(row["attempt_id"] for row in rows)
        if reference_pairs:
            reference_tasks = [task for task in group if task["experiment_role"] == "reference"
                               and (task["iso_class_id"], task["p"]) in reference_pairs]
            actual = freeze_references(manifest, reference_tasks, select_attempts(rows))
            for key, ref in actual.items():
                if ref != references[key]:
                    raise ValueError("Evaluation reference no longer matches its original attempts.")
                verified_references.add(key)
        for row in rows:
            reference = row["experiment_role"] == "reference" and (reference_pairs is None or
                (row["iso_class_id"], row["p"]) in reference_pairs)
            target = costs if row["task_id"] in lookup else reference_costs if reference else None
            if target is not None:
                target.append({k: v for k, v in row.items() if k in
                               ("task_id", "attempt_id", "iso_class_id", "p", "kind", "stop_reason", "counts", "elapsed_seconds",
                                "preparation_seconds", "cost_status", "cost_unavailable", "grid")})
        rows = [row for row in rows if row["task_id"] in lookup]
        for identity, row in select_attempts(rows).items():
            ref = references[row["iso_class_id"], row["p"]]
            graph = (graphs or {}).get(row["iso_class_id"], {})
            exact = graph.get("C_star") if graph.get("exact_status") == "computed" else None
            metrics = [metric]
            if exact is not None and exact > 0:
                metrics += [{"kind": "normalized_gap", "tolerance": tolerance} for tolerance in (.005, .01, .02)]
            calls = {1, row.get("best_call_id")}
            calls.update(evaluate_trace(row, ref["C_ref"], C_star=exact, metric=rule)["first_hit"] for rule in metrics)
            view = {k: v for k, v in row.items() if k not in ("trace", "optimizer_messages", "invalid_result")}
            view["trace"] = [{k: point[k] for k in ("call_id", "C", "theta")}
                             for point in row["trace"] if point["call_id"] in calls]
            selected[identity] = view
    if reference_pairs and verified_references != reference_pairs:
        raise ValueError("Evaluation reference source attempts are missing.")
    interrupted = sorted(set(audit.get("started_attempt_ids", [])) - completed_ids)
    costs_audit = {"input_digest": digest(audit.get("file_hashes", {})),
        "input_files": len(audit.get("file_hashes", {})), "interrupted_attempt_ids_in_read_scopes": interrupted,
        "all_requested_attempt_costs": attempt_cost_totals(costs, save_seconds=audit.get("save_seconds")),
        "reference_attempt_costs": attempt_cost_totals(reference_costs, save_seconds=audit.get("save_seconds"))}
    if by_depth:
        for name, values in (("all_requested_attempt_costs", costs), ("reference_attempt_costs", reference_costs)):
            costs_audit[name + "_by_depth"] = {
                str(p): {**attempt_cost_totals([row for row in values if row["p"] == p],
                    save_seconds=audit.get("save_seconds")),
                    "attempt_ids": sorted(row["attempt_id"] for row in values if row["p"] == p)}
                for p in sorted({task["p"] for task in requested})}
    return selected, execution, costs_audit


def summarize_b2_batch(batch, attempts_directory, b1_batch, b1_attempts, *, analysis_settings=None):
    """Read a bound B2 experiment and separately verify B1 comparison eligibility."""
    manifest, library, tasks = read_batch(batch)
    scoring = _scoring_settings(manifest, analysis_settings)
    metric = resolve_success_metric(scoring) if any(k in scoring for k in ("epsilon", "success_metric")) else resolve_success_metric()
    graphs = {g["iso_class_id"]: g for g in library["graphs"]}
    if manifest["config"].get("kind") != "b2":
        raise ValueError("summarize-b2 requires a B2 batch.")
    fit, binding = read_b2_inputs(batch, manifest)
    baseline, _, baseline_tasks = read_batch(b1_batch)
    if baseline["batch_id"] != binding["source_batch_id"]:
        raise ValueError("B1 comparison is not the explicitly bound source batch.")
    references = {(r["iso_class_id"], r["p"]): r for r in binding["references"]}
    targets = {(t["iso_class_id"], t["p"]) for t in tasks}
    b1_tasks = [t for t in baseline_tasks if t["experiment_role"] == "evaluation"
                and (t["iso_class_id"], t["p"]) in targets]
    selected, execution, warm_audit = _warm_attempt_views(manifest, tasks, attempts_directory, references, tasks, graphs=graphs, metric=metric)
    b1_selected, b1_execution, b1_audit = _warm_attempt_views(baseline, baseline_tasks, b1_attempts, references, b1_tasks, graphs=graphs, metric=metric)
    validate_b1_comparison(manifest, baseline, execution, b1_execution)
    summary = build_warm_summary(library, tasks, selected, references, b1_tasks=b1_tasks,
                                 b1_selected=b1_selected, settings=scoring)
    evaluation_reference_costs = b1_audit.pop("reference_attempt_costs")
    summary.update(batch_id=manifest["batch_id"], b1_batch_id=baseline["batch_id"], fit_id=fit["fit_id"],
        reference_binding_digest=digest(binding), analysis_source=source_identity(),
        execution_id=execution["execution_id"] if execution else None,
        b1_execution_id=b1_execution["execution_id"] if b1_execution else None,
        input_audits={"B1": b1_audit, "B2": warm_audit}, fit_seconds=fit["fit_seconds"],
        training_reference_costs=fit["reference_costs"],
        evaluation_reference_costs=evaluation_reference_costs,
        cost_scope="Original B1/B2 retained attempts charged once; both B2 variants are separate runs. "
                   "Training references at fit time and evaluation references at summary time are separate; "
                   "each reference scope is shared by both variants. Interrupted unrecorded work is unknown.",
        uncertainty_scope="Graph-paired intervals conditional on this frozen training fit; no training-resampling uncertainty.")
    return summary


def _b3_training_costs(manifest, tasks, directory, fits, depths):
    """Reconstruct used training references; charge shared attempts only once."""
    expected = {}
    for fit in fits:
        if fit["reference_binding"]["source_batch_id"] != manifest["batch_id"]:
            raise ValueError("B2 training fit uses a different original reference batch.")
        for ref in fit["reference_binding"]["references"]:
            if ref["p"] in depths:
                key = ref["iso_class_id"], ref["p"]
                if expected.setdefault(key, ref) != ref:
                    raise ValueError("B2 fits disagree on a shared training reference.")
    audit, retained, found, completed = {}, [], set(), set()
    graph_ids = sorted({identity for identity, _ in expected})
    for group, rows, _ in iter_attempt_groups(manifest, tasks, directory,
            graph_ids=graph_ids, metadata_only=True, audit=audit):
        completed.update(row["attempt_id"] for row in rows)
        reference_tasks = [task for task in group if task["experiment_role"] == "reference"
                           and (task["iso_class_id"], task["p"]) in expected]
        if not reference_tasks:
            continue
        reference_ids = {task["task_id"] for task in reference_tasks}
        reference_rows = [row for row in rows if row["task_id"] in reference_ids]
        actual = freeze_references(manifest, reference_tasks, select_attempts(reference_rows))
        for key in expected.keys() & actual.keys():
            if actual[key] != expected[key]:
                raise ValueError("B2 training reference no longer matches its original attempts.")
            found.add(key)
        retained.extend(reference_rows)
    if found != set(expected):
        raise ValueError("B2 training reference source attempts are missing.")
    def costs(rows):
        return {**attempt_cost_totals(rows, save_seconds=audit.get("save_seconds")),
                "attempt_ids": sorted(row["attempt_id"] for row in rows)}
    return {"union_by_depth": {str(p): costs([r for r in retained if r["p"] == p]) for p in depths},
        "union_total": costs(retained),
        "fits": [{"fit_id": fit["fit_id"], "regime": fit["regime"], "fold": fit["fold"],
                  "fit_seconds_joint": fit["fit_seconds"], "fit_depths": sorted(map(int, fit["fits"])),
                  "training_reference_costs_by_depth": {str(p): costs([r for r in retained
                      if r["p"] == p and r["iso_class_id"] in fit["training_graph_ids"]]) for p in depths}}
                 for fit in fits],
        "input_digest": digest(audit.get("file_hashes", {})),
        "interrupted_attempt_ids_in_read_scopes": sorted(set(audit.get("started_attempt_ids", [])) - completed),
        "scope": "Original training-reference attempts; union deduplicated across fits. Joint fit time is not apportioned by depth."}


def summarize_b3_batch(batch, attempts_directory, b1_batch, b1_attempts, b2_cases, *, analysis_settings=None):
    """Compare B3 with the declared B1/B2 cases under matching frozen contracts."""
    manifest, library, tasks = read_batch(batch)
    scoring = _scoring_settings(manifest, analysis_settings)
    metric = resolve_success_metric(scoring) if any(k in scoring for k in ("epsilon", "success_metric")) else resolve_success_metric()
    graphs = {g["iso_class_id"]: g for g in library["graphs"]}
    if manifest["config"].get("kind") != "b3":
        raise ValueError("summarize-b3 requires a B3 batch.")
    prepared, binding = read_b3_inputs(batch, manifest)
    baseline, _, baseline_tasks = read_batch(b1_batch)
    if baseline["batch_id"] != binding["source_batch_id"]:
        raise ValueError("B3 comparison needs its explicitly bound original B1 batch.")
    references = {(ref["iso_class_id"], ref["p"]): ref for ref in binding["references"]}
    targets = {(task["iso_class_id"], task["p"]) for task in tasks}
    b1_tasks = [task for task in baseline_tasks if task["experiment_role"] == "evaluation"
                and (task["iso_class_id"], task["p"]) in targets]
    selected, execution, b3_audit = _warm_attempt_views(manifest, tasks, attempts_directory,
        references, tasks, reference_pairs=set(), by_depth=True, graphs=graphs, metric=metric)
    b1_selected, b1_execution, b1_audit = _warm_attempt_views(baseline, baseline_tasks, b1_attempts,
        references, b1_tasks, reference_pairs=targets, by_depth=True, graphs=graphs, metric=metric)
    comparison_execution = execution or {"environment": prepared["environment"]}
    validate_b1_comparison(manifest, baseline, comparison_execution, b1_execution)
    b2_tasks, b2_selected, fits, cases, audits = [], {}, [], [], {}
    expected_cases = {(case["regime"], case["fold"]) for case in manifest["config"]["comparison_cases"]}
    observed = set()
    for case_batch, case_attempts in b2_cases:
        other, _, planned = read_batch(case_batch)
        if other["config"].get("kind") != "b2":
            raise ValueError("B3 comparison cases must be frozen B2 batches.")
        fit, other_binding = read_b2_inputs(case_batch, other)
        key = fit["regime"], fit["fold"]
        if key in observed or key not in expected_cases:
            raise ValueError("Duplicate or undeclared B2 comparison case.")
        observed.add(key)
        if other_binding["source_batch_id"] != baseline["batch_id"]:
            raise ValueError("B2 comparison does not use the same original B1 batch.")
        other_refs = {(ref["iso_class_id"], ref["p"]): ref for ref in other_binding["references"]}
        requested = [task for task in planned if task["p"] in manifest["config"]["depths"]]
        if any((task["iso_class_id"], task["p"]) not in targets or
               other_refs.get((task["iso_class_id"], task["p"])) != references[task["iso_class_id"], task["p"]]
               for task in requested):
            raise ValueError("B2/B3 evaluation graphs or scoring reference identities disagree.")
        chosen, other_execution, audit = _warm_attempt_views(other, planned, case_attempts,
            other_refs, requested, reference_pairs=set(), by_depth=True, graphs=graphs, metric=metric)
        validate_b1_comparison(manifest, other, comparison_execution, other_execution)
        if set(b2_selected) & set(chosen):
            raise ValueError("B2 cases share an execution task identity.")
        b2_tasks.extend(requested)
        b2_selected.update(chosen)
        fits.append(fit)
        cases.append({"regime": key[0], "fold": key[1], "batch_id": other["batch_id"],
                      "execution_id": other_execution["execution_id"] if other_execution else None,
                      "fit_id": fit["fit_id"]})
        audits[other["batch_id"]] = audit
    if observed != expected_cases:
        raise ValueError("All declared B2 comparison cases are required; no silent fold/variant omission.")
    summary = build_b3_summary(library, tasks, selected, references, b1_tasks=b1_tasks,
        b1_selected=b1_selected, b2_tasks=b2_tasks, b2_selected=b2_selected,
        settings=scoring)
    timings = prepared["preparation"]
    depths = manifest["config"]["depths"]
    summary.update(batch_id=manifest["batch_id"], initialization_set_id=prepared["initialization_set_id"],
        initialization_artifact_id=prepared["artifact_id"], reference_binding_digest=digest(binding),
        b1_batch_id=baseline["batch_id"], b1_execution_id=b1_execution["execution_id"] if b1_execution else None,
        execution_id=execution["execution_id"] if execution else None, b2_cases=cases,
        comparison_inputs={"b1_batch": str(b1_batch), "b1_attempts": str(b1_attempts),
                           "b2_cases": [[str(b), str(a)] for b, a in b2_cases]},
        analysis_source=source_identity(), batch_issues=manifest["issues"],
        input_audits={"B1": b1_audit, "B3": b3_audit, "B2": audits},
        preparation_costs={**timings, "initializer_seconds_by_depth": {
            str(p): math.fsum(row["seconds"] for row in timings["initializers"] if row["p"] == p) for p in depths},
            "counters_by_depth": {str(p): {name: sum(row["counters"][name] for row in prepared["initializations"]
                if row["p"] == p) for name in prepared["initializations"][0]["counters"]} for p in depths}},
        evaluation_reference_costs_by_depth=b1_audit["reference_attempt_costs_by_depth"],
        b2_offline_costs=_b3_training_costs(baseline, baseline_tasks, b1_attempts, fits, depths),
        cost_scope="Physical attempts charged once per task/depth; LOFO/comparison views reuse B3. "
                   "Scoring references are shared evaluation overhead, not initialization. "
                   "External literature-constant optimization and interrupted work are unmeasured, not zero.",
        uncertainty_scope="Graph-paired intervals conditional on frozen fits and depth-specific rules; "
                          "views are not independent replicates and cross-depth differences are not pure depth effects.")
    summary["complete"] = summary["complete"] and not manifest["issues"]
    return summary


def _success_prediction_overlay(path, batch, manifest, prepared, metric):
    """Validate replacement success predictions without rebinding execution artifacts.

    The returned in-memory view is for analysis only. Its original preparation
    and prediction identities remain provenance pointers, not new plan inputs.
    """
    from qaoa_study.learning import validate_b4_model, validate_b5_model, load_b5_boosters
    from qaoa_study.experiments import _b5_model_summary, _validate_b5_explanations

    overlay = read_json(path)
    tree = manifest["config"].get("kind") == "b5"
    if (overlay.get("version") != "success-prediction-overlay-v1"
            or overlay.get("overlay_id") != digest({k: v for k, v in overlay.items() if k != "overlay_id"})
            or overlay.get("source_batch_id") != manifest["batch_id"]
            or overlay.get("source_preparation_id") != prepared["preparation_id"]
            or overlay.get("source_manifest_sha256") != hashlib.sha256((Path(batch) / "manifest.json").read_bytes()).hexdigest()
            or resolve_success_metric(overlay.get("success_metric", LEGACY_SUCCESS_METRIC)) != metric):
        raise ValueError("Success overlay checksum, source batch/preparation or metric differs.")
    original = {m["fit_id"]: m for m in prepared["models"] if m["head"] == "success"}
    replacements, models, native_files = {}, {}, {}
    base = Path(path).resolve().parent
    source = zipfile.ZipFile(base / overlay["source_archive_path"]) if tree else nullcontext()
    with source as archive:
        if tree:
            snapshot = json.loads(archive.read("snapshot.json"))
            prefix = overlay["source_fits_member_prefix"]
            fit_bytes = archive.read(prefix + "manifest.json")
            fit_manifest = json.loads(fit_bytes)
            if (snapshot["snapshot_id"] != overlay["source_snapshot_id"]
                    or snapshot["snapshot_id"] != digest({k: v for k, v in snapshot.items() if k != "snapshot_id"})
                    or hashlib.sha256(fit_bytes).hexdigest() != overlay["source_fit_manifest_sha256"]
                    or snapshot["files"].get(prefix + "manifest.json") != overlay["source_fit_manifest_sha256"]
                    or fit_manifest != prepared["fit_set_manifest"]):
                raise ValueError("B5 overlay original checkpoint or fitting manifest differs.")
        for model in overlay["models"]:
            old = original.get(model.get("source_fit_id"))
            if old is None or model["source_fit_id"] in replacements or model["fit_id"] in models:
                raise ValueError("Success overlay contains a foreign or duplicate model.")
            if tree:
                member = model["source_model_member"]
                relative = member.removeprefix(prefix)
                if not member.startswith(prefix) or relative not in fit_manifest["model_files"]:
                    raise ValueError("B5 overlay source model is outside the original fit manifest.")
                payload = archive.read(member)
                full = json.loads(payload)
                if (hashlib.sha256(payload).hexdigest() != model["source_model_sha256"]
                        or model["source_model_sha256"] != fit_manifest["files"][relative]
                        or snapshot["files"].get(member) != model["source_model_sha256"]
                        or _b5_model_summary(full) != old
                        or full["fit_id"] != digest({k: v for k, v in full.items() if k != "fit_id"})):
                    raise ValueError("B5 overlay source model bytes or identity differ.")
                old = full
            if (model["head"] != "success" or resolve_success_metric(model.get("success_metric", LEGACY_SUCCESS_METRIC)) != metric
                    or model["fit_id"] != digest({k: v for k, v in model.items() if k != "fit_id"})
                    or any(model[k] != old[k] for k in ("p", "group", "regime", "fold", "shuffle_seed",
                        "training_graph_ids", "reference_candidates", "feature_names", "feature_version", "settings", "rules"))):
                raise ValueError("Success overlay model changed its original training scope, features or reference targets.")
            if tree:
                if any(model[k] != old[k] for k in ("library_id", "split_digest", "feature_artifact_id")):
                    raise ValueError("B5 overlay library, split or frozen features differ.")
                payloads = []
                if set(model["booster_paths"]) != set(model["booster_files"]):
                    raise ValueError("B5 overlay must bind every native booster.")
                for name, checksum in model["booster_files"].items():
                    relative = model["booster_paths"][name]
                    native = (base / relative).resolve()
                    if not native.is_relative_to(base) or relative in native_files:
                        raise ValueError("B5 overlay native paths must be distinct and stay within its directory.")
                    payload = native.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != checksum or overlay["native_files"].get(relative) != checksum:
                        raise ValueError("B5 overlay native booster checksum mismatch.")
                    native_files[relative] = checksum
                    payloads.append(payload)
                validate_b5_model(model, load_b5_boosters(payloads))
            else:
                validate_b4_model(model)
            replacements[model["source_fit_id"]] = model
            models[model["fit_id"]] = model
    if tree and native_files != overlay["native_files"]:
        raise ValueError("B5 overlay native inventory differs from its models.")
    if set(replacements) != set(original):
        raise ValueError("Success overlay must cover every original success model exactly once.")
    predictions = {p["source_prediction_id"]: p for p in overlay["predictions"]}
    if (len(predictions) != len(overlay["predictions"])
            or set(predictions) != {p["prediction_id"] for p in prepared["predictions"]}):
        raise ValueError("Success overlay must cover every original prediction exactly once.")
    rows = []
    for old in prepared["predictions"]:
        new = predictions[old["prediction_id"]]
        raw, probability = new["raw_success_prediction"], new["success_probability"]
        if (new["source_success_fit_id"] != old["success_fit_id"]
                or new["success_fit_id"] != replacements[old["success_fit_id"]]["fit_id"]
                or not math.isfinite(raw) or probability != min(1., max(0., raw))):
            raise ValueError("Success overlay prediction model or clipping is inconsistent.")
        rows.append({**old, **{k: new[k] for k in ("success_fit_id", "raw_success_prediction", "success_probability")},
                     "success_metric": metric})
    view = {**prepared, "predictions": rows,
            "models": [replacements[m["fit_id"]] if m["head"] == "success" else m for m in prepared["models"]],
            "analysis_overlay": {"path": str(path), "overlay_id": overlay["overlay_id"],
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "scope": "Analysis-only success predictions; original angles, tasks and attempts remain frozen."}}
    if tree:
        explained = overlay["explanations"]
        _validate_b5_explanations(explained, {**view, "explanation_id": explained["explanation_id"]})
        if any(resolve_success_metric(row.get("success_metric", LEGACY_SUCCESS_METRIC)) != metric for row in explained["rows"]):
            raise ValueError("B5 overlay SHAP metric differs from its models and predictions.")
        view["analysis_explanations"] = explained
        view["analysis_overlay"].update(explanation_id=explained["explanation_id"], native_files=native_files,
            source_snapshot_id=overlay["source_snapshot_id"])
    return view


def summarize_b4_batch(batch, attempts_directory, b1_batch, b1_attempts, b2_cases, b3_batch, b3_attempts, *,
                       analysis_settings=None, success_overlay=None):
    return _summarize_learning_batch(batch, attempts_directory, b1_batch, b1_attempts,
                                     b2_cases, b3_batch, b3_attempts,
                                     analysis_settings=analysis_settings, success_overlay=success_overlay)


def summarize_b5_batch(batch, attempts_directory, b1_batch, b1_attempts, b2_cases,
                       b3_batch, b3_attempts, b4_batch, b4_attempts, *, analysis_settings=None,
                       success_overlay=None, b4_success_overlay=None):
    return _summarize_learning_batch(batch, attempts_directory, b1_batch, b1_attempts,
        b2_cases, b3_batch, b3_attempts, b4_batch=b4_batch, b4_attempts=b4_attempts, analysis_settings=analysis_settings,
        success_overlay=success_overlay, b4_success_overlay=b4_success_overlay)


def _summarize_learning_batch(batch, attempts_directory, b1_batch, b1_attempts, b2_cases,
                              b3_batch, b3_attempts, *, b4_batch=None, b4_attempts=None, analysis_settings=None,
                              success_overlay=None, b4_success_overlay=None):
    """Read graph-sized validated traces; no prediction, training or quantum calls."""
    from qaoa_study.analysis import build_b4_summary, build_b5_summary

    kind = "b4" if b4_batch is None else "b5"
    method = kind.upper()
    manifest, library, tasks = read_batch(batch)
    scoring = _scoring_settings(manifest, analysis_settings)
    metric = resolve_success_metric(scoring) if any(k in scoring for k in ("epsilon", "success_metric")) else resolve_success_metric()
    graphs = {g["iso_class_id"]: g for g in library["graphs"]}
    if manifest["config"].get("kind") != kind:
        raise ValueError(f"summarize-{kind} requires a {method} batch.")
    prepared, binding = (read_b4_inputs if kind == "b4" else read_b5_inputs)(batch, manifest)
    if success_overlay:
        prepared = _success_prediction_overlay(success_overlay, batch, manifest, prepared, metric)
    references = {(r["iso_class_id"], r["p"]): r for r in binding["references"]}
    targets = {(t["iso_class_id"], t["p"]) for t in tasks}
    selected, execution, audit = _warm_attempt_views(manifest, tasks, attempts_directory,
        references, tasks, reference_pairs=set(), by_depth=True, graphs=graphs, metric=metric)
    compare_execution = execution or {"environment": prepared["compute_environment"]}
    baseline, _, baseline_tasks = read_batch(b1_batch)
    if baseline["batch_id"] != binding["source_batch_id"]:
        raise ValueError("B4 requires its original bound B1 batch.")
    requested = [t for t in baseline_tasks if t["experiment_role"] == "evaluation"
                 and (t["iso_class_id"], t["p"]) in targets]
    b1_selected, b1_execution, b1_audit = _warm_attempt_views(baseline, baseline_tasks, b1_attempts,
        references, requested, reference_pairs=targets, by_depth=True, graphs=graphs, metric=metric)
    validate_b1_comparison(manifest, baseline, compare_execution, b1_execution)
    b2_tasks, b2_selected, fits, cases, audits = [], {}, [], [], {}
    expected = {(c["regime"], c["fold"]) for c in manifest["config"]["comparison_cases"]}
    observed = set()
    for case_batch, case_attempts in b2_cases:
        other, _, planned = read_batch(case_batch)
        if other["config"].get("kind") != "b2":
            raise ValueError("B4 comparison case must be B2.")
        fit, other_binding = read_b2_inputs(case_batch, other)
        key = fit["regime"], fit["fold"]
        if key in observed or key not in expected or other_binding["source_batch_id"] != baseline["batch_id"]:
            raise ValueError("Duplicate, undeclared or foreign B2 comparison case.")
        observed.add(key)
        other_refs = {(r["iso_class_id"], r["p"]): r for r in other_binding["references"]}
        if any(other_refs.get((t["iso_class_id"], t["p"])) != references.get((t["iso_class_id"], t["p"])) for t in planned):
            raise ValueError("B4/B2 scoring references differ.")
        chosen, other_execution, other_audit = _warm_attempt_views(other, planned, case_attempts,
            other_refs, planned, reference_pairs=set(), by_depth=True, graphs=graphs, metric=metric)
        validate_b1_comparison(manifest, other, compare_execution, other_execution)
        b2_tasks.extend(planned)
        b2_selected.update(chosen)
        fits.append(fit)
        cases.append({"regime": key[0], "fold": key[1], "batch_id": other["batch_id"], "fit_id": fit["fit_id"],
                      "execution_id": other_execution["execution_id"] if other_execution else None})
        audits[other["batch_id"]] = other_audit
    if observed != expected:
        raise ValueError("All declared B2 comparison scopes are required.")
    other, _, b3_tasks = read_batch(b3_batch)
    if other["config"].get("kind") != "b3":
        raise ValueError("B4 requires a B3 comparison batch.")
    b3_prepared, b3_binding = read_b3_inputs(b3_batch, other)
    if b3_binding["source_batch_id"] != baseline["batch_id"]:
        raise ValueError("B3 comparison uses a different original B1 batch.")
    other_refs = {(r["iso_class_id"], r["p"]): r for r in b3_binding["references"]}
    if any(other_refs.get(key) != references[key] for key in targets):
        raise ValueError("B4/B3 scoring references differ.")
    b3_selected, b3_execution, b3_audit = _warm_attempt_views(other, b3_tasks, b3_attempts,
        other_refs, b3_tasks, reference_pairs=set(), by_depth=True, graphs=graphs, metric=metric)
    validate_b1_comparison(manifest, other, compare_execution,
                           b3_execution or {"environment": b3_prepared["environment"]})
    extra = {}
    if kind == "b5":
        fourth, _, b4_tasks = read_batch(b4_batch)
        if fourth["config"].get("kind") != "b4" or fourth["issues"]:
            raise ValueError("B5 requires its valid original B4 comparison batch.")
        b4_prepared, b4_binding = read_b4_inputs(b4_batch, fourth)
        if b4_success_overlay:
            b4_prepared = _success_prediction_overlay(b4_success_overlay, b4_batch, fourth, b4_prepared, metric)
        other_refs = {(r["iso_class_id"], r["p"]): r for r in b4_binding["references"]}
        if (b4_binding["source_batch_id"] != baseline["batch_id"]
                or any(other_refs.get(key) != references[key] for key in targets)):
            raise ValueError("B5/B4 scoring references differ.")
        b4_selected, b4_execution, b4_audit = _warm_attempt_views(fourth, b4_tasks, b4_attempts,
            other_refs, b4_tasks, reference_pairs=set(), by_depth=True, graphs=graphs, metric=metric)
        validate_b1_comparison(manifest, fourth, compare_execution,
                              b4_execution or {"environment": b4_prepared["compute_environment"]})
        extra = {"b4_tasks": b4_tasks, "b4_selected": b4_selected,
            "b4_predictions": b4_prepared["predictions"],
            "b4_success_models": [m for m in b4_prepared["models"] if m["head"] == "success"],
            "explanations": (prepared["analysis_explanations"] if "analysis_explanations" in prepared
                             else read_b5_explanations(batch, prepared))["rows"]}
    summary = (build_b4_summary if kind == "b4" else build_b5_summary)(
        library, tasks, selected, references, b1_tasks=requested, b1_selected=b1_selected,
        b2_tasks=b2_tasks, b2_selected=b2_selected, b3_tasks=b3_tasks, b3_selected=b3_selected,
        predictions=prepared["predictions"], success_models=[m for m in prepared["models"] if m["head"] == "success"],
        settings=scoring, **extra)
    original_training = {(r["iso_class_id"], r["p"]): r for f in fits for r in f["reference_binding"]["references"]}
    if any(original_training.get((r["iso_class_id"], r["p"])) != r for r in prepared["training_binding"]["references"]):
        raise ValueError("B4 and B2 disagree on shared original training references.")
    training_costs = _b3_training_costs(baseline, baseline_tasks, b1_attempts, fits, [1, 2])
    summary.update(batch_id=manifest["batch_id"], execution_id=execution["execution_id"] if execution else None,
        preparation_id=prepared["preparation_id"], fit_set_id=prepared["fit_set_id"], reference_binding_digest=digest(binding),
        b1_batch_id=baseline["batch_id"], b1_execution_id=b1_execution["execution_id"] if b1_execution else None,
        b3_batch_id=other["batch_id"], b3_execution_id=b3_execution["execution_id"] if b3_execution else None,
        b2_cases=cases, analysis_source=source_identity(), batch_issues=manifest["issues"],
        comparison_inputs={"b1_batch": str(b1_batch), "b1_attempts": str(b1_attempts),
            "b2_cases": [[str(b), str(a)] for b, a in b2_cases], "b3_batch": str(b3_batch), "b3_attempts": str(b3_attempts)},
        input_audits={method: audit, "B1": b1_audit, "B2": audits, "B3": b3_audit},
        shared_training_reference_costs=training_costs, success_label_costs=prepared["success_label_costs"],
        b3_preparation_costs=b3_prepared["preparation"],
        evaluation_reference_costs_by_depth=b1_audit["reference_attempt_costs_by_depth"],
        learning_costs={"feature_construction": prepared["feature_costs"], "prediction": prepared["prediction_timings"],
            "artifact_io": prepared["artifact_io"],
            "fits": [{"fit_id": m["fit_id"], "head": m["head"], "fit_seconds": m["fit_seconds"],
                      "timings": m["timings"],
                      "learning_environment": m["learning_environment"]} for m in prepared["models"]]},
        cost_scope="B1 reference attempts form a shared union across B2/B4; never sum per-model copies. "
            "Original training evaluation pools are shared success-label overhead. B3 external constant cost and "
            "historical feature construction are unmeasured. Full actual attempts include retained faults/retries.")
    if kind == "b5":
        summary["b4_batch_id"] = fourth["batch_id"]
        summary["b4_execution_id"] = b4_execution["execution_id"] if b4_execution else None
        summary["input_audits"]["B4"] = b4_audit
        summary["comparison_inputs"].update(b4_batch=str(b4_batch), b4_attempts=str(b4_attempts))
        summary["learning_costs"]["explanations"] = prepared["explanation_timings"]
        summary["cost_scope"] = summary["cost_scope"].replace("B2/B4", "B2/B4/B5")
    summary["complete"] = summary["complete"] and not manifest["issues"]
    summary["provisional"] = not summary["complete"]
    summary["success_prediction_overlays"] = {
        method: prepared["analysis_overlay"] for method, prepared in
        [(method, prepared), *(([("B4", b4_prepared)]) if kind == "b5" else [])]
        if "analysis_overlay" in prepared}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Freeze reproducible tasks without computing any scientific result")
    plan.add_argument("--library", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    for method, learner, controls in (("b4", "Ridge", "B1/B2/B3"), ("b5", "XGBoost", "B1/B2/B3/B4")):
        fit = commands.add_parser(f"fit-{method}", help=f"Fit the declared CPU {learner} models on training graphs only")
        for option in ("batch", "attempts", "references", "config", "output"):
            fit.add_argument("--" + option, required=True)
        predict = commands.add_parser(f"predict-{method}", help="Freeze predictions before evaluation references")
        for option in ("fits", "library", "output"):
            predict.add_argument("--" + option, required=True)
        plan_learned = commands.add_parser(f"plan-{method}", help="Bind frozen predictions to original references")
        for option in ("batch", "attempts", "references", "predictions", "config", "output"):
            plan_learned.add_argument("--" + option, required=True)
        summary = commands.add_parser(f"summarize-{method}", help=f"Read frozen outputs and {controls} controls")
        options = ["batch", "attempts", "b1-batch", "b1-attempts", "b3-batch", "b3-attempts", "output"]
        if method == "b5":
            options += ["b4-batch", "b4-attempts"]
        for option in options:
            summary.add_argument("--" + option, required=True)
        summary.add_argument("--analysis-settings", help="JSON scoring settings for a new derived analysis")
        summary.add_argument("--success-overlay", help="Verified analysis-only replacement success predictions and B5 SHAP")
        if method == "b5":
            summary.add_argument("--b4-success-overlay", help="Replacement B4 success predictions for this comparison")
        summary.add_argument("--b2-case", nargs=2, action="append", required=True, metavar=("BATCH", "ATTEMPTS"))
    for command in ("fit-b2", "plan-b2"):
        sub = commands.add_parser(command, help="Fit or freeze B2 using complete original training references")
        for option in ("batch", "attempts", "references", "output"):
            sub.add_argument("--" + option, required=True)
        if command == "fit-b2":
            sub.add_argument("--regime", choices=("random", "lofo"), default="random")
            sub.add_argument("--fold", choices=("regular", "er", "ba", "sbm"))
        else:
            sub.add_argument("--fit", required=True)
            sub.add_argument("--config", required=True)
    warm_summary = commands.add_parser("summarize-b2", help="Read bound B2 and comparable original B1 traces")
    for option in ("batch", "attempts", "b1-batch", "b1-attempts", "output"):
        warm_summary.add_argument("--" + option, required=True)
    warm_summary.add_argument("--analysis-settings", help="JSON scoring settings for a new derived analysis")
    prepare = commands.add_parser("prepare-b3", help="Explicitly compute and freeze graph-only B3 initial angles")
    for option in ("library", "config", "output"):
        prepare.add_argument("--" + option, required=True)
    prepare.add_argument("--backend", choices=("default.qubit", "lightning.gpu"), default="default.qubit")
    b3_plan = commands.add_parser("plan-b3", help="Bind frozen B3 initial angles to original B1 references")
    for option in ("batch", "attempts", "references", "initializations", "config", "output"):
        b3_plan.add_argument("--" + option, required=True)
    b3_summary = commands.add_parser("summarize-b3", help="Read B3 and all declared original B1/B2 comparison cases")
    for option in ("batch", "attempts", "b1-batch", "b1-attempts", "output"):
        b3_summary.add_argument("--" + option, required=True)
    b3_summary.add_argument("--analysis-settings", help="JSON scoring settings for a new derived analysis")
    b3_summary.add_argument("--b2-case", nargs=2, action="append", required=True, metavar=("BATCH", "ATTEMPTS"))
    run = commands.add_parser("run", help="Check only unless --execute is explicit")
    run.add_argument("--batch", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--backend", choices=("default.qubit", "lightning.gpu"), default="default.qubit")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--partitioned", action="store_true", help="Store graph/role groups and seal complete groups")
    run.add_argument("--graph-ids", nargs="+", help="Restrict execution to these frozen graph identities")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--max-tasks", type=int, help="Maximum actual attempts during this invocation")
    run.add_argument("--roles", nargs="+", choices=("tier1", "reference", "evaluation", "gradient_diagnostic", "warm_start"))
    for command in ("freeze", "summarize"):
        sub = commands.add_parser(command, help="Read existing attempts; no QAOA/solver/grid execution")
        sub.add_argument("--batch", required=True)
        sub.add_argument("--attempts", required=True)
        sub.add_argument("--output", required=True)
        if command == "summarize":
            sub.add_argument("--analysis-settings", help="JSON scoring settings for a new derived analysis")
            sub.add_argument("--references", required=True)
            sub.add_argument("--partitioned-output", action="store_true", help="Write bounded-memory per-graph tables and a root index")
    args = parser.parse_args(argv)
    try:
        scoring_override = read_json(args.analysis_settings) if getattr(args, "analysis_settings", None) else None
        if args.command not in ("run", "fit-b4", "fit-b5") and Path(args.output).exists():
            raise FileExistsError("Choose a new output path; frozen outputs are not overwritten.")
        if args.command in ("fit-b4", "fit-b5"):
            result = (save_b4_fits if args.command == "fit-b4" else save_b5_fits)(args.output, args.batch, args.references, args.attempts, read_json(args.config))
            print(json.dumps({"fit_set_id": result["manifest"]["fit_set_id"], "models": len(result["models"])}))
            return 0
        if args.command in ("predict-b4", "predict-b5"):
            result = (save_b4_predictions if args.command == "predict-b4" else save_b5_predictions)(args.output, args.fits, args.library)
            print(json.dumps({"preparation_id": result["preparation_id"], "predictions": len(result["predictions"])}))
            return 0
        if args.command in ("plan-b4", "plan-b5"):
            manifest = (save_b4_batch if args.command == "plan-b4" else save_b5_batch)(args.output, args.predictions, args.batch, args.references,
                                    args.attempts, read_json(args.config))
            print(json.dumps({k: manifest[k] for k in ("batch_id", "task_count", "issues")}))
            return 2 if manifest["issues"] else 0
        if args.command in ("summarize-b4", "summarize-b5"):
            summarizer = summarize_b4_batch if args.command == "summarize-b4" else summarize_b5_batch
            extra = {} if args.command == "summarize-b4" else {"b4_batch": args.b4_batch, "b4_attempts": args.b4_attempts,
                "b4_success_overlay": args.b4_success_overlay}
            summary = summarizer(args.batch, args.attempts, args.b1_batch, args.b1_attempts,
                                 args.b2_case, args.b3_batch, args.b3_attempts, analysis_settings=scoring_override,
                                 success_overlay=getattr(args, "success_overlay", None), **extra)
            save_summary(args.output, summary)
            print(json.dumps({"complete": summary["complete"], "comparisons": len(summary["comparisons"])}))
            return 0 if summary["complete"] else 2
        if args.command == "prepare-b3":
            prepared = save_b3_initializations(args.output, args.library, read_json(args.config), backend=args.backend)
            print(json.dumps({"initialization_set_id": prepared["initialization_set_id"],
                              "initializations": len(prepared["initializations"])}))
            return 0
        if args.command == "plan-b3":
            manifest = save_b3_batch(args.output, args.initializations, args.batch, args.references,
                                    args.attempts, read_json(args.config))
            print(json.dumps({k: manifest[k] for k in ("batch_id", "task_count", "issues")}))
            return 2 if manifest["issues"] else 0
        if args.command == "summarize-b3":
            summary = summarize_b3_batch(args.batch, args.attempts, args.b1_batch, args.b1_attempts, args.b2_case, analysis_settings=scoring_override)
            save_summary(args.output, summary)
            print(json.dumps({"complete": summary["complete"], "comparisons": len(summary["comparisons"])}))
            return 0 if summary["complete"] else 2
        if args.command == "fit-b2":
            fit = save_b2_fit(args.output, args.batch, args.references, args.attempts, regime=args.regime, fold=args.fold)
            print(json.dumps({"fit_id": fit["fit_id"], "training_graphs": len(fit["training_graph_ids"])}))
            return 0
        if args.command == "plan-b2":
            manifest = save_b2_batch(args.output, args.fit, args.batch, args.references, args.attempts, read_json(args.config))
            print(json.dumps({k: manifest[k] for k in ("batch_id", "task_count", "issues")}))
            return 2 if manifest["issues"] else 0
        if args.command == "summarize-b2":
            summary = summarize_b2_batch(args.batch, args.attempts, args.b1_batch, args.b1_attempts, analysis_settings=scoring_override)
            save_summary(args.output, summary)
            print(json.dumps({"complete": summary["complete"], "comparisons": len(summary["comparisons"])}))
            return 0 if summary["complete"] else 2
        if args.command == "plan":
            manifest = save_batch(args.output, args.library, read_json(args.config))
            print(json.dumps({key: manifest[key] for key in ("batch_id", "task_count", "counts", "issues", "coverage")}, indent=2))
            return 2 if manifest["issues"] else 0
        if args.command == "run":
            report = run_worker(args.batch, args.output, backend=args.backend, execute=args.execute,
                                shard_index=args.shard_index, shard_count=args.shard_count,
                                max_tasks=args.max_tasks, roles=args.roles,
                                partitioned=args.partitioned, graph_ids=args.graph_ids)
            print(json.dumps(report, indent=2))
            return 1 if report.get("unresolved_execution_faults") else 2 if report["issues"] else 0
        if args.command == "freeze":
            manifest, _, tasks = read_batch(args.batch)
            attempts, _ = load_attempts(manifest, tasks, args.attempts, metadata_only=True)
            references = freeze_references(manifest, tasks, select_attempts(attempts))
            save_references(args.output, references)
            print(json.dumps({"references": len(references), "complete": sum(r["complete"] for r in references.values())}))
            return 0 if all(ref["complete"] for ref in references.values()) else 2
        if args.partitioned_output:
            index = save_partitioned_summary(args.output, args.batch, args.attempts, args.references, analysis_settings=scoring_override)
            print(json.dumps({"complete": index["complete"], "counts": index["counts"], "output": args.output}))
            return 0 if index["complete"] else 2
        summary = summarize_batch(args.batch, args.attempts, args.references, analysis_settings=scoring_override)
        save_summary(args.output, summary)
        print(json.dumps({"graph_depth_rows": len(summary["graph_qaoa"]),
                          "restart_rows": len(summary["restarts"]), "output": args.output}))
        return 0
    except (ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
