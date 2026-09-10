"""Plan, explicitly execute, freeze and summarize the same Plan A task list."""

import argparse
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys

from qaoa_study.analysis import build_summary
from qaoa_study.experiments import (
    _record_directory, digest, freeze_references, iter_attempt_groups, load_attempts, read_batch, read_references,
    run_worker, save_batch, save_references, select_attempts, source_identity,
)
from qaoa_study.records import execution_failed, read_json, write_once_json


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


def _empty_summary(library, manifest):
    summary = build_summary(library, [], {}, {}, manifest["config"].get("analysis"))
    summary.update(all_attempt_costs_by_role={}, selected_attempts={})
    return summary


def _append_graph_summary(summary, library, graph, manifest, tasks, attempts, references, audit, completed_ids):
    selected = select_attempts(attempts)
    part = build_summary({**library, "graphs": [graph]}, tasks, selected, references,
                         manifest["config"].get("analysis"))
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


def summarize_batch(batch, attempts_directory, references_directory):
    """Reconstruct one graph at a time; return the legacy combined tables."""
    manifest, library, tasks, references = _summary_inputs(batch, references_directory)
    graphs = {graph["iso_class_id"]: graph for graph in library["graphs"]}
    summary = _empty_summary(library, manifest)
    audit, completed_ids, execution = {}, set(), None
    for identity, graph_tasks, attempts, graph_audit, execution in _iter_graph_results(manifest, tasks, attempts_directory):
        _append_graph_summary(summary, library, graphs[identity], manifest, graph_tasks, attempts,
                              references, graph_audit, completed_ids)
        _merge_audit(audit, graph_audit)
    _finish_summary(summary, manifest, execution, audit, completed_ids, source_identity())
    return summary


def save_partitioned_summary(output, batch, attempts_directory, references_directory):
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
        summary = _empty_summary(library, manifest)
        counts = _append_graph_summary(summary, library, graphs[identity], manifest, graph_tasks, attempts,
                                       references, audit, completed_ids)
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
             "settings": summary["settings"] if entries else manifest["config"].get("analysis"), "coverage": manifest["coverage"],
             "attempt_selection_rule": manifest["attempt_selection_rule"], "complete": complete,
             "completion_scope": "All configured tasks execution-valid and all required frozen references complete; scientific label counts reported separately.",
             "counts": totals, "table_rows": row_counts, "graphs": entries}
    write_once_json(output / "manifest.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Freeze reproducible tasks without computing any scientific result")
    plan.add_argument("--library", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
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
    run.add_argument("--roles", nargs="+", choices=("tier1", "reference", "evaluation", "gradient_diagnostic"))
    for command in ("freeze", "summarize"):
        sub = commands.add_parser(command, help="Read existing attempts; no QAOA/solver/grid execution")
        sub.add_argument("--batch", required=True)
        sub.add_argument("--attempts", required=True)
        sub.add_argument("--output", required=True)
        if command == "summarize":
            sub.add_argument("--references", required=True)
            sub.add_argument("--partitioned-output", action="store_true", help="Write bounded-memory per-graph tables and a root index")
    args = parser.parse_args(argv)
    try:
        if args.command != "run" and Path(args.output).exists():
            raise FileExistsError("Choose a new output path; frozen outputs are not overwritten.")
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
            index = save_partitioned_summary(args.output, args.batch, args.attempts, args.references)
            print(json.dumps({"complete": index["complete"], "counts": index["counts"], "output": args.output}))
            return 0 if index["complete"] else 2
        summary = summarize_batch(args.batch, args.attempts, args.references)
        save_summary(args.output, summary)
        print(json.dumps({"graph_depth_rows": len(summary["graph_qaoa"]),
                          "restart_rows": len(summary["restarts"]), "output": args.output}))
        return 0
    except (ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
