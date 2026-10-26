"""Run the matched B1–B3 depth and finite-shot COBYQA study from saved graphs.

Use plan to inspect the frozen graph roster, start to launch/resume on ICRN,
status --watch 10 to monitor, and stop to drain active tasks safely.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qaoa_study import experiments as exp
from qaoa_study.graphs import graph_split_coverage
from qaoa_study.optimize import ShotOptimizerSettings
from qaoa_study.parallel import ParallelSettings
from qaoa_study.records import annotate_library, read_json, read_library, save_library, write_once_json
from scripts import shot_benchmark as shot


SIZES = (12, 14, 16, 18)
FAMILIES = ("regular", "er", "ba", "sbm", "ws", "rgg")


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parent_library(path):
    """Read graph inputs; retain draw-log identity without parsing its 133 MB log."""
    import pyarrow.parquet as pq

    manifest = read_json(path / "manifest.json")
    for name in ("graphs.parquet", "config.json", "splits.json"):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Graph input changed: {path / name}")
    rows = pq.read_table(path / "graphs.parquet").to_pylist()
    for row in rows:
        for field in ("provenance", "iso_mappings", "feature_missing"):
            row[field] = json.loads(row.pop(field + "_json"))
    splits = read_json(path / "splits.json")
    if any(set(ids) != {g["iso_class_id"] for g in rows if g["graph_split"] == split}
           for split, ids in splits.items()):
        raise ValueError(f"Graph membership differs from its frozen split: {path}")
    return {**manifest["library_metadata"], "graphs": rows}, {
        "library_id": manifest["library_metadata"]["library_id"],
        "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        "parent_files": manifest["files"],
    }


def load_selection(data_root, per_cell=10, seed=20261026):
    """Return (derived_library, roster) from a data/manual directory.

    Evaluation uses six families at four sizes, with two settings each. Rank
    saved held-out graphs first, then other nontraining graphs, using only
    seeded identity hashes. Original n12/n16 training membership is retained.
    Missing exact/feature annotations are identified for the caller to compute.
    """
    if not isinstance(per_cell, int) or per_cell < 1:
        raise ValueError("per_cell must be a positive integer.")
    data_root = Path(data_root)
    parents, provenance = {}, {}
    for name in ("b1-full-a100-v1", "new-families-full-v1", "size-ws-v2", "size-rgg35-v2"):
        library, evidence = _parent_library(data_root / name / "library")
        parents[name] = library
        provenance[name] = {"path": name + "/library", **evidence}
    raw_path = data_root / "size-original4-v2/generated.json"
    saved = read_json(raw_path)
    raw = saved["library"]
    if _digest(raw) != saved["digest"]:
        raise ValueError(f"Frozen graph draws changed: {raw_path}")
    parents["size-original4-v2"] = raw
    provenance["size-original4-v2"] = {
        "path": "size-original4-v2/generated.json", "library_id": raw["library_id"],
        "generated_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "library_digest": saved["digest"],
    }

    original = parents["b1-full-a100-v1"]["graphs"]
    excluded_training = {g["iso_class_id"] for g in original if g["graph_split"] == "training"}
    training = [deepcopy(g) for g in original
                if g["iso_class_id"] in excluded_training and g["n"] in SIZES]
    pools = defaultdict(list)
    origins = {}
    for name, library in parents.items():
        for graph in library["graphs"]:
            source = graph["provenance"][0]
            family, n = source["family"], graph["n"]
            if n not in SIZES or (name == "new-families-full-v1" and family != "ws"):
                continue
            origins[graph["iso_class_id"]] = name
            if graph["iso_class_id"] not in excluded_training:
                pools[graph["representative_cell"]].append(graph)

    selected, cells, roster_rows, selection = [], [], [], []
    feature_names = original[0]["features"].keys()
    for family in FAMILIES:
        settings = ("rewire-0.1", "rewire-0.5") if family == "ws" else (
            ("target-degree-3", "target-degree-5") if family == "rgg" else ("k3", "k5"))
        for n in SIZES:
            for setting in settings:
                cell_id = f"n{n}:{family}:{setting}"
                pool = sorted(pools[cell_id], key=lambda g: (
                    g["graph_split"] != "evaluation", _digest([seed, cell_id, g["iso_class_id"]])))
                if len(pool) < per_cell:
                    raise ValueError(f"{cell_id}: need {per_cell} nontraining graphs; found {len(pool)}.")
                chosen = pool[:per_cell]
                source = chosen[0]["provenance"][0]
                cells.append({key: deepcopy(source[key]) for key in
                              ("cell_id", "n", "family", "target_degree", "generation_group") if key in source})
                roster_rows.append({"family": family, "n": n, "setting": setting,
                                    "evaluation": len(chosen), "available_nontraining": len(pool),
                                    "inherited_evaluation": sum(g["graph_split"] == "evaluation" for g in chosen),
                                    "promoted_tier1": sum(g["tier"] == 1 for g in chosen)})
                for graph in chosen:
                    graph_id = graph["iso_class_id"]
                    selection.append({"graph_id": graph_id, "source": origins[graph_id],
                                      "source_split": graph["graph_split"], "source_tier": graph["tier"]})
                    record = deepcopy(graph)
                    record.update(tier=2, graph_split="evaluation")
                    if "features" not in record:
                        record.update(features=dict.fromkeys(feature_names), feature_missing={},
                                      feature_version=None, C_star=None, optimal_cut_side=None,
                                      exact_status="not_computed_development")
                    selected.append(record)

    graphs = training + selected
    counts = {"tier2": len(graphs), "training": len(training), "evaluation": len(selected)}
    split_summary = {"requested": counts, "actual": counts, "shortfall": dict.fromkeys(counts, 0),
                     "provisional": False, "split_version": "weekly-inherited-training-v1",
                     "split_seed": seed, "coverage": graph_split_coverage(graphs, cells),
                     "algorithm": "Preserve original n12/n16 training; prefer held-out evaluation, then "
                                  "nontraining Tier1; rank by SHA256(seed, cell_id, graph_id)."}
    identity = {"parents": provenance, "per_cell": per_cell, "seed": seed,
                "training_graph_ids": sorted(g["iso_class_id"] for g in training), "selection": selection}
    library_id = "weekly-depth-20261026-v1:" + _digest(identity)
    library = {"library_id": library_id, "graphs": graphs, "attempts": [], "cells": cells,
               "config": {"dataset_namespace": "weekly-depth-20261026-v1", "sizes": list(SIZES),
                          "families": list(FAMILIES), "seed": seed, "split_seed": seed,
                          "tier2_count": len(graphs), "training_count": len(training),
                          "evaluation_per_cell": per_cell, "exact_sizes": list(SIZES),
                          "exact_chunk_size": 65536},
               "split_summary": split_summary, "selection_provenance": identity,
               "parent_library_ids": sorted(v["library_id"] for v in provenance.values()),
               "wl_settings": deepcopy(parents["b1-full-a100-v1"]["wl_settings"])}
    training_counts = Counter((g["provenance"][0]["family"], g["n"]) for g in training)
    roster = {"library_id": library_id, "training": len(training), "evaluation": len(selected),
              "cells": roster_rows, "training_counts": [
                  {"family": family, "n": n, "count": count}
                  for (family, n), count in sorted(training_counts.items())],
              "annotation_required": [g["iso_class_id"] for g in graphs if g.get("C_star") is None],
              "parents": provenance}
    return library, roster


def _read_plan(root):
    plan = read_json(Path(root) / "plan.json")
    if exp.digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Frozen study plan changed.")
    return plan


def plan_study(root, config_path=None, data_root=None, library_path=None):
    """Freeze graph selection and costs; quantum and exact computations start later."""
    root = Path(root).resolve()
    if (root / "plan.json").exists():
        plan = _read_plan(root)
        if config_path is not None and exp.digest(read_json(config_path)) != plan["input_config_digest"]:
            raise ValueError("Different settings require a new study root.")
        return plan
    config = read_json(config_path or REPO / "configs/depth-noise-study.json")
    input_config_digest = exp.digest(config)
    if library_path:
        library = read_library(library_path)
        roster = {"library_id": library["library_id"], "input_library": str(Path(library_path).resolve())}
    else:
        if data_root is None:
            raise ValueError("First plan/start requires --data-root PATH or --library PATH.")
        library, roster = load_selection(data_root, **config["selection"])
    training = sum(g["graph_split"] == "training" for g in library["graphs"])
    evaluation = sum(g["graph_split"] == "evaluation" for g in library["graphs"])
    depths, count = config["depths"], len(library["graphs"])
    rules = read_json(REPO / config.pop("b3_rules_file"))
    config["b3_rules"] = {str(p): rules[str(p)] for p in depths if p >= 3}
    config["baseline"].update(depths=depths, expected_library={
        "graphs": count, "tier2": count, "training": training, "evaluation": evaluation})
    for name in ("b2", "b3"):
        config[name]["depths"] = depths
    config["b3"]["b3_rules"] = config["b3_rules"]
    exp.validate_config(config["baseline"])
    ShotOptimizerSettings(**config["shot"])
    references = sum(1 if p == 1 else exp._reference_restarts(config["baseline"], p) for p in depths)
    tasks = {"references": count * references,
             "b1": evaluation * len(depths) * config["baseline"]["evaluation_restarts"],
             "b2": evaluation * len(depths) * 2, "b3": evaluation * len(depths),
             "shots": evaluation * len(depths) * config["repetitions"] * 11}
    plan = {"version": "depth-noise-study-v1", "config": config, "source": exp.source_identity(),
            "input_config_digest": input_config_digest,
            "library": library, "roster": roster, "training": training, "evaluation": evaluation,
            "task_counts": tasks, "planned": sum(tasks.values()),
            "scope": "References on all selected graphs; B1-B3 evaluation only on held-out graphs. "
                     "Tier1 and gradient diagnostics in the generic batch are outside this study."}
    plan["plan_id"] = exp.digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    write_once_json(root / "plan.json", plan)
    shot._stamp(root, state="planned", phase="planned", planned=plan["planned"], completed=0)
    return plan


def _directory(destination, build):
    """Publish complete library/batch artifacts; interrupted preparation is retriable."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        stage.rmdir()
        build(stage)
        stage.rename(destination)


def _prepare_library(root, plan, stopped):
    target = root / "library"
    if target.exists():
        return True
    library = deepcopy(plan["library"])
    for record in library["graphs"]:
        if record.get("C_star") is not None and record.get("feature_version"):
            continue
        if stopped():
            return False
        saved = root / "annotations" / (exp.digest(record["iso_class_id"]) + ".json")
        if saved.exists():
            annotated = read_json(saved)
        else:
            annotated = annotate_library({**library, "graphs": [record]}, exact_sizes=[record["n"]])["graphs"][0]
            write_once_json(saved, annotated)
        record.update(annotated)
        shot._stamp(root, phase="annotations", active_graph=record["iso_class_id"])
    _directory(target, lambda path: save_library(path, library))
    return True


def run_study(root, parallel):
    """Resume whole recorded attempts and drain active work after a stop request."""
    root = Path(root).resolve()
    plan = _read_plan(root)
    config = plan["config"]
    if exp.source_identity() != plan["source"]:
        raise ValueError("The checkout changed after planning. Keep this checkout or create a new study root.")
    counts, total = plan["task_counts"], plan["planned"]
    receipts = root / "completed"
    stopped = lambda: (root / "stop.request").exists()
    finished = sum(count for name, count in counts.items() if (receipts / (name + ".json")).exists())
    shot._stamp(root, state="running", phase="prepare", pid=os.getpid(), error=None,
                planned=total, completed=finished, active_tasks=[])

    def execute(name, function, *, sampled=False):
        nonlocal finished
        receipt = receipts / (name + ".json")
        if receipt.exists():
            return True
        if stopped():
            return False
        expected = counts[name]

        def progress(report):
            faults = report.get("execution_failures", report.get("unresolved_execution_faults", 0))
            done = report["completed"] - faults if sampled else expected - report["missing_tasks"] - faults
            shot._stamp(root, phase=name, completed=finished + done, stage_completed=done,
                        stage_planned=expected, active_tasks=report.get("active_tasks", []),
                        execution_failures=faults, resources=report.get("resources", {}))

        shot._stamp(root, phase=name, stage_completed=0, stage_planned=expected)
        report = function(progress)
        progress(report)
        complete = (report["complete"] and not report["execution_failures"] if sampled else
                    not report["missing_tasks"] and not report["unresolved_execution_faults"])
        if complete:
            write_once_json(receipt, report)
            finished += expected
        return complete

    with ExitStack() as context:
        if config["backend"] == "lightning.gpu":
            from qaoa_study.parallel import NvmlMemory, _worker_environment
            probe = NvmlMemory(parallel.gpu_device)
            context.callback(probe.close)
            context.enter_context(_worker_environment(probe.uuid, False))
        if not _prepare_library(root, plan, stopped):
            return False
        verification = root / "backend-verification.json"
        if not verification.exists() and not stopped():
            from scripts.verify_backend import verify_shots
            shot._stamp(root, phase="verify")
            checked = verify_shots(config["backend"], config["depths"])
            if not checked["passed"]:
                raise RuntimeError(f"Backend verification failed: {checked}")
            write_once_json(verification, checked)
        if stopped():
            return False
        baseline, attempts, refs = root / "baseline", root / "baseline-attempts", root / "references"
        _directory(baseline, lambda path: exp.save_batch(path, root / "library", config["baseline"]))
        common = {"backend": config["backend"], "execute": True, "parallel": parallel, "should_stop": stopped}
        if not execute("references", lambda callback: exp.run_worker(
                baseline, attempts, roles=["reference"], progress=callback, **common)):
            return False
        if not refs.exists():
            shot._stamp(root, phase="freeze-references")
            manifest, _, tasks = exp.read_batch(baseline)
            references = {}
            for group, rows, _ in exp.iter_attempt_groups(manifest, tasks, attempts, metadata_only=True):
                references.update(exp.freeze_references(manifest, group, exp.select_attempts(rows)))
            if not all(ref["complete"] for ref in references.values()):
                raise RuntimeError("A reference failed its declared validity check; inspect its saved attempts.")
            _directory(refs, lambda path: exp.save_references(path, references))
        evaluation = [g["iso_class_id"] for g in plan["library"]["graphs"] if g["graph_split"] == "evaluation"]
        if not execute("b1", lambda callback: exp.run_worker(
                baseline, attempts, roles=["evaluation"], graph_ids=evaluation, progress=callback, **common)):
            return False
        fit = root / "b2-fit.json"
        if not fit.exists():
            shot._stamp(root, phase="fit-b2")
            exp.save_b2_fit(fit, baseline, refs, attempts, depths=config["depths"])
        b2 = root / "b2"
        _directory(b2, lambda path: exp.save_b2_batch(path, fit, baseline, refs, attempts, config["b2"]))
        if not execute("b2", lambda callback: exp.run_worker(
                b2, root / "b2-attempts", progress=callback, **common)):
            return False
        initializations, b3 = root / "b3-initializations.json", root / "b3"
        if not initializations.exists():
            exp.save_b3_initializations(initializations, root / "library", config["b3"], backend=config["backend"])
        _directory(b3, lambda path: exp.save_b3_batch(path, initializations, baseline, refs, attempts, config["b3"]))
        if not execute("b3", lambda callback: exp.run_worker(
                b3, root / "b3-attempts", progress=callback, **common)):
            return False
        shot_plan = root / "shot-plan.json"
        if not shot_plan.exists():
            shot._stamp(root, phase="plan-shots")
            shot.plan_benchmark(baseline, [fit], None, None, shot_plan, references=[refs],
                depths=config["depths"], repetitions=config["repetitions"], settings=ShotOptimizerSettings(**config["shot"]),
                methods=["COBYQA"], backend=config["backend"], seed=config["shot_seed"],
                initializations=["B1", "B2", "B3"], b3_rules=config["b3_rules"])
        if not execute("shots", lambda callback: shot.run_benchmark(shot_plan, root / "shot-attempts",
                execute=True, retry_interrupted=True, parallel=parallel, should_stop=stopped, progress=callback), sampled=True):
            return False
    shot._stamp(root, state="complete", phase="complete", completed=total, active_tasks=[])
    return True


def _work(root, parallel):
    try:
        if not run_study(root, parallel):
            shot._stamp(root, state="paused", active_tasks=[])
    except BaseException as error:
        shot._stamp(root, state="paused", active_tasks=[], error=f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "start", "run", "status", "stop", "_background"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        if name in ("plan", "start", "run"):
            command.add_argument("--data-root", help="Existing ICRN data/manual directory")
            command.add_argument("--library", help="Use an existing complete library instead of historical selection")
            command.add_argument("--config", help="Defaults to configs/depth-noise-study.json for a new study")
        if name in ("start", "run"):
            shot._parallel_arguments(command, workers=4, inherit=True)
        if name == "status":
            command.add_argument("--watch", type=float, default=0)
            command.add_argument("--json", action="store_true")
        if name == "_background":
            command.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command in ("status", "stop"):
        return shot._lifecycle(args)
    if args.command == "_background":
        with os.fdopen(args.lock_fd, "a+b"):
            return _work(root, ParallelSettings(**read_json(root / "runner.json")["parallel"]))
    if args.command != "plan" and shot._status(root)["active_lock"]:
        print(json.dumps(shot._status(root), indent=2))
        return
    plan = plan_study(root, args.config, args.data_root, args.library)
    if args.command == "plan":
        print(json.dumps({key: plan[key] for key in ("plan_id", "training", "evaluation", "task_counts", "planned")}, indent=2))
        return
    path = root / "runner.json"
    previous = read_json(path) if path.exists() else {}
    parallel = shot._parallel_settings(args, previous.get("parallel"), default_workers=4)
    with shot.run_lock(root / "runner.lock") as guard:
        path.write_text(json.dumps({"plan_id": plan["plan_id"], "parallel": asdict(parallel)}, indent=2) + "\n", encoding="utf8")
        (root / "stop.request").unlink(missing_ok=True)
        if args.command == "run":
            return _work(root, parallel)
        if sys.platform != "linux":
            raise ValueError("Background start requires Linux; use run for foreground validation.")
        with (root / "console.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                "_background", "--root", str(root), "--lock-fd", str(guard.fileno())],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(guard.fileno(),))
    print(json.dumps({"pid": child.pid, "log": str(root / "console.log"), "root": str(root)}))


if __name__ == "__main__":
    main()
