import argparse
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qaoa_study import experiments as exp
from qaoa_study.parallel import ParallelSettings
from qaoa_study.records import execution_failed, read_json, save_library, write_once_json

NUMERICAL_FILES = ("qaoa_study/qaoa.py", "qaoa_study/optimize.py")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _verify_inputs(data):
    if exp.digest({k: v for k, v in data.items() if k != "input_id"}) != data["input_id"]:
        raise ValueError("Reference inputs changed; export a new immutable input version.")
    graphs = {r["iso_class_id"]: r for r in data["library"]["graphs"]}
    ids = data["evaluation_graph_ids"]
    if ids != sorted(set(ids)) or not ids or any(graphs[g]["graph_split"] != "evaluation" for g in ids):
        raise ValueError("Reference scope must contain unique original evaluation graphs.")
    if len(data["old_references"]) != len(ids) or {r["iso_class_id"] for r in data["old_references"]} != set(ids):
        raise ValueError("Old reference coverage differs from evaluation scope.")
    for ref in data["old_references"]:
        if ref["p"] != 3 or not ref["complete"] or exp.digest({k:v for k,v in ref.items() if k != "reference_id"}) != ref["reference_id"]:
            raise ValueError("Invalid old frozen reference.")
    if set(data["numerical_source_sha256"]) != set(NUMERICAL_FILES):
        raise ValueError("Parent numerical source binding is incomplete.")
    for name, expected in data["numerical_source_sha256"].items():
        if _sha((REPO / name).read_bytes()) != expected:
            raise ValueError(f"Parent numerical source changed: {name}")


def _verify_protocol(config, inputs):
    settings = config["reference_search"]
    prefixes = settings["prefixes"]
    count = exp._reference_restarts(config, 3)
    if (len(prefixes) < 2 or any(type(n) is not int or n < 1 for n in prefixes)
            or prefixes != sorted(set(prefixes)) or prefixes[-1] != count or count % 2):
        raise ValueError("Reference prefixes must increase to the even declared new-pool count.")
    if any(len(r["planned_task_ids"]) != settings["old_reference_starts"] for r in inputs["old_references"]):
        raise ValueError("Original reference size differs from the declared protocol.")
    if "reference_start_design" in config and prefixes != [count // 2, count]:
        raise ValueError("Sobol reference prefixes must compare the two balanced replicas.")


def export_inputs(archive_path, output, diagnostic=None):

    import pyarrow.parquet as pq

    if Path(output).exists():
        raise FileExistsError("Choose a new input file.")
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [n.removesuffix("library/manifest.json") for n in archive.namelist()
                      if n.endswith("/library/manifest.json")]
        if len(candidates) != 1:
            raise ValueError("Expected one completed study library in this archive.")
        prefix = candidates[0]

        def raw(name):
            payload = archive.read(prefix + name)
            hashes[prefix + name] = _sha(payload)
            return payload

        manifest = json.loads(raw("library/manifest.json"))
        payloads = {name: raw("library/"+name) for name in manifest["files"]}
        if any(_sha(payloads[name]) != sha for name, sha in manifest["files"].items()):
            raise ValueError("Completed parent library payload checksum mismatch.")
        graphs = pq.read_table(io.BytesIO(payloads["graphs.parquet"])).to_pylist()
        for graph in graphs:
            for key in ("provenance", "iso_mappings", "feature_missing"):
                graph[key] = json.loads(graph.pop(key+"_json"))
        splits = json.loads(payloads["splits.json"])
        if any(set(ids) != {g["iso_class_id"] for g in graphs if g["graph_split"] == split} for split, ids in splits.items()):
            raise ValueError("Completed library split mismatch.")
        library = {**manifest["library_metadata"], "graphs": graphs,
                   "config": json.loads(payloads["config.json"]),
                   "attempts": [json.loads(line) for line in payloads["generation.jsonl"].splitlines()]}
        baseline = json.loads(raw("baseline/manifest.json"))
        identity = {key: baseline[key] for key in ("library_id", "config", "source_id", "tasks_sha256")}
        if (exp.digest(identity) != baseline["batch_id"]
                or exp.digest(baseline["source"]["files"]) != baseline["source_id"]
                or baseline["source"]["source_id"] != baseline["source_id"]):
            raise ValueError("Parent batch or source identity mismatch.")
        if (baseline["library_id"] != library["library_id"]
                or baseline["library_manifest_sha256"] != hashes[prefix+"library/manifest.json"]):
            raise ValueError("Parent baseline is not bound to this completed library.")
        task_bytes = raw("baseline/tasks.jsonl")
        if _sha(task_bytes) != baseline["tasks_sha256"]:
            raise ValueError("Parent task checksum mismatch.")
        tasks = [json.loads(line) for line in task_bytes.splitlines()]
        if any(exp.digest({k:v for k,v in t.items() if k != "task_id"}) != t["task_id"] for t in tasks):
            raise ValueError("Parent task identity mismatch.")
        refs_manifest = json.loads(raw("references/manifest.json"))
        if exp.digest(refs_manifest["files"]) != refs_manifest["reference_set_id"]:
            raise ValueError("Old reference-set identity mismatch.")
        ids = sorted(g["iso_class_id"] for g in graphs if g["graph_split"] == "evaluation")
        references = []
        keys = set()
        for name in refs_manifest["files"]:
            ref = json.loads(raw("references/"+name))
            key = ref["iso_class_id"], ref["p"]
            if (name != ref["reference_id"] + ".json" or key in keys
                    or exp.digest({k:v for k,v in ref.items() if k != "reference_id"}) != ref["reference_id"]
                    or ref["library_id"] != library["library_id"] or ref["batch_id"] != baseline["batch_id"]):
                raise ValueError("Old reference content identity or parent binding mismatch.")
            keys.add(key)
            references.append(ref)
        references = [r for r in references if r["iso_class_id"] in ids and r["p"] == 3]
    data = {"version": "p3-independent-reference-inputs-v1", "library": library,
            "evaluation_graph_ids": ids, "old_references": references,
            "old_task_seeds": sorted({t["seed"] for t in tasks}),
            "parent_optimizer": baseline["config"]["optimizer"]["3"],
            "numerical_source_sha256": {name: baseline["source"]["files"][name] for name in NUMERICAL_FILES},
            "parent": {"archive_name": Path(archive_path).name, "library_manifest_sha256": hashes[prefix+"library/manifest.json"],
                       "batch_id": baseline["batch_id"], "source_id": baseline["source_id"],
                       "reference_set_id": refs_manifest["reference_set_id"], "member_sha256": hashes}}
    if diagnostic:
        payload = Path(diagnostic).read_bytes()
        details = json.loads(payload)["graph_details"]
        data["diagnostic_only"] = {"source_sha256": _sha(payload),
            "attained_floor": {r["graph"]: r["achieved_floor"] for r in details},
            "use": "Flag known reference deficits only; never initialize or contribute to the reference maximum."}
    data["input_id"] = exp.digest(data)
    _verify_inputs(data)
    write_once_json(Path(output), data)
    return {"input_id": data["input_id"], "library_graphs": len(graphs), "evaluation_graphs": len(ids), "old_references": len(references)}


def export_followup_inputs(archive_path, graph_ids_path, output):





    if Path(output).exists():
        raise FileExistsError("Choose a new follow-up input file.")
    ids = read_json(graph_ids_path)
    if (not isinstance(ids, list) or not ids or any(not isinstance(gid, str) for gid in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Graph IDs must be a sorted, distinct, nonempty JSON list.")
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        roots = [name.removesuffix("inputs.json") for name in archive.namelist() if name.endswith("/inputs.json")]
        if len(roots) != 1:
            raise ValueError("Expected one previous reference-search input file.")
        root = roots[0]

        def raw(name):
            payload = archive.read(name)
            hashes[name] = _sha(payload)
            return payload

        parent = json.loads(raw(root + "inputs.json"))
        _verify_inputs(parent)
        if not set(ids) <= set(parent["evaluation_graph_ids"]):
            raise ValueError("Follow-up graphs must belong to the previous evaluation scope.")
        manifest = json.loads(raw(root + "batch/manifest.json"))
        identity = {key: manifest[key] for key in ("library_id", "config", "source_id", "tasks_sha256")}
        task_bytes = raw(root + "batch/tasks.jsonl")
        if _sha(task_bytes) != manifest["tasks_sha256"]:
            raise ValueError("Previous reference batch or task identity mismatch.")
        tasks = [json.loads(line) for line in task_bytes.splitlines()]
        if (exp.digest(identity) != manifest["batch_id"]
                or exp.digest(manifest["source"]["files"]) != manifest["source_id"]
                or manifest["source"]["source_id"] != manifest["source_id"]
                or manifest["library_id"] != parent["library"]["library_id"]
                or tasks != exp.build_tasks(parent["library"], manifest["config"])):
            raise ValueError("Previous reference batch or task identity mismatch.")
        for name, digest in manifest["source"]["files"].items():
            if _sha(raw(name)) != digest:
                raise ValueError(f"Archived source hash mismatch: {name}")
        finals = [name.removesuffix("union-references/manifest.json") for name in archive.namelist()
                  if name.startswith(root) and name.endswith("/union-references/manifest.json")]
        if len(finals) != 1:
            raise ValueError("Expected one finalized union reference set.")
        final = finals[0]

        def reference_set(folder):
            index = json.loads(raw(final + folder + "/manifest.json"))
            if exp.digest(index["files"]) != index["reference_set_id"]:
                raise ValueError("Previous reference-set identity mismatch.")
            refs = {}
            for name in index["files"]:
                ref = json.loads(raw(final + folder + "/" + name))
                gid = ref["iso_class_id"]
                if (name != ref["reference_id"] + ".json" or gid in refs or ref["p"] != 3
                        or not ref["complete"] or ref["library_id"] != manifest["library_id"]
                        or ref["batch_id"] != manifest["batch_id"]
                        or exp.digest({k: v for k, v in ref.items() if k != "reference_id"}) != ref["reference_id"]):
                    raise ValueError("Previous reference content or source binding mismatch.")
                refs[gid] = ref
            if set(refs) != set(parent["evaluation_graph_ids"]):
                raise ValueError("Previous reference coverage mismatch.")
            return index, refs

        _, new = reference_set("new-references")
        union_index, union = reference_set("union-references")
        old = {r["iso_class_id"]: r for r in parent["old_references"]}
        for gid, ref in union.items():
            planned = sorted(t["task_id"] for t in tasks if t["iso_class_id"] == gid)
            winner = old[gid] if old[gid]["C_ref"] >= new[gid]["C_ref"] else new[gid]
            if (new[gid]["planned_task_ids"] != planned or new[gid]["valid_task_ids"] != planned
                    or ref["planned_task_ids"] != sorted(old[gid]["planned_task_ids"] + planned)
                    or ref["valid_task_ids"] != ref["planned_task_ids"]
                    or ref["parent_reference_ids"] != [old[gid]["reference_id"], new[gid]["reference_id"]]
                    or ref["C_ref"] != winner["C_ref"] or ref["source"] != winner["source"]):
                raise ValueError("Previous union does not match its complete parent pools.")
    with Path(archive_path).open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    data = deepcopy(parent)
    data.pop("input_id")
    data.update(version="p3-followup-reference-inputs-v1", evaluation_graph_ids=ids,
                old_references=[union[gid] for gid in ids],
                old_task_seeds=sorted(set(parent["old_task_seeds"]) | {t["seed"] for t in tasks}),
                parent={"archive_name": Path(archive_path).name,
                        "archive_sha256": archive_sha256,
                        "input_id": parent["input_id"], "batch_id": manifest["batch_id"],
                        "source_id": manifest["source_id"], "reference_set_id": union_index["reference_set_id"],
                        "member_sha256": hashes})
    if "diagnostic_only" in data:
        data["diagnostic_only"]["attained_floor"] = {gid: parent["diagnostic_only"]["attained_floor"][gid] for gid in ids}
    data["input_id"] = exp.digest(data)
    _verify_inputs(data)
    write_once_json(Path(output), data)
    return {"input_id": data["input_id"], "graphs": len(ids), "library_graphs": len(data["library"]["graphs"]),
            "parent_reference_set_id": union_index["reference_set_id"], "prior_seeds": len(data["old_task_seeds"])}


def prepare(inputs_path, config_path, root):

    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Reference study roots are create-only; choose a new root.")
    inputs = read_json(inputs_path)
    _verify_inputs(inputs)
    config = read_json(config_path)
    if config["depths"] != [3] or config["optimizer"]["3"] != inputs["parent_optimizer"]:
        raise ValueError("p3 depth and optimizer budget must match the original study.")
    _verify_protocol(config, inputs)
    config["reference_graph_ids"] = inputs["evaluation_graph_ids"]
    tasks = exp.build_tasks(inputs["library"], config)
    if any(t["experiment_role"] != "reference" for t in tasks):
        raise ValueError("Reference-only task scope was not enforced.")
    overlap = set(inputs["old_task_seeds"]) & {t["seed"] for t in tasks}
    if overlap:
        raise ValueError("New and original random streams overlap.")
    root.mkdir(parents=True)
    write_once_json(root / "inputs.json", inputs)
    save_library(root / "library", inputs["library"])
    manifest = exp.save_batch(root / "batch", root / "library", config)
    if manifest["issues"]:
        raise ValueError(f"Incomplete reference inputs: {manifest['issues']}")
    result = {"input_id": inputs["input_id"], "batch_id": manifest["batch_id"], "source_id": manifest["source_id"],
              "config_sha256": _sha(Path(config_path).read_bytes()), "tasks": len(tasks),
              "graphs": len(config["reference_graph_ids"]), "old_seed_overlap": 0,
              "scope": "New reference pool only; original library roles and references are retained unchanged."}
    write_once_json(root / "preparation.json", result)
    return result


def _read_root(root):
    root = Path(root).resolve()
    inputs, prepared = read_json(root/"inputs.json"), read_json(root/"preparation.json")
    _verify_inputs(inputs)
    manifest, library, tasks = exp.read_batch(root / "batch")
    if (prepared["input_id"] != inputs["input_id"] or prepared["batch_id"] != manifest["batch_id"]
            or library != inputs["library"] or manifest["config"]["reference_graph_ids"] != inputs["evaluation_graph_ids"]):
        raise ValueError("Prepared reference study no longer matches its original inputs.")
    return inputs, manifest, library, tasks


def _observations(root):
    inputs, manifest, library, tasks = _read_root(root)
    attempts = exp.load_attempts(manifest, tasks, Path(root)/"attempts", metadata_only=True)[0] if (Path(root)/"attempts/manifest.json").exists() else []
    return inputs, manifest, library, tasks, attempts, exp.select_attempts(attempts)


def status(root):
    _, manifest, _, tasks, attempts, selected = _observations(root)
    valid = {key: row for key, row in selected.items() if not execution_failed(row)}
    by_graph = defaultdict(int)
    for task in tasks:
        by_graph[task["iso_class_id"]] += int(task["task_id"] in valid)
    prefixes = manifest["config"]["reference_search"]["prefixes"]
    return {"batch_id": manifest["batch_id"], "planned": len(tasks), "valid": len(valid),
            "attempts": len(attempts), "missing": len(tasks)-len(selected),
            "unresolved_execution_faults": len(selected)-len(valid), "valid_by_graph": dict(by_graph),
            "complete_graphs_by_prefix": {str(n): sum(all(t["task_id"] in valid for t in tasks if t["iso_class_id"] == g and t["restart_id"] < n) for g in by_graph) for n in prefixes},
            "complete": len(valid) == len(tasks), "stop_requested": (Path(root)/"stop.request").exists()}


def run(root, *, backend="default.qubit", workers=4, execute=False, max_tasks=None):
    _read_root(root)
    root = Path(root).resolve()
    stopped = [False]
    previous = {}
    last_progress = [None]

    def progress(report):
        values = {key: report.get(key, 0) for key in ("executed", "valid_completed", "missing_tasks", "unresolved_execution_faults")}
        if values != last_progress[0]:
            print(json.dumps(values), file=sys.stderr, flush=True)
            last_progress[0] = values

    with exp._writer_lock(root) if execute else nullcontext():
        if execute:
            (root/"stop.request").unlink(missing_ok=True)
            for number in (signal.SIGINT, signal.SIGTERM):
                previous[number] = signal.signal(number, lambda *_: stopped.__setitem__(0, True))
        try:
            return exp.run_worker(root/"batch", root/"attempts", backend=backend, execute=execute,
                roles=["reference"], parallel=ParallelSettings(workers=workers), max_tasks=max_tasks,
                should_stop=lambda: stopped[0] or (root/"stop.request").exists(), progress=progress)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def summarize_references(inputs, manifest, library, tasks, selected):

    new = exp.freeze_references(manifest, tasks, selected)
    if not new or not all(ref["complete"] for ref in new.values()):
        raise ValueError("All declared reference tasks must be execution-valid before final freezing.")
    old = {(r["iso_class_id"], r["p"]): r for r in inputs["old_references"]}
    stars = {r["iso_class_id"]: r["C_star"] for r in library["graphs"]}
    settings = manifest["config"]["reference_search"]
    _verify_protocol(manifest["config"], inputs)
    count = exp._reference_restarts(manifest["config"], 3)
    union, rows = {}, []
    for key, ref in sorted(new.items()):
        previous = old[key]
        scores = {t["restart_id"]: selected[t["task_id"]]["best_seen"] for t in tasks if (t["iso_class_id"], t["p"]) == key}
        prefixes = {str(n): max(scores[i] for i in range(n)) for n in settings["prefixes"]}
        halves = [max(scores[i] for i in range(start, start+count//2)) for start in (0, count//2)]
        winner = previous if previous["C_ref"] >= ref["C_ref"] else ref
        combined = {**ref, "candidate_rule": "max_of_old_and_new_independent_reference_pools",
            "planned_task_ids": sorted(previous["planned_task_ids"] + ref["planned_task_ids"]),
            "valid_task_ids": sorted(previous["valid_task_ids"] + ref["valid_task_ids"]),
            "selected_attempt_ids": previous["selected_attempt_ids"] + ref["selected_attempt_ids"],
            "C_ref": winner["C_ref"], "source": deepcopy(winner["source"]),
            "parent_reference_ids": [previous["reference_id"], ref["reference_id"]],
            "source_batch_ids": list(dict.fromkeys(previous.get("source_batch_ids", [previous["batch_id"]]) + [ref["batch_id"]])),
            "winning_reference_id": winner["reference_id"]}
        combined.pop("reference_id")
        combined["reference_id"] = exp.digest(combined)
        union[key] = combined
        scale = stars[key[0]]
        half_gap = abs(halves[0]-halves[1])/scale
        prefix_gain = (prefixes[str(count)]-prefixes[str(settings["prefixes"][-2])])/scale
        floor = inputs.get("diagnostic_only", {}).get("attained_floor", {}).get(key[0])
        rows.append({"iso_class_id": key[0], "p": 3, "C_star": scale, "old_reference": previous["C_ref"],
            "new_reference": ref["C_ref"], "union_reference": combined["C_ref"], "prefix_best": prefixes,
            "prefix_gain_over_old_normalized": {n: (value-previous["C_ref"])/scale for n,value in prefixes.items()},
            "union_gain_over_old_normalized": (combined["C_ref"]-previous["C_ref"])/scale,
            "independent_half_best": halves, "half_gap_normalized": half_gap,
            "last_prefix_gain_normalized": prefix_gain,
            "stability_flag": max(half_gap, prefix_gain) > settings["stability_gap_tolerance"],
            "known_attained_floor_diagnostic_only": floor,
            "known_floor_gap_normalized": None if floor is None else (floor-combined["C_ref"])/scale,
            "known_floor_deficit_flag": floor is not None and floor-combined["C_ref"] > settings["diagnostic_gap_tolerance"]*scale})
        if "reference_start_design" in manifest["config"]:
            rows[-1]["component_best"] = {
                component: max(scores[t["restart_id"]] for t in tasks
                               if t["iso_class_id"] == key[0] and t["start_design"]["component"] == component)
                for component in manifest["config"]["reference_start_design"]["components"]}
            rows[-1]["halves_are_balanced_independent_replicas"] = True
    report = {"batch_id": manifest["batch_id"], "input_id": inputs["input_id"], "graphs": len(rows),
        "new_starts_per_graph": count, "old_starts_per_graph": settings["old_reference_starts"], "rows": rows,
        "scope": "Reference stability diagnostics, not a global-optimality certificate. Evaluation scores never enter either frozen reference set.",
        "old_references_modified": False, "automatic_budget_extension": False}
    return new, union, report


def summarize(root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Reference summaries are create-only.")
    inputs, manifest, library, tasks, attempts, selected = _observations(root)
    new, union, report = summarize_references(inputs, manifest, library, tasks, selected)
    report["all_attempt_costs"] = exp.attempt_cost_totals(attempts)
    report["analysis_source"] = exp.source_identity()
    output.mkdir(parents=True)
    exp.save_references(output/"new-references", new)
    exp.save_references(output/"union-references", union)
    write_once_json(output/"summary.json", report)
    return {"graphs": len(new), "new_reference_set": "new-references", "old_plus_new_set": "union-references",
            "stability_flags": sum(r["stability_flag"] for r in report["rows"]),
            "known_floor_deficits": sum(r["known_floor_deficit_flag"] for r in report["rows"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or inspect an independent depth-three QAOA reference search.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-inputs")
    export.add_argument("--archive", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--diagnostic")
    followup = commands.add_parser("export-followup-inputs")
    followup.add_argument("--archive", required=True)
    followup.add_argument("--graph-ids", required=True)
    followup.add_argument("--output", required=True)
    planning = commands.add_parser("prepare")
    planning.add_argument("--inputs", required=True)
    planning.add_argument("--config", required=True)
    planning.add_argument("--root", required=True)
    for name in ("status", "run", "stop", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        if name == "summarize":
            command.add_argument("--output", required=True)
        if name == "run":
            command.add_argument("--backend", choices=("default.qubit", "lightning.gpu"), default="default.qubit")
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--max-tasks", type=int)
            command.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export-inputs":
            result = export_inputs(args.archive, args.output, args.diagnostic)
        elif args.command == "export-followup-inputs":
            result = export_followup_inputs(args.archive, args.graph_ids, args.output)
        elif args.command == "prepare":
            result = prepare(args.inputs, args.config, args.root)
        elif args.command == "status":
            result = status(args.root)
        elif args.command == "run":
            result = run(args.root, backend=args.backend, workers=args.workers, execute=args.execute, max_tasks=args.max_tasks)
        elif args.command == "stop":
            _read_root(args.root)
            (Path(args.root)/"stop.request").touch()
            result = {"stop_requested": True, "scope": "Drain admitted attempts; do not terminate workers."}
        else:
            result = summarize(args.root, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("issues") or result.get("unresolved_execution_faults") else 0
    except (OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
