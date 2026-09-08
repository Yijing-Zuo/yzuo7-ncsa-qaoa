"""Manually compare backend values/gradients and time complete QAOA restarts.

Without --execute this entry prints a plan only. It never installs dependencies.
The fixed-parameter comparison and the full restart are separate measurements.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import networkx as nx
import numpy as np
import pennylane as qml

from qaoa_study.optimize import OptimizerSettings, optimize_qaoa, random_angles
from qaoa_study.experiments import environment, source_identity
from qaoa_study.qaoa import make_qaoa, p1_expectation, qaoa_loss_and_gradient
from qaoa_study.records import execution_failed, read_json, save_run, write_once_json


def _memory_sample(gpu=False):
    """Use measured process high-water RSS; optional device-wide point samples.

    ru_maxrss is a process-lifetime high-water mark, not a per-case delta.
    nvidia-smi values include other processes and do not measure a peak.
    Unsupported measurements are null with a reason, never zero.
    """
    result = {"process_high_water_rss_bytes": None, "rss_scope": "process_lifetime",
              "rss_reason": None, "gpu_device_samples": None,
              "gpu_scope": "device_wide_point_sample_not_process_peak"}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_high_water_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        result["rss_reason"] = "resource.getrusage is unavailable on this platform"
    if gpu:
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            result["gpu_device_samples"] = query.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            result["gpu_reason"] = f"{type(error).__name__}: {error}"
    else:
        result["gpu_reason"] = "sampling not requested"
    return result


def _joint(circuit, theta):
    started = perf_counter()
    with qml.Tracker(circuit.device) as tracker:
        loss, gradient = qaoa_loss_and_gradient(theta, circuit)
    return {"C": -loss, "gradient_C": (-gradient).tolist(),
            "seconds": perf_counter() - started, "tracker_totals": dict(tracker.totals)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--backend", choices=("default.qubit", "lightning.gpu"), default="lightning.gpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--depths", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--seed", type=int, default=45191)
    parser.add_argument("--config", default="configs/optimize-development.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempts-dir", type=Path,
                        help="Fresh complete-attempt directory; defaults beneath data/manual/backend.")
    parser.add_argument("--sample-gpu-memory", action="store_true")
    parser.add_argument("--cpu-restart", action="store_true",
                        help="Also time a full CPU restart when verifying GPU; fixed CPU comparisons always run.")
    args = parser.parse_args(argv)
    if any(n < 4 or n % 2 for n in args.sizes):
        parser.error("sizes must be even and at least 4 for the fixed 3-regular verification graph")
    config = read_json(args.config)
    # Development uses one settings object; experiment configs key it by depth.
    if "optimizer" not in config:
        parser.error("configuration must contain optimizer settings")
    configured = config["optimizer"]
    by_depth = {p: OptimizerSettings(**(configured[str(p)] if str(p) in configured else configured))
                for p in args.depths}
    plan = {"execute": args.execute, "backends": list(dict.fromkeys(["default.qubit", args.backend])),
            "restart_backends": list(dict.fromkeys([args.backend] + (["default.qubit"] if args.cpu_restart else []))),
            "sizes": args.sizes, "depths": args.depths, "graph": "3-regular, graph seed = seed+n",
            "seed": args.seed, "settings": {str(p): asdict(value) for p, value in by_depth.items()},
            "fixed_parameters": {"1": [0.37, 0.21], "2": [0.37, -0.81, 0.21, 0.46]},
            "score_atol": 1e-9, "gradient_atol": 1e-6, "gradient_rtol": 1e-5}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output JSON path")
    if args.output.exists():
        raise FileExistsError(args.output)
    attempt_dir = args.attempts_dir or Path("data/manual/backend") / args.output.stem
    if attempt_dir.exists():
        raise FileExistsError(attempt_dir)
    attempt_dir.mkdir(parents=True)
    verification_id = uuid4().hex
    report = {"verification_version": 2, "verification_id": verification_id,
              "created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
              "environment": environment(args.backend), "source": source_identity(),
              "attempts_directory": str(attempt_dir),
              "scope": "Only requested sizes/depths; no GPU, resource or pilot claim for unmeasured cases.",
              "cases": [], "status": "completed"}
    failed = False
    try:
        for n in args.sizes:
            graph_started = perf_counter()
            graph = nx.random_regular_graph(3, n, seed=args.seed + n)
            graph_seconds = perf_counter() - graph_started
            for p in args.depths:
                seed = int(np.random.SeedSequence([args.seed, n, p]).generate_state(1)[0])
                theta0 = random_angles(p, seed)
                case = {"n": n, "p": p, "seed": seed, "theta0": theta0.tolist(),
                        "edges": [list(edge) for edge in graph.edges],
                        "graph_preparation_seconds": graph_seconds, "backends": {}}
                report["cases"].append(case)
                for backend in plan["backends"]:
                    entry = {"memory_before": _memory_sample(args.sample_gpu_memory)}
                    case["backends"][backend] = entry
                    started = perf_counter()
                    circuit = make_qaoa(graph, p, backend=backend)
                    entry["construction_seconds"] = perf_counter() - started
                    theta_fixed = plan["fixed_parameters"][str(p)]
                    started = perf_counter()
                    with qml.Tracker(circuit.device) as preparation_tracker:
                        prepared, _ = qml.workflow.construct_batch(circuit, level="device")(
                            qml.numpy.array(theta_fixed, dtype=np.float64, requires_grad=True))
                    entry["preprocessing_probe"] = {
                        "seconds": perf_counter() - started, "tape_count": len(prepared),
                        "tracker_totals": dict(preparation_tracker.totals),
                        "scope": "separate construction probe; runtime still applies its own preprocessing",
                    }
                    # First execution includes lazy state preparation; second is warm.
                    entry["joint_cold"] = _joint(circuit, theta_fixed)
                    entry["joint_warm"] = _joint(circuit, theta_fixed)
                    if p == 1:
                        error = abs(entry["joint_warm"]["C"] - p1_expectation(graph, *theta_fixed))
                        entry["p1_analytic_comparison"] = {"score_error": error, "passed": error <= plan["score_atol"]}
                        failed |= error > plan["score_atol"]
                    entry["restart_status"] = "not_requested"
                    if backend not in plan["restart_backends"]:
                        entry["memory_after"] = _memory_sample(args.sample_gpu_memory)
                        continue
                    started = perf_counter()
                    result = optimize_qaoa(graph, p, theta0, by_depth[p], circuit=circuit, backend=backend)
                    entry["complete_restart_seconds"] = perf_counter() - started
                    result.update(attempt_id=uuid4().hex, verification_id=verification_id,
                                  seed=seed, experiment_role="backend_validation")
                    attempt = attempt_dir / f"n{n}-p{p}-{backend}-{result['attempt_id']}.json"
                    started = perf_counter()
                    save_run(attempt, result)
                    entry["attempt_io_seconds"] = perf_counter() - started
                    entry.update(attempt_file=attempt.name, stop_reason=result["stop_reason"], counts=result["counts"])
                    entry["restart_status"] = "execution_fault" if execution_failed(result) else "completed"
                    failed |= execution_failed(result)
                    if result["theta_final"] is not None:
                        endpoint = _joint(circuit, result["theta_final"])
                        entry["endpoint_verification"] = endpoint
                        entry["endpoint_error"] = abs(endpoint["C"] - result["C_final"])
                        failed |= entry["endpoint_error"] > plan["score_atol"]
                    entry["memory_after"] = _memory_sample(args.sample_gpu_memory)
                if args.backend != "default.qubit":
                    cpu, gpu = (case["backends"][name]["joint_warm"] for name in plan["backends"])
                    score_error = abs(cpu["C"] - gpu["C"])
                    gradient_error = float(np.max(np.abs(np.subtract(cpu["gradient_C"], gpu["gradient_C"]))))
                    okay = score_error <= plan["score_atol"] and np.allclose(
                        cpu["gradient_C"], gpu["gradient_C"], atol=plan["gradient_atol"], rtol=plan["gradient_rtol"])
                    case["comparison"] = {"score_error": score_error, "gradient_max_error": gradient_error,
                                          "passed": bool(okay), "trajectories_required_equal": False}
                    failed |= not okay
    except Exception as error:
        failed = True
        report["error"] = f"{type(error).__name__}: {error}"
    report["status"] = "failed" if failed else "completed"
    write_once_json(args.output, report)
    print(f"{args.output}: {report['status']}; complete attempts in {attempt_dir}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
