"""Read-only Plan A labels and diagnostics from selected immutable attempts.

No graph generation, optimization, circuit evaluation, or file access occurs
here. A logical task contributes at most its already selected attempt. Defaults
are versioned implementation choices, to be confirmed by the manual pilot.
"""

from collections import defaultdict
import math

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .records import evaluate_trace, execution_failed


ANALYSIS_VERSION = "plan-a-analysis-v2"
DEFAULT_ANALYSIS_SETTINGS = {
    "epsilon": 0.5,
    "evaluation_restarts": 50,
    "cluster_threshold": 0.05,
    "cluster_sensitivity": [0.025, 0.1],
    "energy_bins": 20,
}


def _finite(value):
    return value is not None and math.isfinite(value)


def wilson_interval(successes: int, planned: int) -> list | None:
    """Two-sided 95% Wilson score interval, explicitly using the supplied n."""
    if not 0 <= successes <= planned:
        raise ValueError("Expected 0 <= successes <= denominator.")
    if not planned:
        return None
    z = 1.959963984540054
    rate = successes / planned
    denominator = 1 + z * z / planned
    center = (rate + z * z / (2 * planned)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / planned + z * z / (4 * planned**2)) / denominator
    return [0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == planned else min(1.0, center + half)]


def evaluate_restart(task: dict, attempt: dict | None, reference: dict | None,
                     epsilon: float = 0.5) -> dict:
    """Separate terminal success, first finite hit, and execution completeness."""
    row = {key: task.get(key) for key in (
        "task_id", "library_id", "iso_class_id", "p", "experiment_role", "pool",
        "method", "restart_id", "seed", "theta0", "budget", "protocol_version", "graph_split",
    )}
    reference_ready = bool(reference and reference.get("complete") and _finite(reference.get("C_ref")))
    row.update(
        attempt_id=attempt.get("attempt_id") if attempt else None,
        batch_id=attempt.get("batch_id") if attempt else None,
        reference_id=reference.get("reference_id") if reference else None,
        reference_complete=reference_ready,
        C_ref=reference.get("C_ref") if reference_ready else None,
        epsilon=epsilon, C_final=None, theta_final=None, best_seen=None, best_theta=None,
        final_call_id=None, best_call_id=None,
        stop_reason=None, counts=None, elapsed_seconds=None, execution_valid=False,
        cost_status=None, cost_unavailable=None,
        execution_fault=False, missing=attempt is None, first_hit=None, hit=None,
        terminal_success=None, reference_exceeded=None,
    )
    if attempt is None:
        return row
    for name in ("C_final", "theta_final", "best_seen", "best_theta", "final_call_id", "best_call_id",
                 "stop_reason", "counts", "elapsed_seconds", "cost_status", "cost_unavailable"):
        row[name] = attempt.get(name)
    valid = bool(attempt.get("run_completed") and not execution_failed(attempt)
                 and _finite(attempt.get("C_final")))
    row.update(execution_valid=valid, execution_fault=not valid)
    if reference_ready:
        row.update(evaluate_trace(attempt, reference["C_ref"], epsilon))
        # Incomplete/corrupt attempts cannot become terminal successes even if
        # they happen to carry a finite endpoint. Their finite trace is retained.
        row["terminal_success"] = bool(valid and row["terminal_success"])
    return row


def hit_statistics(rows: list[dict]) -> dict:
    """Empirical cumulative hits across all planned logical restarts.

    Missing, failed and never-hit tasks stay in the denominator. There is no
    censoring assumption and no fabricated event at a run's stopping budget.
    Quantiles are inverse empirical CDF values, not interpolated call numbers.
    """
    planned = len(rows)
    hits = sorted(row["first_hit"] for row in rows if row["first_hit"] is not None)
    curve = [{"objective_calls": 0, "hits": 0, "planned": planned, "fraction": 0.0 if planned else None}]
    for call in sorted(set(hits)):
        count = sum(value <= call for value in hits)
        curve.append({"objective_calls": call, "hits": count, "planned": planned,
                      "fraction": count / planned})

    def quantile(fraction):
        required = math.ceil(fraction * planned)
        if not planned or len(hits) < required:
            return {"value": None, "status": "not_reached"}
        return {"value": hits[required - 1], "status": "reached"}

    return {
        "population": "all_planned_logical_restarts", "planned": planned,
        "hits": len(hits), "curve": curve,
        "median": quantile(0.5), "p90": quantile(0.9),
        "conditional_on_hit": {
            "population": "selected_attempts_with_finite_threshold_hit", "count": len(hits),
            "median": hits[math.ceil(0.5 * len(hits)) - 1] if hits else None,
            "p90": hits[math.ceil(0.9 * len(hits)) - 1] if hits else None,
        },
    }


def pool_statistics(rows: list[dict], expected_restarts: int = 50) -> dict:
    """Keep planned denominators; incomplete labels remain explicitly null.

    The provisional rate/interval count known successes over planned tasks.
    With missing outcomes this is a lower-bound bookkeeping summary, not a
    completed binomial inference. Faults must be resolved before label fitting.
    """
    planned = len(rows)
    valid = sum(row["execution_valid"] for row in rows)
    missing = sum(row["missing"] for row in rows)
    faults = sum(row["execution_fault"] for row in rows)
    reference_ready = bool(planned and all(row["reference_complete"] for row in rows))
    successes = sum(row["terminal_success"] is True for row in rows) if reference_ready else None
    complete_execution = bool(planned and valid == planned)
    configured_complete = bool(complete_execution and planned == expected_restarts and reference_ready)
    scientific_complete = bool(configured_complete and planned == 50)
    return {
        "successes": successes, "planned": planned, "valid": valid,
        "execution_faults": faults, "missing": missing, "expected_restarts": expected_restarts,
        "complete_execution": complete_execution, "reference_complete": reference_ready,
        "configured_pool_complete": configured_complete,
        "scientific_label_complete": scientific_complete,
        "success_rate_label": successes / planned if scientific_complete else None,
        "fixed_budget_success_fraction": successes / planned if reference_ready else None,
        "wilson95_planned": wilson_interval(successes, planned) if reference_ready else None,
        "provisional": not scientific_complete,
        "provisional_interpretation": "known_successes/planned; missing outcomes are unresolved",
        "hit_statistics": hit_statistics(rows) if reference_ready else None,
    }


def periodic_distances(angles, p: int, beta_period: float = math.pi) -> np.ndarray:
    """Euclidean geodesic distance on the declared radian product torus."""
    values = np.asarray(angles, dtype=np.float64).reshape((-1, 2 * p))
    periods = np.asarray([2 * math.pi] * p + [beta_period] * p)
    wrapped = np.mod(values, periods)
    delta = np.abs(wrapped[:, None, :] - wrapped[None, :, :])
    return np.sqrt(np.sum(np.minimum(delta, periods - delta)**2, axis=-1))


def _entropy(counts) -> float | None:
    total = sum(counts)
    if not total:
        return None
    return -math.fsum((count / total) * math.log(count / total) for count in counts if count)


def cluster_endpoints(rows: list[dict], p: int, threshold: float = 0.05,
                      beta_period: float = math.pi) -> dict:
    """Complete-linkage endpoint diversity; every cluster diameter <= threshold.

    Rows are sorted by task identity before deterministic tie handling. Cluster
    labels are named by the smallest task ID, and the best cluster contains the
    largest finite endpoint score (ties: smallest task ID). These are observed
    endpoints, not an enumeration of local minima or basins of attraction.
    """
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("Clustering threshold must be finite and positive.")
    endpoints = sorted((row for row in rows if row["execution_valid"]
                        and row.get("theta_final") is not None), key=lambda row: row["task_id"])
    distances = periodic_distances([row["theta_final"] for row in endpoints], p, beta_period)
    labels = fcluster(linkage(squareform(distances, checks=False), method="complete"),
                      t=threshold, criterion="distance") if len(endpoints) > 1 else np.ones(len(endpoints), dtype=int)
    groups = defaultdict(list)
    for label, row in zip(labels, endpoints, strict=True):
        groups[int(label)].append(row)
    clusters = [{"cluster_id": members[0]["task_id"], "count": len(members),
                 "task_ids": [row["task_id"] for row in members],
                 "attempt_ids": [row["attempt_id"] for row in members]}
                for members in groups.values()]
    clusters.sort(key=lambda cluster: cluster["cluster_id"])
    best = min(endpoints, key=lambda row: (-row["C_final"], row["task_id"])) if endpoints else None
    best_cluster = next((cluster for cluster in clusters if best["task_id"] in cluster["task_ids"]), None) if best else None
    return {
        "algorithm": "complete_linkage", "metric": "radian_torus_euclidean",
        "gamma_period": 2 * math.pi, "beta_period": beta_period, "threshold": threshold,
        "planned": len(rows), "valid_endpoints": len(endpoints), "excluded": len(rows) - len(endpoints),
        "cluster_count": len(clusters), "clusters": clusters,
        "shannon_entropy_nats": _entropy([cluster["count"] for cluster in clusters]),
        "best_cluster_id": best_cluster["cluster_id"] if best_cluster else None,
        "best_endpoint_task_id": best["task_id"] if best else None,
        "best_cluster_frequency": best_cluster["count"] / len(endpoints) if best_cluster else None,
    }


def energy_distribution(rows: list[dict], C_star: float | None, bins: int = 20) -> dict:
    """Valid terminal scores with fixed uniform bins on [0,C_star].

    The last bin includes C_star. Out-of-range values are reported, never
    silently clipped or used to redefine the bin edges. Histogram entropy is
    conditional on in-range finite endpoints and differs from cluster entropy.
    """
    if bins < 1 or int(bins) != bins:
        raise ValueError("Expected a positive integer bin count.")
    endpoints = [row for row in rows if row["execution_valid"] and _finite(row["C_final"])]
    values = [row["C_final"] for row in endpoints]
    result = {
        "C_source": "selected_attempt_C_final", "population": "execution_valid_endpoints",
        "values": values, "task_ids": [row["task_id"] for row in endpoints],
        "count": len(values), "mean": float(np.mean(values)) if values else None,
        "population_variance": float(np.var(values)) if values else None,
        "minimum": min(values) if values else None, "maximum": max(values) if values else None,
        "C_star": C_star, "bin_edges": None, "bin_counts": None,
        "shannon_entropy_nats": None, "out_of_range": None,
        "entropy_population": "finite_endpoints_within_fixed_exact_cut_range",
    }
    if _finite(C_star) and C_star > 0:
        edges = np.linspace(0, C_star, bins + 1)
        counts = np.histogram(values, bins=edges)[0].tolist()
        result.update(bin_edges=edges.tolist(), bin_counts=counts,
                      shannon_entropy_nats=_entropy(counts),
                      out_of_range=sum(value < 0 or value > C_star for value in values))
    return result


def gradient_statistics(samples: list[dict], p: int, edge_count: int, population: str) -> dict:
    """Population (ddof=0) score-gradient variance and norms, raw and per edge.

    Uniform samples and optimization trajectory samples must be passed in
    separate calls. The latter are adaptive observations, not uniform draws.
    Grouped attempt/call references identify the raw coordinates and gradients;
    the summary does not duplicate full traces or device-work dictionaries.
    """
    valid = [row for row in samples if row.get("gradient") is not None
             and np.shape(row["gradient"]) == (2 * p,) and np.all(np.isfinite(row["gradient"]))]
    gradients = np.asarray([row["gradient"] for row in valid], dtype=np.float64).reshape((-1, 2 * p))
    sources = {}
    for row in valid:
        key = (row.get("task_id"), row.get("attempt_id"), row.get("seed"))
        if key not in sources:
            sources[key] = {"task_id": key[0], "attempt_id": key[1], "seed": key[2], "call_ids": [],
                            "gradient_source": row.get("gradient_source", "caller_supplied.gradient"),
                            "coordinate_source": row.get("coordinate_source", "caller_supplied.theta")}
        sources[key]["call_ids"].append(row.get("call_id"))

    def summarize(array):
        return {"mean": array.mean(axis=0).tolist(), "population_variance": array.var(axis=0).tolist(),
                "norms": np.linalg.norm(array, axis=1).tolist(),
                "mean_norm": float(np.linalg.norm(array, axis=1).mean())} if len(array) else None

    return {
        "population": population, "gradient_convention": "positive_C", "ddof": 0,
        "sample_count": len(valid), "excluded_samples": len(samples) - len(valid),
        "raw_C": summarize(gradients), "C_per_edge": summarize(gradients / edge_count) if edge_count else None,
        "sample_sources": list(sources.values()),
    }


def _costs(rows):
    """Sum verified measurements only; any unknown makes its total unknown.

    Record-validation failures can retain independently verified counters. A
    null counter or an explicit ``cost_unavailable`` entry never becomes zero;
    known subtotals and the affected attempt identities remain inspectable.
    """
    names = ("objective_calls", "gradient_requests", "device_executions", "device_derivatives", "device_vjps", "iterations")
    metrics = (*names, "elapsed_seconds", "preparation_seconds", "analytic_evaluations", "analytic_points_requested")
    known, unknown = {name: [] for name in metrics}, {name: [] for name in metrics}
    affected = []
    for row in rows:
        unavailable = row.get("cost_unavailable") or {}
        values = {name: (row.get("counts") or {}).get(name) for name in names}
        values.update(elapsed_seconds=row.get("elapsed_seconds"), preparation_seconds=row.get("preparation_seconds"))
        values.update({name: (row.get("grid") or {}).get(name, None if row.get("kind") == "p1_grid" else 0)
                       for name in ("analytic_evaluations", "analytic_points_requested")})
        missing = []
        for name, value in values.items():
            # The partial-validation contract retains finite counters only
            # when independently verified. A wholly unverified legacy wrapper
            # cannot promote its reported counters into known measurements.
            unverified = row.get("cost_status") == "unverified_invalid_result" and name in names
            if not _finite(value) or name in unavailable or unverified:
                unknown[name].append(row.get("attempt_id"))
                missing.append(name)
            else:
                known[name].append(value)
        if missing:
            affected.append({"task_id": row.get("task_id"), "attempt_id": row.get("attempt_id"),
                             "cost_status": row.get("cost_status"), "unknown_metrics": missing,
                             "cost_unavailable": unavailable})
    subtotals = {name: math.fsum(values) if name.endswith("seconds") else sum(values)
                 for name, values in known.items()}
    return {**{name: None if unknown[name] else subtotals[name] for name in metrics},
            "known_subtotals": subtotals,
            "unknown_counts": {name: len(attempts) for name, attempts in unknown.items()},
            "unknown_attempt_ids_by_metric": unknown,
            "unverified_attempt_count": len(affected), "cost_provenance": affected,
            "population": "all_selected_attempts_including_execution_faults",
            "backend_warning": "Tracker counters retain backend-specific semantics; no equivalent-energy conversion"}


def build_summary(library: dict, tasks: list[dict], selected_attempts: dict,
                  references: dict, analysis_settings: dict | None = None) -> dict:
    """Rebuild graph, restart, and diagnostic rows without any computation I/O.

    References are keyed by (iso_class_id,p). The executor chooses attempts by
    its frozen rule before this function; scores never choose attempts here.
    A scientifically complete success label requires the declared 50-run pool,
    a complete frozen reference, and every logical restart execution-valid.
    Only training graphs can supply model-fitting labels.
    """
    settings = {**DEFAULT_ANALYSIS_SETTINGS, **(analysis_settings or {})}
    if settings["epsilon"] != 0.5:
        raise ValueError("The primary Plan A success threshold is fixed at epsilon=0.5.")
    graph_by_id = {row["iso_class_id"]: row for row in library["graphs"]}
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Task identities must be unique; each logical restart counts once.")
    if set(selected_attempts) - set(task_by_id):
        raise ValueError("A selected attempt has no planned task.")
    grouped_tasks = defaultdict(list)
    for task in tasks:
        if task["library_id"] != library["library_id"] or task["iso_class_id"] not in graph_by_id:
            raise ValueError("Task and frozen library identities disagree.")
        if task["graph_split"] != graph_by_id[task["iso_class_id"]]["graph_split"]:
            raise ValueError("Task graph split disagrees with the frozen library.")
        attempt = selected_attempts.get(task["task_id"])
        if attempt and attempt.get("task_id") != task["task_id"]:
            raise ValueError("Selected attempt belongs to another logical task.")
        grouped_tasks[task["iso_class_id"], task["p"]].append(task)
    graph_rows, restart_rows, diagnostics = [], [], []
    for (identity, p), group in sorted(grouped_tasks.items()):
        graph = graph_by_id[identity]
        reference = references.get((identity, p))
        if reference and any(reference.get(key) != expected for key, expected in
                             (("library_id", library["library_id"]), ("iso_class_id", identity), ("p", p))):
            raise ValueError("Reference and graph/depth identities disagree.")
        if reference:
            if "protocol_version" in reference and any(task["protocol_version"] != reference["protocol_version"] for task in group):
                raise ValueError("Reference and task protocol versions disagree.")
            if "planned_task_ids" in reference and set(reference["planned_task_ids"]) != {
                    task["task_id"] for task in group if task["experiment_role"] == "reference"}:
                raise ValueError("Reference comes from a different planned reference pool.")
            if "batch_id" in reference and any(attempt.get("batch_id") != reference["batch_id"]
                                               for task in group if (attempt := selected_attempts.get(task["task_id"]))):
                raise ValueError("Reference and selected attempt batches disagree.")
        rows = [evaluate_restart(task, selected_attempts.get(task["task_id"]), reference, settings["epsilon"])
                for task in group if task["kind"] == "optimization"]
        evaluation = [row for row in rows if row["experiment_role"] == "evaluation"]
        tier1 = [row for row in rows if row["experiment_role"] == "tier1"]
        valid_tier1 = [row for row in tier1 if row["execution_valid"] and _finite(row["best_seen"])]
        best = min(valid_tier1, key=lambda row: (-row["best_seen"], row["task_id"])) if valid_tier1 else None
        C_ref = reference.get("C_ref") if reference and reference.get("complete") else None
        C_star = graph.get("C_star") if graph.get("exact_status") == "computed" else None
        for row in rows:
            row.update(C_star=C_star, C_source="selected_attempt_C_final",
                       ratio_exact=row["C_final"] / C_star if _finite(row["C_final"]) and C_star else None,
                       gap_ref=C_ref - row["C_final"] if _finite(C_ref) and _finite(row["C_final"]) else None)
        restart_rows.extend(rows)
        pool = pool_statistics(evaluation, settings["evaluation_restarts"])
        provenance = {"library_id": library["library_id"], "iso_class_id": identity, "p": p,
                      "reference_id": reference.get("reference_id") if reference else None,
                      "task_ids": sorted(task["task_id"] for task in group),
                      "attempt_ids": [selected_attempts[task["task_id"]]["attempt_id"] for task in group
                                      if task["task_id"] in selected_attempts]}
        graph_rows.append({
            **provenance, "graph_split": graph["graph_split"], "n": graph["n"],
            "C_star": C_star, "exact_status": graph.get("exact_status"), "C_ref": C_ref,
            "evaluation": pool,
            "training_label_eligible": bool(graph["graph_split"] == "training" and pool["scientific_label_complete"]),
            "tier1": {"C_source": "tier1_selected_valid_attempt_trace_best_seen", "planned": len(tier1),
                      "valid": len(valid_tier1), "tier1_best_seen": best["best_seen"] if best else None,
                      "source_task_id": best["task_id"] if best else None,
                      "source_attempt_id": best["attempt_id"] if best else None,
                      "source_call_id": best["best_call_id"] if best else None,
                      "bias_C_ref_minus_tier1_best_seen": C_ref - best["best_seen"] if best and _finite(C_ref) else None,
                      "ratio_exact": best["best_seen"] / C_star if best and C_star else None},
            "reference_ratio_exact": C_ref / C_star if _finite(C_ref) and C_star else None,
            "reference_ratio_C_source": "frozen_reference_C_ref",
            "reference_exceeded_count": sum(row["reference_exceeded"] is True for row in evaluation),
            "costs_by_role": {role: _costs([selected_attempts[task["task_id"]] for task in group
                                           if task["experiment_role"] == role and task["task_id"] in selected_attempts])
                              for role in sorted({task["experiment_role"] for task in group})},
        })
        uniform, trajectory = [], []
        for task in group:
            attempt = selected_attempts.get(task["task_id"])
            if not attempt:
                continue
            origin = {"task_id": task["task_id"], "attempt_id": attempt["attempt_id"], "seed": task["seed"]}
            if task["kind"] == "gradient":
                uniform.extend({**origin, "call_id": sample["call_id"], "gradient": sample.get("gradient"),
                                "gradient_source": "gradient_samples[*].gradient; match call_id (positive C)",
                                "coordinate_source": "gradient_samples[*].theta; match call_id"}
                               for sample in attempt.get("gradient_samples", []))
            elif task["experiment_role"] == "evaluation":
                trajectory.extend({**origin, "call_id": sample["call_id"],
                                   "gradient": [-value for value in sample["gradient"]],
                                   "gradient_source": "-trace[*].gradient; match call_id (stored loss gradient)",
                                   "coordinate_source": "trace[*].theta; match call_id"}
                                  for sample in attempt["trace"] if sample.get("gradient") is not None)
        edge_count = graph.get("m", len(graph.get("edges", [])))
        diagnostics.append({
            **provenance, "population": "evaluation_pool_observed_terminal_solutions",
            "clusters": cluster_endpoints(evaluation, p, settings["cluster_threshold"]),
            "clusters_beta_pi_over_2": cluster_endpoints(
                evaluation, p, settings["cluster_threshold"], beta_period=math.pi / 2),
            "cluster_threshold_sensitivity": [cluster_endpoints(evaluation, p, threshold)
                                              for threshold in settings["cluster_sensitivity"]],
            "energy_distribution": energy_distribution(evaluation, C_star, settings["energy_bins"]),
            "uniform_gradients": gradient_statistics(uniform, p, edge_count, "independent_uniform_periodic_points"),
            "trajectory_gradients": gradient_statistics(trajectory, p, edge_count, "evaluation_optimizer_trials"),
            "gradient_planned_tasks": sum(task["kind"] == "gradient" for task in group),
            "gradient_missing_tasks": sum(task["kind"] == "gradient" and task["task_id"] not in selected_attempts for task in group),
            "gradient_failed_tasks": sum(task["kind"] == "gradient" and task["task_id"] in selected_attempts
                                         and execution_failed(selected_attempts[task["task_id"]]) for task in group),
        })
    return {"analysis_version": ANALYSIS_VERSION, "settings": settings, "library_id": library["library_id"],
            "graph_qaoa": graph_rows, "restarts": restart_rows, "diagnostics": diagnostics}


WARM_ANALYSIS_VERSION = "plan-a-warm-analysis-v1"
DEFAULT_WARM_SETTINGS = {"epsilon": 0.5, "bootstrap_seed": 20260912, "bootstrap_samples": 2000}
_WARM_METRICS = ("C_initial", "C_final", "ratio_initial_exact", "ratio_final_exact",
                 "gap_initial_ref", "gap_final_ref", "terminal_success_fraction", "hit_fraction")


def _warm_restart(task, attempt, reference, graph, epsilon):
    row = evaluate_restart(task, attempt, reference, epsilon)
    row.update({key: task.get(key) for key in ("regime", "fold", "variant", "fit_id")})
    first = next(iter(attempt.get("trace", [])), None) if attempt else None
    if first and (first["call_id"] != 1 or first.get("theta") != task["theta0"]):
        raise ValueError("Warm comparison requires the actual first call at frozen theta0.")
    initial = first["C"] if first and _finite(first.get("C")) else None
    exact = graph.get("C_star") if graph.get("exact_status") == "computed" else None
    row.update(C_initial=initial, initial_call_id=1 if initial is not None else None,
               initial_score_source="trace[0].C at frozen theta0; no new objective evaluation",
               initial_score_available=initial is not None, C_star=exact)
    for name, score in (("initial", initial), ("final", row["C_final"])):
        row[f"ratio_{name}_exact"] = score / exact if _finite(score) and exact else None
        row[f"gap_{name}_ref"] = row["C_ref"] - score if _finite(score) and row["reference_complete"] else None
    return row


def _warm_pool(rows):
    """One graph: average B1 repeats before any across-graph calculation."""
    planned = len(rows)
    valid = sum(row["execution_valid"] for row in rows)
    ready = all(row["reference_complete"] for row in rows)
    complete = valid == planned and ready and all(row["initial_score_available"] for row in rows)
    metrics, observed = {}, {}
    for name in _WARM_METRICS[:6]:
        values = [row[name] for row in rows if row["execution_valid"] and _finite(row[name])]
        mean = math.fsum(values) / len(values) if values else None
        metrics[name] = mean if len(values) == planned else None
        observed[name] = {"mean": mean, "valid_observed": len(values), "planned": planned}
    metrics["terminal_success_fraction"] = sum(row["terminal_success"] is True for row in rows) / planned if ready else None
    metrics["hit_fraction"] = sum(row["first_hit"] is not None for row in rows) / planned if ready else None
    return {"planned": planned, "valid": valid, "missing": sum(row["missing"] for row in rows),
            "execution_faults": sum(row["execution_fault"] for row in rows),
            "reference_complete": ready, "complete": complete, "provisional": not complete,
            "metrics": metrics, "observed_scores": observed,
            "first_hits": [row["first_hit"] for row in rows],
            "task_ids": [row["task_id"] for row in rows], "attempt_ids": [row["attempt_id"] for row in rows],
            "success_interpretation": "known successes/planned; unresolved outcomes retained, not treated as completed failures"}


def _warm_hit_curve(pools):
    """F(k)=mean_graph mean_planned_restart 1[first_hit<=k], with no censoring."""
    if not all(pool["reference_complete"] for pool in pools):
        return None
    events = defaultdict(list)
    for pool in pools:
        for call in pool["first_hits"]:
            if call is not None:
                events[call].append(1 / (len(pools) * pool["planned"]))
    curve = [{"objective_calls": 0, "fraction": 0.0}]
    masses = []
    for call in sorted(events):
        masses.append(math.fsum(events[call]))
        curve.append({"objective_calls": call, "fraction": min(1.0, math.fsum(masses))})
    quantiles = {}
    for name, fraction in (("median", 0.5), ("p90", 0.9)):
        hit = next((point["objective_calls"] for point in curve if point["fraction"] >= fraction - 1e-12), None)
        quantiles[name] = {"value": hit, "status": "reached" if hit is not None else "not_reached"}
    return {"population": "all_planned_graphs_equal_weight; within_graph_all_planned_restarts_equal_weight",
            "graphs": len(pools), "planned_restarts": sum(pool["planned"] for pool in pools),
            "provisional": any(pool["provisional"] for pool in pools), "curve": curve, **quantiles}


def _warm_comparison(rows, settings, *, baseline_method="B1", warm_method="B2", metrics_names=_WARM_METRICS):
    complete = all(row["complete"] for row in rows)
    draws = np.random.default_rng(settings["bootstrap_seed"]).integers(
        len(rows), size=(settings["bootstrap_samples"], len(rows))) if complete and len(rows) > 1 else None
    metrics = {}
    for name in metrics_names:
        baseline = [row[baseline_method]["metrics"][name] for row in rows]
        warm = [row[warm_method]["metrics"][name] for row in rows]
        available = all(_finite(value) for value in baseline + warm)
        delta = np.asarray(warm) - np.asarray(baseline) if available else None
        metrics[name] = {
            baseline_method: math.fsum(baseline) / len(rows) if all(_finite(v) for v in baseline) else None,
            warm_method: math.fsum(warm) / len(rows) if all(_finite(v) for v in warm) else None,
            f"difference_{warm_method}_minus_{baseline_method}": float(delta.mean()) if available else None,
            "paired_bootstrap_ci95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
            if available and draws is not None else None,
            "ci_status": "available" if available and draws is not None else
                         "provisional" if not complete else "fewer_than_two_graphs" if len(rows) < 2 else "metric_unavailable",
        }
    return {"graphs": len(rows), "complete": complete, "provisional": not complete,
            "weighting": f"{baseline_method} mean within each graph, then equal graph weights; {warm_method} one start per graph",
            "difference_interpretation": f"{warm_method} minus {baseline_method}; provisional known-success differences are not bounds",
            "metrics": metrics, "hit_statistics": {
                method: _warm_hit_curve([row[method] for row in rows]) for method in (baseline_method, warm_method)},
            "bootstrap": {"unit": "paired iso_class_id", "seed": settings["bootstrap_seed"],
                          "samples": settings["bootstrap_samples"], "interval": "percentile 95%",
                          "scope": "mean paired differences; no CI when planned execution/reference incomplete"}}


def build_warm_summary(library: dict, tasks: list[dict], selected: dict, references: dict, *,
                       b1_tasks: list[dict], b1_selected: dict, settings: dict | None = None) -> dict:
    """Compare a frozen single-start B2 batch with separately validated B1 traces.

    The caller validates source batches, references, fits and attempt provenance.
    This pure summary enforces task/graph identities, one B2 start per variant,
    and identical optimizer budgets. It never selects attempts by their scores.
    Missing/faulted planned runs remain in denominators; formal paired intervals
    require every planned run and reference. No existing B1 label rule changes.
    """
    settings = {**DEFAULT_WARM_SETTINGS, **(settings or {})}
    if (settings["epsilon"] != 0.5 or type(settings["bootstrap_samples"]) is not int or
            settings["bootstrap_samples"] < 1 or type(settings["bootstrap_seed"]) is not int or settings["bootstrap_seed"] < 0):
        raise ValueError("Warm analysis requires epsilon=0.5 and fixed positive bootstrap count/nonnegative integer seed.")
    graphs = {graph["iso_class_id"]: graph for graph in library["graphs"]}
    warm_groups, baseline_tasks = defaultdict(list), defaultdict(list)
    seen = set()
    for method, planned, attempts in (("B1", b1_tasks, b1_selected), ("B2", tasks, selected)):
        identities = {task["task_id"] for task in planned}
        if len(identities) != len(planned) or set(attempts) - identities or identities & seen:
            raise ValueError("Warm comparison contains duplicate/unplanned task identities.")
        seen.update(identities)
        for task in planned:
            graph = graphs.get(task["iso_class_id"])
            if (not graph or task["library_id"] != library["library_id"] or task["graph_split"] != graph["graph_split"]
                    or task["kind"] != "optimization" or task["method"] != method
                    or task["experiment_role"] != ("evaluation" if method == "B1" else "warm_start")):
                raise ValueError("Warm comparison task, graph, method or role disagrees.")
            if task["task_id"] in attempts and attempts[task["task_id"]]["task_id"] != task["task_id"]:
                raise ValueError("Selected warm comparison attempt belongs to another task.")
            if method == "B1":
                baseline_tasks[task["iso_class_id"], task["p"]].append(task)
            else:
                if (task["regime"] not in ("random", "lofo") or task["variant"] not in ("medoid", "aligned_median")
                        or (task["regime"] == "random") != (task["fold"] is None) or graph["graph_split"] != "evaluation"):
                    raise ValueError("Unsupported warm regime/fold/variant or non-evaluation target graph.")
                key = tuple(task[name] for name in ("regime", "fold", "p", "variant", "fit_id"))
                warm_groups[key].append(task)
    restarts, graph_rows, comparisons, evaluated_b1 = [], [], [], {}
    target_sets = {}
    for key, group in sorted(warm_groups.items(), key=lambda item: repr(item[0])):
        metadata = dict(zip(("regime", "fold", "p", "variant", "fit_id"), key))
        identities = {task["iso_class_id"] for task in group}
        target_key = key[:3] + key[4:]
        if len(identities) != len(group) or target_sets.setdefault(target_key, identities) != identities:
            raise ValueError("Each warm variant must have exactly one task per same target graph/depth.")
        paired = []
        for task in sorted(group, key=lambda row: row["iso_class_id"]):
            identity, p = task["iso_class_id"], task["p"]
            baseline = sorted(baseline_tasks.get((identity, p), []), key=lambda row: row["task_id"])
            if not baseline or any(row["budget"] != task["budget"] for row in baseline):
                raise ValueError("Warm comparison needs a planned B1 pool with exactly the same optimizer budget.")
            reference = references.get((identity, p))
            if reference and any(reference.get(name) != value for name, value in
                                 (("library_id", library["library_id"]), ("iso_class_id", identity), ("p", p))):
                raise ValueError("Warm reference and target graph/depth identities disagree.")
            if (identity, p) not in evaluated_b1:
                evaluated_b1[identity, p] = [_warm_restart(row, b1_selected.get(row["task_id"]), reference,
                                                         graphs[identity], settings["epsilon"]) for row in baseline]
                restarts.extend(evaluated_b1[identity, p])
            warm = _warm_restart(task, selected.get(task["task_id"]), reference, graphs[identity], settings["epsilon"])
            restarts.append(warm)
            b1, b2 = _warm_pool(evaluated_b1[identity, p]), _warm_pool([warm])
            row = {**metadata, "library_id": library["library_id"], "iso_class_id": identity,
                   "graph_split": graphs[identity]["graph_split"], "n": graphs[identity]["n"],
                   "reference_id": reference.get("reference_id") if reference else None,
                   "B1": b1, "B2": b2, "complete": b1["complete"] and b2["complete"],
                   "provisional": not (b1["complete"] and b2["complete"]),
                   "budget": task["budget"]}
            paired.append(row)
            graph_rows.append(row)
        comparisons.append({**metadata, **_warm_comparison(paired, settings)})
    used = {row["task_id"] for row in restarts}
    return {"analysis_version": WARM_ANALYSIS_VERSION, "settings": settings, "library_id": library["library_id"],
            "graph_qaoa": graph_rows, "restarts": restarts, "diagnostics": [], "comparisons": comparisons,
            "complete": bool(comparisons) and all(row["complete"] for row in comparisons),
            "selected_costs_by_method": {method: _costs([attempt for identity, attempt in attempts.items() if identity in used])
                                         for method, attempts in (("B1", b1_selected), ("B2", selected))}}


B3_ANALYSIS_VERSION = "plan-a-b3-comparison-v1"


def _b3_pool(rows, method):
    pool = _warm_pool(rows)
    calls = _costs(rows)["objective_calls"]
    pool["metrics"]["objective_calls"] = calls / len(rows) if calls is not None else None
    if method == "B1":
        pool["terminal_success_wilson95_restarts"] = (
            wilson_interval(sum(row["terminal_success"] is True for row in rows), len(rows))
            if pool["complete"] else None)
    return pool


def _b3_success_intervals(rows, method, settings):
    """Separate graph-mixture uncertainty from nominal binary-graph Wilson intervals."""
    pools = [row[method] for row in rows]
    values = [pool["metrics"]["terminal_success_fraction"] for pool in pools]
    ready = all(_finite(value) for value in values)
    complete = all(row["complete"] for row in rows)
    result = {"planned_graphs": len(rows), "complete": complete,
              "fraction": math.fsum(values) / len(rows) if ready else None}
    if method == "B1":
        draws = np.random.default_rng(settings["bootstrap_seed"]).integers(
            len(rows), size=(settings["bootstrap_samples"], len(rows))) if complete and len(rows) > 1 else None
        result.update(graph_bootstrap_ci95=np.quantile(np.asarray(values)[draws].mean(axis=1), [0.025, 0.975]).tolist()
                      if ready and draws is not None else None,
                      scope="equal graph weights over within-graph random-restart fractions; no pooled binomial interval")
    else:
        successes = sum(int(value) for value in values) if ready else None
        result.update(known_successes=successes,
                      wilson95_nominal=wilson_interval(successes, len(rows)) if ready and complete else None,
                      scope="nominal independent-graph Bernoulli approximation; stratified-design coverage not guaranteed")
    return result


def build_b3_summary(library: dict, tasks: list[dict], selected: dict, references: dict, *,
                     b1_tasks: list[dict], b1_selected: dict, b2_tasks: list[dict],
                     b2_selected: dict, settings: dict | None = None) -> dict:
    """Compare each frozen B3 depth with B1 and both B2 variants in each supplied scope.

    The caller audits batches, fits, source/environment compatibility, reference
    bindings and required case coverage. Inputs contain only the requested depths.
    B3 has one global task per evaluation graph/depth; LOFO views reuse that task.
    This function checks complete planned target sets rather than intersecting them,
    and keeps missing attempts in their planned denominators. No file or device I/O.
    """
    warm = build_warm_summary(library, b2_tasks, b2_selected, references,
                             b1_tasks=b1_tasks, b1_selected=b1_selected, settings=settings)
    settings = warm["settings"]
    graphs = {row["iso_class_id"]: row for row in library["graphs"]
              if row["tier"] == 2 and row["graph_split"] == "evaluation"}
    depths = {task["p"] for task in tasks}
    lookup = {(task["iso_class_id"], task["p"]): task for task in tasks}
    identities = {task["task_id"] for task in tasks}
    if (not depths or not depths <= {1, 2} or len(lookup) != len(tasks) or len(identities) != len(tasks)
            or set(selected) - identities or identities & {row["task_id"] for row in warm["restarts"]}
            or set(lookup) != {(identity, p) for identity in graphs for p in depths}):
        raise ValueError("B3 requires exactly one global task per evaluation graph/depth and distinct planned identities.")
    rules = defaultdict(set)
    b3_rows = []
    for key, task in sorted(lookup.items()):
        reference = references.get(key)
        if (task["library_id"] != library["library_id"] or task["graph_split"] != "evaluation"
                or task["method"] != "B3" or task["kind"] != "optimization" or task["experiment_role"] != "warm_start"
                or task["restart_id"] != 0 or not task.get("rule_id") or not task.get("initialization_id")
                or (reference is not None and task.get("reference_id") != reference.get("reference_id"))
                or (task["task_id"] in selected and selected[task["task_id"]]["task_id"] != task["task_id"])):
            raise ValueError("B3 task, rule, initialization, reference or selected attempt identity disagrees.")
        rules[task["p"]].add(task["rule_id"])
        row = _warm_restart(task, selected.get(task["task_id"]), reference, graphs[key[0]], settings["epsilon"])
        row.update({name: task.get(name) for name in ("rule_id", "initialization_id", "initialization_set_id")})
        b3_rows.append(row)
    if any(len(values) != 1 for values in rules.values()):
        raise ValueError("B3 requires one frozen rule per depth.")

    restart_rows = {row["task_id"]: row for row in (*warm["restarts"], *b3_rows)}
    b1_groups, scopes = defaultdict(list), defaultdict(list)
    for row in warm["restarts"]:
        if row["method"] == "B1":
            b1_groups[row["iso_class_id"], row["p"]].append(row)
    for task in b2_tasks:
        scopes[task["regime"], task["fold"]].append(task)
    if ("random", None) not in scopes:
        raise ValueError("B3 comparison requires the declared random B2 case.")
    b3_pools = {(row["iso_class_id"], row["p"]): _b3_pool([row], "B3") for row in b3_rows}
    b1_pools = {key: _b3_pool(rows, "B1") for key, rows in b1_groups.items()}
    graph_rows, comparisons = [], []
    for (regime, fold), group in sorted(scopes.items(), key=lambda item: repr(item[0])):
        expected = {identity for identity, graph in graphs.items() if regime == "random"
                    or (graph["lofo_eligible"] and graph["families"] == [fold])}
        indexed = {(task["iso_class_id"], task["p"], task["variant"]): task for task in group}
        if (not expected or len({task["fit_id"] for task in group}) != 1 or len(indexed) != len(group)
                or set(indexed) != {(identity, p, variant) for identity in expected for p in depths
                                    for variant in ("medoid", "aligned_median")}):
            raise ValueError("Each B2 case must provide both variants and every declared graph/depth under one fit.")
        for p in sorted(depths):
            for method, variant in (("B1", None), ("B2", "medoid"), ("B2", "aligned_median")):
                metadata = {"regime": regime, "fold": fold, "p": p, "baseline_method": method,
                            "variant": variant, "fit_id": group[0]["fit_id"] if method == "B2" else None,
                            "rule_id": next(iter(rules[p]))}
                paired = []
                for identity in sorted(expected):
                    task = lookup[identity, p]
                    b2 = indexed[identity, p, variant or "medoid"]
                    if b2["budget"] != task["budget"]:
                        raise ValueError("B3 and its B1/B2 comparison need identical optimizer budgets.")
                    baseline = (b1_pools[identity, p] if method == "B1"
                                else _b3_pool([restart_rows[b2["task_id"]]], "B2"))
                    b3 = b3_pools[identity, p]
                    row = {**metadata, "library_id": library["library_id"], "iso_class_id": identity,
                           "graph_split": "evaluation", "n": graphs[identity]["n"],
                           "reference_id": restart_rows[task["task_id"]]["reference_id"],
                           method: baseline, "B3": b3, "budget": task["budget"],
                           "complete": baseline["complete"] and b3["complete"],
                           "provisional": not (baseline["complete"] and b3["complete"])}
                    paired.append(row)
                    graph_rows.append(row)
                comparison = _warm_comparison(paired, settings, baseline_method=method, warm_method="B3",
                                               metrics_names=(*_WARM_METRICS, "objective_calls"))
                comparison["success_intervals"] = {name: _b3_success_intervals(paired, name, settings)
                                                   for name in (method, "B3")}
                comparisons.append({**metadata, **comparison})
    used = set(restart_rows)
    selected_inputs = (("B1", b1_tasks, b1_selected), ("B2", b2_tasks, b2_selected), ("B3", tasks, selected))
    return {"analysis_version": B3_ANALYSIS_VERSION, "settings": settings, "library_id": library["library_id"],
            "graph_qaoa": graph_rows, "restarts": list(restart_rows.values()), "diagnostics": [],
            "comparisons": comparisons, "complete": all(row["complete"] for row in comparisons),
            "selected_costs_by_method": {method: _costs([row for identity, row in attempts.items() if identity in used])
                                         for method, _, attempts in selected_inputs},
            "selected_costs_by_method_depth": {
                method: {str(p): _costs([attempts[task["task_id"]] for task in planned if task["p"] == p
                                        and task["task_id"] in attempts and task["task_id"] in used]) for p in sorted(depths)}
                for method, planned, attempts in selected_inputs}}
