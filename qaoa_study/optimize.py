"""Exact and finite-shot optimizers with separate endpoints and work budgets."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from .qaoa import cut_sample_statistics, make_qaoa, make_qaoa_sampler, qaoa_loss_and_gradient


@dataclass(frozen=True)
class OptimizerSettings:
    """Development settings, to be calibrated before production by the pilot."""

    max_evaluations: int = 48
    maxiter: int = 30
    maxls: int = 20
    ftol: float = 1e-10
    gtol: float = 1e-7

    def __post_init__(self):
        if any(value < 1 or int(value) != value for value in
               (self.max_evaluations, self.maxiter, self.maxls)):
            raise ValueError("Evaluation, iteration and line-search limits must be positive integers.")
        if not all(np.isfinite(value) and value >= 0 for value in (self.ftol, self.gtol)):
            raise ValueError("Optimizer tolerances must be finite and nonnegative.")


def random_angles(p: int, seed: int) -> np.ndarray:
    """Draw one reproducible B1 start, gamma in [0,2pi), beta in [0,pi)."""
    if p not in (1, 2):
        raise ValueError("This study implements p=1 and p=2 only.")
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)))


def failure_record(theta0, settings: OptimizerSettings, error: Exception, *,
                   p=None, stop_reason="program_error") -> dict:
    """Represent a failure before objective entry, with no invented computation.

    This has the same completion/endpoint/trace fields as an optimizer attempt.
    Task identity, attempt identity and batch provenance belong to the caller.
    """
    return {
        "record_version": 3, "run_completed": True, "p": p,
        "theta0": [float(value) if np.isfinite(value) else None for value in theta0],
        "settings": asdict(settings), "theta_final": None, "C_final": None,
        "final_call_id": None, "best_seen": None, "best_theta": None,
        "best_call_id": None, "trace": [],
        "counts": {name: 0 for name in ("objective_calls", "gradient_requests",
                                        "device_executions", "device_derivatives", "device_vjps", "iterations")},
        "elapsed_seconds": 0.0, "stop_reason": stop_reason, "optimizer": None,
        "error": f"{type(error).__name__}: {error}",
    }


class _RunStopped(Exception):
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail


def _run_lbfgsb(objective, theta0, settings: OptimizerSettings, device=None) -> dict:
    """Record one joint loss/gradient objective; separate for controlled tests.

    The public QAOA entry supplies the device. A test-only classical objective
    may omit it; its quantum-work counts are then zero. No other backend is used.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)
    if theta0.ndim != 1 or not np.all(np.isfinite(theta0)):
        raise ValueError("theta0 must be a finite one-dimensional vector.")
    started = perf_counter()
    trace = []
    accepted = None
    counts = {name: 0 for name in ("objective_calls", "gradient_requests",
                                  "device_executions", "device_derivatives", "device_vjps", "iterations")}

    def evaluate(theta):
        nonlocal accepted
        # Charge each attempted joint computation before entering the device.
        if counts["objective_calls"] >= settings.max_evaluations:
            raise _RunStopped("budget_exhausted")
        counts["objective_calls"] += 1
        counts["gradient_requests"] += 1
        row = {"call_id": counts["objective_calls"],
               "theta": [float(value) if np.isfinite(value) else None for value in theta],
               "C": None, "loss": None, "gradient": None, "accepted": False,
               "valid": False, "elapsed_seconds": None, "device_work": {}, "error": None}
        tracker = qml.Tracker(device) if device is not None else None
        failure = None
        try:
            if not np.all(np.isfinite(theta)):
                raise _RunStopped("numerical_error", f"Nonfinite trial angles: {theta!r}; null marks invalid coordinates.")
            with tracker if tracker is not None else nullcontext():
                loss, gradient = objective(theta)
            loss = float(loss)
            gradient = np.asarray(gradient, dtype=np.float64)
            if np.isfinite(loss):
                row.update(loss=loss, C=-loss)
            if gradient.shape == theta.shape and np.all(np.isfinite(gradient)):
                row["gradient"] = gradient.tolist()
            if row["C"] is None or row["gradient"] is None:
                failure = _RunStopped("numerical_error", "Nonfinite loss/gradient or incorrect gradient shape.")
            else:
                row["valid"] = True
        except _RunStopped as stopped:
            failure = stopped
        except Exception as error:
            failure = _RunStopped("evaluation_error", f"{type(error).__name__}: {error}")
        finally:
            row["device_work"] = {} if tracker is None else dict(tracker.totals)
            counts["device_executions"] += int(row["device_work"].get("executions", 0))
            counts["device_derivatives"] += int(row["device_work"].get("derivatives", 0))
            counts["device_vjps"] += int(row["device_work"].get("vjps", 0))
            row["elapsed_seconds"] = perf_counter() - started
            row["error"] = failure.detail if failure else None
            trace.append(row)
        if failure:
            raise failure
        if len(trace) == 1:
            row["accepted"] = True
            accepted = row
        return loss, gradient

    def callback(xk):
        nonlocal accepted
        # SciPy invokes this only for NEW_X, after accepting the line search.
        row = next(row for row in reversed(trace)
                   if row["valid"] and np.array_equal(row["theta"], xk))
        row["accepted"] = True
        accepted = row
        counts["iterations"] += 1

    raw = None
    error = None
    try:
        result = minimize(evaluate, theta0, method="L-BFGS-B", jac=True, bounds=None,
                          callback=callback,
                          options={"maxfun": settings.max_evaluations, "maxiter": settings.maxiter,
                                   "maxls": settings.maxls, "ftol": settings.ftol, "gtol": settings.gtol})
        raw = {name: getattr(result, name) for name in ("status", "success", "message", "nit", "nfev", "njev")}
        raw.update(x=result.x.tolist(), fun=float(result.fun), jac=result.jac.tolist())
        if result.success:
            reason = "converged"
        elif result.status == 1 and counts["iterations"] >= settings.maxiter:
            reason = "iteration_limit"
        elif result.status == 1 and counts["objective_calls"] >= settings.max_evaluations:
            reason = "budget_exhausted"
        elif result.status == 2:
            reason = "line_search_failed"
        else:
            reason = "optimizer_failed"
    except _RunStopped as stopped:
        reason, error = stopped.reason, stopped.detail
    except Exception as failed:
        reason, error = "optimizer_failed", f"{type(failed).__name__}: {failed}"
    finite_rows = [row for row in trace if row["C"] is not None]
    best = max(finite_rows, key=lambda row: row["C"], default=None)
    return {
        "record_version": 3, "run_completed": True,
        "theta0": theta0.tolist(), "settings": asdict(settings),
        "theta_final": accepted["theta"] if accepted else None,
        "C_final": accepted["C"] if accepted else None,
        "final_call_id": accepted["call_id"] if accepted else None,
        "best_seen": best["C"] if best else None,
        "best_theta": best["theta"] if best else None,
        "best_call_id": best["call_id"] if best else None,
        "trace": trace, "counts": counts, "elapsed_seconds": perf_counter() - started,
        "stop_reason": reason, "optimizer": raw, "error": error,
    }


def optimize_qaoa(graph, p: int, theta0, settings: OptimizerSettings, *,
                  circuit=None, backend: str = "default.qubit") -> dict:
    """Run the single study objective, with no reference value or early hit stop.

    C_final is the last accepted valid point, including when a later trial fails.
    best_seen may belong to an unaccepted trial. A failed run remains failed even
    if it retains a valid preceding endpoint. Reruns start again at theta0.
    A worker may supply a QNode already built for this graph and depth; the
    caller owns that correspondence. Its device is used without reconstruction,
    while each call creates fresh optimizer state, counters and trace.
    """
    if np.shape(theta0) != (2 * p,):
        raise ValueError(f"Expected {2 * p} angles in [gamma..., beta...] order.")
    try:
        if circuit is None:
            circuit = make_qaoa(graph, p, backend=backend)
    except Exception as error:
        result = failure_record(theta0, settings, error, p=p, stop_reason="evaluation_error")
    else:
        result = _run_lbfgsb(lambda theta: qaoa_loss_and_gradient(theta, circuit),
                            theta0, settings, device=circuit.device)
    # Preserve the input graph without assuming its labels are JSON scalars.
    nodes = list(graph)
    indices = {v: i for i, v in enumerate(nodes)}
    result.update(p=p, n=len(nodes), edges=[[indices[u], indices[v]] for u, v in graph.edges],
                  angle_order="gamma_then_beta_radians", optimizer_method="L-BFGS-B",
                  method=None, bounds=None, backend=circuit.device.name if circuit is not None else backend)
    return result


@dataclass(frozen=True)
class ShotOptimizerSettings:
    """Candidate finite-shot settings; validation has a separate shot budget."""

    shots_per_evaluation: int = 1024
    total_shots: int = 32768
    validation_shots: int = 4096
    cobyla_rhobeg: float = 0.5
    cobyla_tol: float = 0.001
    spsa_a: float = 0.05
    spsa_c: float = 0.1
    spsa_A: float = 10.0
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101

    def __post_init__(self):
        for name in ("shots_per_evaluation", "total_shots", "validation_shots"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < (0 if name == "total_shots" else 2)):
                raise ValueError("Shot counts must be integers; batches need at least two shots.")
            object.__setattr__(self, name, int(value))
        for name in ("cobyla_rhobeg", "cobyla_tol", "spsa_a", "spsa_c", "spsa_alpha", "spsa_gamma"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.spsa_A) or self.spsa_A < 0 or self.cobyla_tol > self.cobyla_rhobeg:
            raise ValueError("SPSA stability constant must be nonnegative; COBYLA tol must not exceed rhobeg.")


def _run_shot_optimizer(objective, theta0, settings, method, perturbation_seed,
                        evaluation_mode="finite_shots"):
    """Optimize negative sampled cut; never choose an endpoint from trace maxima.

    SPSA uses independent Rademacher perturbations, a_k=a/(A+k+1)^alpha
    and c_k=c/(k+1)^gamma. Its explicit two-call loop avoids global RNG state
    and hidden monitoring calls. An unpaired final batch is left unused.
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    if theta.ndim != 1 or not np.all(np.isfinite(theta)):
        raise ValueError("theta0 must be a finite one-dimensional vector.")
    if method not in ("COBYLA", "SPSA", "fixed"):
        raise ValueError("method must be COBYLA, SPSA or fixed.")
    if evaluation_mode not in ("finite_shots", "exact_control"):
        raise ValueError("evaluation_mode must be finite_shots or exact_control.")
    batch_shots = settings.shots_per_evaluation if evaluation_mode == "finite_shots" else 0
    limit = settings.total_shots // settings.shots_per_evaluation
    if method != "fixed" and limit < (len(theta) + 2 if method == "COBYLA" else 2):
        raise ValueError("Shot budget is too small for this optimizer's initial evaluations.")
    trace, raw, error, updates = [], None, None, 0
    started = perf_counter()

    def evaluate(point):
        if len(trace) >= limit:
            raise _RunStopped("budget_exhausted")
        row = {"call_id": len(trace) + 1,
               "theta": [float(value) if np.isfinite(value) else None for value in point],
               "shots_requested": batch_shots, "statistics": None, "error": None}
        trace.append(row)
        try:
            if not np.all(np.isfinite(point)):
                raise ValueError("Nonfinite trial angles.")
            statistics = objective(point)
            if statistics["shots"] != batch_shots or not np.isfinite(statistics["C_mean"]):
                raise ValueError("Invalid sampled score statistics or shot count.")
            if batch_shots and not all(np.isfinite(statistics[key]) and statistics[key] >= 0
                                       for key in ("sample_variance", "standard_error")):
                raise ValueError("Invalid sampled variance or standard error.")
            row["statistics"] = statistics
            return -statistics["C_mean"]
        except Exception as failed:
            row["error"] = f"{type(failed).__name__}: {failed}"
            raise _RunStopped("evaluation_error", row["error"]) from failed
        finally:
            row["elapsed_seconds"] = perf_counter() - started

    def remember(point):
        nonlocal theta
        if np.shape(point) != theta.shape or not np.all(np.isfinite(point)):
            raise _RunStopped("numerical_error", "Optimizer returned invalid angles.")
        theta = np.asarray(point, dtype=np.float64).copy()

    try:
        if method == "COBYLA":
            result = minimize(evaluate, theta, method="COBYLA", callback=remember,
                              options={"maxiter": limit, "rhobeg": settings.cobyla_rhobeg,
                                       "tol": settings.cobyla_tol})
            remember(result.x)
            raw = {"status": int(result.status), "success": bool(result.success),
                   "message": str(result.message), "nfev": int(result.nfev), "fun": float(result.fun)}
            reason = "converged" if result.success else (
                "budget_exhausted" if len(trace) == limit else "optimizer_failed")
        elif method == "SPSA":
            rng = np.random.default_rng(perturbation_seed)
            for k in range(limit // 2):
                delta = rng.choice([-1.0, 1.0], size=theta.shape)
                ck = settings.spsa_c / (k + 1) ** settings.spsa_gamma
                plus, minus = evaluate(theta + ck * delta), evaluate(theta - ck * delta)
                gradient = (plus - minus) * delta / (2 * ck)
                ak = settings.spsa_a / (settings.spsa_A + k + 1) ** settings.spsa_alpha
                remember(theta - ak * gradient)
                updates += 1
            reason = "budget_exhausted"
        else:
            reason = "fixed_start"
    except _RunStopped as stopped:
        reason, error = stopped.reason, stopped.detail
    except Exception as failed:
        reason, error = "optimizer_failed", f"{type(failed).__name__}: {failed}"
    return {
        "record_type": "finite_shot_optimization", "record_version": 1, "run_completed": True,
        "optimizer_method": method, "evaluation_mode": evaluation_mode,
        "nominal_evaluation_limit": limit,
        "theta0": np.asarray(theta0).tolist(), "theta_final": theta.tolist(),
        "settings": asdict(settings), "trace": trace, "optimizer": raw,
        "stop_reason": reason, "error": error, "elapsed_seconds": perf_counter() - started,
        "counts": {"optimization_evaluations": len(trace),
                   "optimization_shots_requested": len(trace) * batch_shots,
                   "optimization_shots_completed": sum(row["statistics"]["shots"] for row in trace
                                                        if row["statistics"] is not None),
                   "optimization_exact_evaluations": len(trace) if not batch_shots else 0,
                   "spsa_updates": updates},
    }


def optimize_qaoa_shots(graph, p: int, theta0, settings: ShotOptimizerSettings, *,
                       method: str, measurement_seed: int, perturbation_seed: int,
                       validation_seed: int, backend: str = "default.qubit", audit_exact: bool = True,
                       evaluation_mode: str = "finite_shots") -> dict:
    """Run a finite-shot method, then independently assess its frozen endpoint.

    Every requested objective batch consumes the optimization budget, including
    failed requests. Completed shots count successfully returned valid samples;
    failed device calls have unknown actual work. Validation and optional exact
    auditing never feed back into the optimizer. Failed runs remain failed even
    if their retained endpoint can be assessed. Seed roles must remain distinct.
    Invalid input/configuration raises before execution; device failures are
    retained in the returned record with any completed work.
    """
    if (isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer))
            or p not in (1, 2) or np.shape(theta0) != (2 * p,)):
        raise ValueError("Expected p=1 or p=2 and angles in [gamma..., beta...] order.")
    if backend not in ("default.qubit", "lightning.qubit"):
        raise ValueError("sampled backend must be default.qubit or lightning.qubit.")
    seeds = (measurement_seed, perturbation_seed, validation_seed)
    if any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0
           for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers.")
    p = int(p)
    measurement_seed, perturbation_seed, validation_seed = map(int, seeds)
    if measurement_seed == validation_seed:
        raise ValueError("Measurement and validation seeds must differ.")
    sampler, exact_circuit = None, None
    device_executions = {"optimization": 0, "validation": 0, "exact_audit": 0}
    started = perf_counter()

    def execute(circuit, theta, phase):
        tracker = qml.Tracker(circuit.device)
        try:
            with tracker:
                return circuit(np.asarray(theta, dtype=np.float64))
        finally:
            device_executions[phase] += int(tracker.totals.get("executions", 0))

    def exact_score(theta, phase):
        nonlocal exact_circuit
        if exact_circuit is None:
            exact_circuit = make_qaoa(graph, p, backend="default.qubit")
        score = float(execute(exact_circuit, theta, phase))
        if not np.isfinite(score):
            raise ValueError("Nonfinite exact score.")
        return score

    def objective(theta):
        nonlocal sampler
        if evaluation_mode == "exact_control":
            return {"C_mean": exact_score(theta, "optimization"), "sample_variance": None,
                    "standard_error": None, "shots": 0}
        if sampler is None:
            sampler = make_qaoa_sampler(graph, p, shots=settings.shots_per_evaluation,
                                        seed=measurement_seed, backend=backend)
        return cut_sample_statistics(graph, execute(sampler, theta, "optimization"))

    result = _run_shot_optimizer(objective, theta0, settings, method, perturbation_seed, evaluation_mode)
    result.update(p=p, n=len(graph), backend=backend, angle_order="gamma_then_beta_radians",
                  audit_backend="default.qubit", exact_control_backend="default.qubit",
                  seeds={"measurement": measurement_seed, "perturbation": perturbation_seed,
                         "validation": validation_seed}, validation=None, validation_error=None,
                  exact_audit=None, exact_audit_error=None)
    result["counts"].update(validation_shots_requested=settings.validation_shots,
                            validation_shots_completed=0, exact_audit_evaluations=0)
    times = {"optimization": result["elapsed_seconds"]}
    phase_started = perf_counter()
    try:
        validation = make_qaoa_sampler(graph, p, shots=settings.validation_shots,
                                       seed=validation_seed, backend=backend)
        result["validation"] = cut_sample_statistics(
            graph, execute(validation, result["theta_final"], "validation"))
        result["counts"]["validation_shots_completed"] = result["validation"]["shots"]
    except Exception as failed:
        result["validation_error"] = f"{type(failed).__name__}: {failed}"
    times["validation"] = perf_counter() - phase_started
    phase_started = perf_counter()
    if audit_exact:
        try:
            result["counts"]["exact_audit_evaluations"] = 1
            initial = exact_score(theta0, "exact_audit")
            final = initial
            if not np.array_equal(theta0, result["theta_final"]):
                result["counts"]["exact_audit_evaluations"] += 1
                final = exact_score(result["theta_final"], "exact_audit")
            result["exact_audit"] = {"C_initial": initial, "C_final": final}
        except Exception as failed:
            result["exact_audit_error"] = f"{type(failed).__name__}: {failed}"
    times["exact_audit"] = perf_counter() - phase_started if audit_exact else 0.0
    result["counts"]["device_executions"] = sum(device_executions.values())
    result["device_executions_by_phase"] = device_executions
    result["elapsed_seconds_by_phase"] = times
    result["elapsed_seconds"] = perf_counter() - started
    return result
