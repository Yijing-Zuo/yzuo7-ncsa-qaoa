"""One L-BFGS-B run with accepted endpoints and an enforced evaluation budget."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from .qaoa import make_qaoa, qaoa_loss_and_gradient


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
