"""Compute one small graph's exact optimum and QAOA score; no optimization."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pennylane as qml

from qaoa_study.exact import exact_maxcut
from qaoa_study.qaoa import make_qaoa, p1_expectation, qaoa_gradient, qaoa_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=("edge", "triangle", "path5", "cycle5"), default="triangle")
    parser.add_argument("--p", type=int, choices=(1, 2), default=1)
    parser.add_argument("--theta", type=float, nargs="+", help="Radian angles: gamma..., beta...")
    parser.add_argument("--output", type=Path, help="Optionally save the printed JSON record.")
    args = parser.parse_args()
    graphs = {
        "edge": nx.path_graph(2),
        "triangle": nx.cycle_graph(3),
        "path5": nx.path_graph(5),
        "cycle5": nx.cycle_graph(5),
    }
    graph = graphs[args.graph]
    theta = np.asarray(args.theta if args.theta is not None else
                       ([0.37, 0.21] if args.p == 1 else [0.37, -0.81, 0.21, 0.46]), dtype=np.float64)
    if theta.shape != (2 * args.p,):
        parser.error(f"--p {args.p} requires {2 * args.p} angles.")

    optimum, side = exact_maxcut(graph)
    circuit = make_qaoa(graph, args.p)
    calls = []
    # These are three real API requests, not optimizer iterations or a budget.
    for kind, evaluate in (
        ("score", lambda: float(circuit(theta))),
        ("loss", lambda: float(qaoa_loss(theta, circuit))),
        ("score_gradient", lambda: qaoa_gradient(theta, circuit).tolist()),
    ):
        start = perf_counter()
        with qml.Tracker(circuit.device) as tracker:
            value = evaluate()
        calls.append({"kind": kind, "value": value, "elapsed_seconds": perf_counter() - start,
                      "device_tracker_totals": tracker.totals})
    record = {
        "graph": args.graph, "nodes": list(graph.nodes), "edges": list(graph.edges),
        "p": args.p, "theta_gamma_then_beta_radians": theta.tolist(),
        "C_star": optimum, "optimal_cut_side": sorted(side),
        "calls": calls,
    }
    if args.p == 1:
        record["p1_analytic_score"] = p1_expectation(graph, *theta)
    text = json.dumps(record, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
