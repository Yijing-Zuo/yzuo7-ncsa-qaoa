"""Run one small development optimization and save its complete attempt."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import networkx as nx

from qaoa_study.optimize import OptimizerSettings, optimize_qaoa, random_angles
from qaoa_study.records import read_json, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimize-development.json")
    parser.add_argument("--graph", choices=("triangle", "path5", "cycle5"), default="triangle")
    parser.add_argument("--p", type=int, choices=(1, 2), default=1)
    parser.add_argument("--theta", type=float, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    graphs = {"triangle": nx.cycle_graph(3), "path5": nx.path_graph(5), "cycle5": nx.cycle_graph(5)}
    theta = args.theta if args.theta is not None else random_angles(args.p, seed)
    result = optimize_qaoa(graphs[args.graph], args.p, theta, OptimizerSettings(**config["optimizer"]))
    attempt_id = uuid4().hex
    result.update(attempt_id=attempt_id, completed_utc=datetime.now(timezone.utc).isoformat(),
                  graph_name=args.graph, initialization="explicit" if args.theta is not None else "B1_uniform",
                  method=None if args.theta is not None else "B1",
                  seed=None if args.theta is not None else seed, experiment_role="development",
                  pool=None, graph_split="not_applicable", config=config)
    output = args.output or Path("data/development/optimization") / f"{attempt_id}.json"
    save_run(output, result)
    print(f"{output}: {result['stop_reason']}, C_final={result['C_final']}, counts={result['counts']}")


if __name__ == "__main__":
    main()
