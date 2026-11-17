import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.linalg import lstsq


def paired_intervals(method, baseline, metrics):
    def index(rows):
        mapping = {r["graph"]: r for r in rows}
        if not mapping or len(mapping) != len(rows):
            raise ValueError("Expected one observation per graph in each method.")
        return mapping

    left, right = index(method), index(baseline)
    if left.keys() != right.keys():
        raise ValueError("Methods must cover the same graphs.")
    differences = []
    cells = defaultdict(list)
    for i, graph in enumerate(sorted(left)):
        a, b = left[graph], right[graph]
        if a["cell"] != b["cell"]:
            raise ValueError(f"Design cells disagree for {graph}.")
        cells[a["cell"]].append(i)
        differences.append([a[m] - b[m] for m in metrics])
    values = np.asarray(differences)
    n = len(values)
    rng = np.random.default_rng(20261115)
    draws = np.zeros((2000, n))
    cell_weights = np.zeros(n)
    for indices in cells.values():
        k = len(indices)
        cell_weights[indices] = 1 / k / len(cells)
        draws[:, indices] = rng.multinomial(k, np.full(k, 1 / k), size=2000)
    result = {}
    for scheme, weights in (("graph_equal", np.full(n, 1 / n)), ("cell_equal", cell_weights)):
        means = weights @ values
        intervals = np.quantile((draws * weights) @ values, [.025, .975], axis=0)
        result[scheme] = {name: {"mean": float(means[j]), "ci95": intervals[:, j].tolist()}
                          for j, name in enumerate(metrics)}
    return result


def adjusted_family(current, historical):
    old_rates = {(r["iso_class_id"], r["p"]): r["success_rate"] for r in historical}
    lookup = {(r["iso_class_id"], r["p"]): r for r in current if r["scientific_label_complete"]}
    ids = sorted({k[0] for k in lookup})
    graph = [lookup[key, 1] for key in ids]
    strata = [(n, d) for n in (12, 16, 20, 24) for d in (3, 5)]
    families = ("regular", "ba", "sbm")
    centers = {s: np.mean([r["mean_degree"] for r in graph if (r["n"], r["degree_band"]) == s])
               for s in strata}
    names = ([f"n{n}_band{d}" for n, d in strata]
             + [f"{f}_minus_er_band{d}" for d in (3, 5) for f in families]
             + ["actual_degree_centered"])
    x = np.array([[float((r["n"], r["degree_band"]) == s) for s in strata]
                  + [float(r["family"] == f and r["degree_band"] == d) for d in (3, 5) for f in families]
                  + [r["mean_degree"] - centers[r["n"], r["degree_band"]]] for r in graph])
    outcomes = [f"p{p}_{m}" for p in (1, 2) for m in ("old_success", "new_success", "final_quality")]
    y = np.array([[v for p in (1, 2) for v in (old_rates[key, p], lookup[key, p]["success_rate"],
                  lookup[key, p]["evaluation_mean_ratio"])] for key in ids])
    cells = defaultdict(list)
    for i, r in enumerate(graph):
        cells[r["n"], r["family"], r["degree_band"]].append(i)
    rng = np.random.default_rng(20260930)
    groups = [indices for _, indices in sorted(cells.items())]
    resamples = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in groups]) for _ in range(2000)]
    models = {}
    for model in ("linear", "quadratic"):
        design = x if model == "linear" else np.column_stack([x, x[:, -1] ** 2])
        columns = names if model == "linear" else names + ["actual_degree_centered_squared"]
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank != design.shape[1]:
            raise ValueError("Rank-deficient adjusted-family design.")
        samples = []
        for ix in resamples:
            b, _, rank, _ = lstsq(design[ix], y[ix], lapack_driver="gelsy")
            if rank != design.shape[1]:
                raise ValueError("Rank-deficient bootstrap design.")
            samples.append(b)
        limits = np.quantile(samples, [.025, .975], axis=0)
        models[model] = {
            term: {outcome: {"estimate": float(beta[j, k]), "ci95": limits[:, j, k].tolist()}
                   for k, outcome in enumerate(outcomes)} for j, term in enumerate(columns)
        }
    return {"graphs": len(graph), "bootstrap_samples": 2000, "seed": 20260930,
            "outcomes": outcomes, "models": models}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("paired", help="Compare JSON arrays with one row per graph and arm.")
    pair.add_argument("method", type=Path, help="Rows with graph, cell, and numeric metric columns.")
    pair.add_argument("baseline", type=Path, help="Rows with matching graph and cell identifiers.")
    pair.add_argument("--metrics", nargs="+", required=True)
    family = commands.add_parser("family", help="Fit linear and quadratic adjusted family contrasts against ER.")
    family.add_argument("current", type=Path, help="Normalized graph metrics in Parquet format.")
    family.add_argument("historical", type=Path, help="Parquet columns: iso_class_id, p, success_rate.")
    args = parser.parse_args()
    if args.command == "paired":
        result = paired_intervals(json.loads(args.method.read_text(encoding="utf8")),
                                  json.loads(args.baseline.read_text(encoding="utf8")), args.metrics)
    else:
        columns = ["iso_class_id", "p", "success_rate", "scientific_label_complete", "mean_degree",
                   "n", "degree_band", "family", "evaluation_mean_ratio"]
        result = adjusted_family(pq.read_table(args.current, columns=columns).to_pylist(),
                                 pq.read_table(args.historical, columns=columns[:3]).to_pylist())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
