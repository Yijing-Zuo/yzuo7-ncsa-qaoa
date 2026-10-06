"""Build or revise a frozen library, without QAOA optimization."""

import argparse
from collections import Counter
from pathlib import Path

from qaoa_study.graphs import generate_library, resplit_library
from qaoa_study.records import annotate_library, check_library, read_json, read_library, save_library, write_once_json


def ws_rgg_config(graphs_per_cell):
    """Set an explicit extension size; every new graph remains evaluation-only."""
    if type(graphs_per_cell) is not int or graphs_per_cell < 1:
        raise ValueError("graphs_per_cell must be a positive integer.")
    config = read_json(Path(__file__).resolve().parents[1] / "configs/graphs-ws-rgg.json")
    if graphs_per_cell > config["max_attempts_per_cell"]:
        raise ValueError("Requested graphs per cell exceed the generation attempt limit.")
    return {**config, "quota_per_cell": graphs_per_cell,
            "tier2_count": len(config["cells"]) * graphs_per_cell, "training_count": 0}


def matched_ws_rgg_config(original_library):
    """Match each original size's cell quota while retaining calibrated WS/RGG geometry."""
    config = ws_rgg_config(1)
    original = original_library["config"]
    sizes = original.get("sizes", [])
    expected_sizes = {cell["n"] for cell in config["cells"]}
    if (not isinstance(sizes, list) or any(type(n) is not int for n in sizes)
            or len(sizes) != len(expected_sizes) or set(sizes) != expected_sizes):
        raise ValueError("Original sizes must cover all calibrated WS/RGG sizes exactly once.")
    quota = original.get("quota_per_cell")
    if type(quota) is int:
        quota = {str(n): quota for n in sizes}
    if (not isinstance(quota, dict) or set(quota) != {str(n) for n in sizes}
            or any(type(value) is not int or value < 1 for value in quota.values())):
        raise ValueError("Original quota_per_cell must give a positive integer for every size.")
    max_attempts = original.get("max_attempts_per_cell")
    if type(max_attempts) is not int or max_attempts < max(quota.values()):
        raise ValueError("Original max_attempts_per_cell must be an integer at least as large as every quota.")
    return {**config, "quota_per_cell": dict(quota), "max_attempts_per_cell": max_attempts,
            "tier2_count": sum(quota[str(cell["n"])] for cell in config["cells"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Generation settings; defaults to configs/graphs-development.json.")
    parser.add_argument("--output", type=Path, default=Path("data/development/stage3/library"))
    parser.add_argument("--evidence", default="validation/stage3-library.json")
    parser.add_argument("--allow-study-build", action="store_true",
                        help="Explicitly enable the potentially expensive candidate-study exact annotations.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resplit-from", type=Path, help="Read this frozen library instead of generating graphs.")
    source.add_argument("--extend-ws-rgg", type=Path, metavar="ORIGINAL_LIBRARY",
                        help="Build a separate WS/RGG library, excluding the complete original library.")
    quantity = parser.add_mutually_exclusive_group()
    quantity.add_argument("--graphs-per-cell", type=int,
                          help="Number of new graphs in each of 16 WS/RGG cells.")
    quantity.add_argument("--match-original-quotas", action="store_true",
                          help="Use the original library's quota for each node count in all WS/RGG cells.")
    parser.add_argument("--exclude-library", type=Path, action="append", default=[],
                        help="Reject graphs isomorphic to this prior library; repeat for multiple libraries.")
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args(argv)
    if args.extend_ws_rgg is not None:
        if args.config is not None or args.split_seed is not None:
            parser.error("--extend-ws-rgg uses configs/graphs-ws-rgg.json and its frozen split seed")
        if not args.match_original_quotas and (args.graphs_per_cell is None or args.graphs_per_cell < 1):
            parser.error("--extend-ws-rgg requires --match-original-quotas or a positive --graphs-per-cell")
    elif args.graphs_per_cell is not None or args.match_original_quotas:
        parser.error("--graphs-per-cell and --match-original-quotas require --extend-ws-rgg")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; frozen libraries are not overwritten.")
    if Path(args.evidence).exists():
        raise FileExistsError("Choose a new evidence path; previous validation evidence is retained.")
    if args.resplit_from is not None:
        if args.exclude_library:
            parser.error("--exclude-library applies only to generation, not resplitting")
        if args.split_seed is None:
            parser.error("--resplit-from requires --split-seed")
        library = resplit_library(read_library(args.resplit_from), split_seed=args.split_seed)
    else:
        excluded = ([args.extend_ws_rgg] if args.extend_ws_rgg is not None else []) + args.exclude_library
        excluded_libraries = [read_library(path) for path in excluded]
        if args.match_original_quotas:
            config = matched_ws_rgg_config(excluded_libraries[0])
        elif args.extend_ws_rgg is not None:
            config = ws_rgg_config(args.graphs_per_cell)
        else:
            config = read_json(args.config or "configs/graphs-development.json")
        if config["purpose"] != "development_validation_only":
            if not args.allow_study_build:
                parser.error("Study builds require explicit --allow-study-build.")
            sizes = {cell["n"] for cell in config["cells"]} if "cells" in config else set(config["sizes"])
            if not sizes <= set(config["exact_sizes"]):
                raise ValueError("Study builds require exact MaxCut annotations for every configured size.")
        library = generate_library(config, excluded_libraries=excluded_libraries)
        library = annotate_library(library, config["exact_sizes"], config["exact_chunk_size"])
    save_library(args.output, library)
    restored = read_library(args.output)
    if restored != library:
        raise AssertionError("Parquet roundtrip changed research data.")
    evidence = {"library_id": library["library_id"], "path": str(args.output),
                "roundtrip_equal": True, **check_library(restored),
                "by_n": dict(Counter(record["n"] for record in restored["graphs"])),
                "by_representative_family": dict(Counter(record["provenance"][0]["family"]
                                                         for record in restored["graphs"])),
                "cells": restored["cells"], "split_summary": restored["split_summary"],
                "attempts": len(restored["attempts"]),
                "multi_family_classes": sum(len(record["families"]) > 1 for record in restored["graphs"])}
    write_once_json(args.evidence, evidence)
    print(f"Saved {len(restored['graphs'])} graphs: {args.output}; checks: {evidence['split_counts']}")


if __name__ == "__main__":
    main()
