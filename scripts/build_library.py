"""Build or revise a frozen library, without QAOA optimization."""

import argparse
from collections import Counter
from pathlib import Path

from qaoa_study.graphs import generate_library, resplit_library
from qaoa_study.records import annotate_library, check_library, read_json, read_library, save_library, write_once_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graphs-development.json")
    parser.add_argument("--output", type=Path, default=Path("data/development/stage3/library"))
    parser.add_argument("--evidence", default="validation/stage3-library.json")
    parser.add_argument("--allow-study-build", action="store_true",
                        help="Explicitly enable the potentially expensive candidate-study exact annotations.")
    parser.add_argument("--resplit-from", type=Path, help="Read this frozen library instead of generating graphs.")
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; frozen libraries are not overwritten.")
    if Path(args.evidence).exists():
        raise FileExistsError("Choose a new evidence path; previous validation evidence is retained.")
    if args.resplit_from is not None:
        if args.split_seed is None:
            parser.error("--resplit-from requires --split-seed")
        library = resplit_library(read_library(args.resplit_from), split_seed=args.split_seed)
    else:
        config = read_json(args.config)
        if config["purpose"] != "development_validation_only":
            if not args.allow_study_build:
                parser.error("Study builds require explicit --allow-study-build after pilot/configuration review.")
            if not set(config["sizes"]) <= set(config["exact_sizes"]):
                raise ValueError("Study builds require exact MaxCut annotations for every configured size.")
        library = generate_library(config)
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
