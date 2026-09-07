"""Build and freeze the small development library, without QAOA optimization."""

import argparse
from collections import Counter
from pathlib import Path

from qaoa_study.graphs import generate_library
from qaoa_study.records import annotate_library, check_library, read_json, read_library, save_library, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graphs-development.json")
    parser.add_argument("--output", type=Path, default=Path("data/development/stage3/library"))
    parser.add_argument("--evidence", default="validation/stage3-library.json")
    args = parser.parse_args()
    config = read_json(args.config)
    if config["purpose"] != "development_validation_only":
        raise ValueError("This entry point runs only the development configuration.")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; frozen libraries are not overwritten.")
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
    write_json(args.evidence, evidence)
    print(f"Saved {len(restored['graphs'])} graphs: {args.output}; checks: {evidence['split_counts']}")


if __name__ == "__main__":
    main()
