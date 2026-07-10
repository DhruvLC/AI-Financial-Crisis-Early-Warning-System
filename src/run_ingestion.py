"""CLI entry point for the data ingestion module.

Examples:
    python src/run_ingestion.py --config configs/ingestion.yaml
    python src/run_ingestion.py --only kaggle_bankruptcy fred
    python src/run_ingestion.py --list
"""
from __future__ import annotations

import argparse
import sys

from ingestion.runner import IngestionRunner
from ingestion.sources import SOURCE_REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser(description="Financial EWS data ingestion")
    ap.add_argument("--config", default="configs/ingestion.yaml")
    ap.add_argument("--only", nargs="*", help="subset of source names to run")
    ap.add_argument("--list", action="store_true", help="list known sources and exit")
    args = ap.parse_args()

    if args.list:
        print("Available sources:")
        for name in SOURCE_REGISTRY:
            print(f"  - {name}")
        return 0

    runner = IngestionRunner(args.config)
    results = runner.run(only=args.only)

    print("\n─── Ingestion summary ───")
    for r in results:
        mark = "✓" if r.status == "success" else "✗"
        detail = f"{r.n_rows} rows" if r.status == "success" else r.error
        print(f"  {mark} {r.source:<20} {r.status:<8} {detail}")

    failed = [r for r in results if r.status != "success"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
