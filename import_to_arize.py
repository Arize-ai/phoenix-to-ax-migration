#!/usr/bin/env python3
"""
Phoenix to Arize Import Tool

This script provides a unified interface to import data from Phoenix export
to Arize. It can import datasets, experiments, and traces (with evaluations).
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from importers import (
    import_datasets_with_experiments,
    import_traces,
)
from utils import parse_import_args

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


def import_datasets_experiments_wrapper(export_dir: str, api_key: str, space_id: str) -> bool:
    try:
        logger.info("Importing datasets and experiments...")
        results_path = RESULTS_DIR / "dataset_experiment_import_results.json"
        result = import_datasets_with_experiments.import_all(
            export_dir=export_dir,
            space_id=space_id,
            arize_api_key=api_key,
            results_file=str(results_path),
        )

        if "error" in result:
            logger.error(f"Error importing datasets and experiments: {result['error']}")
            return False

        datasets_created = result.get("datasets_created", 0)
        datasets_existing = result.get("datasets_existing", 0)
        experiments_created = result.get("total_experiments_created", 0)

        logger.info(
            f"Import complete: {datasets_created} datasets created, "
            f"{experiments_created} experiments created"
        )
        logger.info(f"Results saved to {results_path}")

        return (datasets_created + datasets_existing) > 0
    except Exception as e:
        logger.error(f"Error importing datasets and experiments: {e}")
        return False


def import_traces_wrapper(export_dir: str, api_key: str, space_id: str) -> bool:
    try:
        logger.info("Importing traces...")
        results_path = RESULTS_DIR / "trace_import_results.json"

        try:
            result = import_traces.import_traces(
                export_dir=export_dir,
                space_id=space_id,
                arize_api_key=api_key,
                results_file=str(results_path),
            )
        except Exception as import_error:
            logger.error(f"Error importing traces: {import_error}")
            if results_path.exists():
                try:
                    with open(results_path, "r") as f:
                        result = json.load(f)
                        logger.info("Loaded existing trace import results from file")
                except (json.JSONDecodeError, IOError) as load_error:
                    logger.error(f"Failed to load existing results: {load_error}")
                    return False
            else:
                return False

        if not result or not isinstance(result, dict):
            logger.error("No traces were imported")
            return False

        success_count = sum(
            1 for value in result.values()
            if isinstance(value, dict) and value.get("status") in ["imported", "skipped"]
        )
        total_processed = sum(1 for k, v in result.items() if k not in ["projects", "timestamp"] and isinstance(v, dict))

        logger.info(f"Successfully processed traces from {success_count}/{total_processed} projects")
        logger.info(f"Trace import results saved to {results_path}")
        return success_count > 0
    except Exception as e:
        logger.error(f"Error importing traces: {e}")
        return False


def main() -> None:
    load_dotenv()

    args = parse_import_args()

    api_key = (os.getenv("ARIZE_API_KEY") or "").strip()
    if not api_key:
        logger.error("No Arize API key provided. Set the ARIZE_API_KEY environment variable.")
        return

    space_id = (os.getenv("ARIZE_SPACE_ID") or "").strip()
    if not space_id:
        logger.error("No Arize Space ID found. Set ARIZE_SPACE_ID in your .env file.")
        return

    export_dir = os.getenv("PHOENIX_EXPORT_DIR", "phoenix_export")

    if not (args.all or args.datasets_experiments or args.traces):
        logger.error("No import type selected. Use --help to see available options.")
        return

    successful_imports = []
    failed_imports = []

    if args.all or args.datasets_experiments:
        logger.info("Step 1/2: Importing datasets and experiments...")
        if import_datasets_experiments_wrapper(export_dir, api_key, space_id):
            successful_imports.append("datasets-experiments")
        else:
            failed_imports.append("datasets-experiments")

    if args.all or args.traces:
        logger.info("Step 2/2: Importing traces (with evaluations)...")
        if import_traces_wrapper(export_dir, api_key, space_id):
            successful_imports.append("traces")
        else:
            failed_imports.append("traces")

    if successful_imports:
        logger.info(f"Successfully imported: {', '.join(successful_imports)}")

    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
