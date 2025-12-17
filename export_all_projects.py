#!/usr/bin/env python3
"""
Phoenix Export Tool

This script exports data from Phoenix (datasets, experiments, traces with evaluations)
for import into Arize.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from phoenix.client import Client as PhoenixClient

from exporters import (
    export_datasets_with_experiments,
    export_traces,
)
from utils import create_client_with_retry, parse_export_args

logger = logging.getLogger(__name__)

load_dotenv()

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    args = parse_export_args()

    base_url = os.environ.get("PHOENIX_ENDPOINT")
    if not base_url:
        logger.error("No Phoenix Endpoint URL provided. Set the PHOENIX_ENDPOINT environment variable.")
        return

    api_key = os.environ.get("PHOENIX_API_KEY")
    base_export_dir = os.environ.get("PHOENIX_EXPORT_DIR", "phoenix_export")
    base_url = base_url.rstrip("/")

    logger.info(f"Connecting to Phoenix server at {base_url}")
    logger.info(f"Exporting data to: {os.path.abspath(base_export_dir)}")

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = create_client_with_retry(base_url=base_url, headers=headers)
    phoenix_client = PhoenixClient(base_url=base_url, api_key=api_key)

    successful_exports = []
    failed_exports = []

    if not (
        args.all
        or args.datasets_experiments
        or args.traces
    ):
        logger.error("No export type selected. Use --help to see available options.")
        return

    projects_dir = os.path.join(base_export_dir, "projects")
    os.makedirs(projects_dir, exist_ok=True)

    if args.all or args.datasets_experiments:
        logger.info("Exporting datasets and experiments...")
        datasets_dir = os.path.join(base_export_dir, "datasets")
        results_file = os.path.join(RESULTS_DIR, "dataset_experiment_export_results.json")

        try:
            results = export_datasets_with_experiments.export_all(
                client=client,
                phoenix_client=phoenix_client,
                output_dir=datasets_dir,
                results_file=results_file,
            )

            if results:
                logger.info(
                    f"Successfully exported {results['datasets']} datasets, "
                    f"{results['total_experiments']} experiments"
                )
                successful_exports.append("datasets-experiments")
            else:
                failed_exports.append("datasets-experiments")
        except Exception as e:
            logger.error(f"Error exporting datasets and experiments: {e}")
            failed_exports.append("datasets-experiments")

    if args.all or args.traces:
        logger.info("Exporting traces, evaluations, annotations, and project metadata...")
        results_file = os.path.join(RESULTS_DIR, "trace_export_results.json")

        try:
            results = export_traces.export_traces(
                client=client,
                phoenix_client=phoenix_client,
                output_dir=projects_dir,
                project_names=None,
                results_file=results_file,
            )

            if results:
                success_count = sum(1 for p in results.values() if p.get("status") == "exported")
                total_traces = sum(p.get("trace_count", 0) for p in results.values())
                total_evaluations = sum(p.get("evaluation_count", 0) for p in results.values())
                total_annotations = sum(p.get("annotation_count", 0) for p in results.values())

                logger.info(
                    f"Successfully exported {total_traces} traces, {total_evaluations} evaluations, "
                    f"and {total_annotations} human annotations from {success_count} projects"
                )
                successful_exports.append("traces")
            else:
                failed_exports.append("traces")
        except Exception as e:
            logger.error(f"Error exporting traces: {e}")
            failed_exports.append("traces")

    print("\n=== Export Summary ===")
    if successful_exports:
        logger.info(f"Successfully exported: {', '.join(successful_exports)}")

    if failed_exports:
        logger.error(f"Failed to export: {', '.join(failed_exports)}")

    print("\n=== Export Results Files ===")
    for result_type in ["dataset_experiment", "trace"]:
        result_file = RESULTS_DIR / f"{result_type}_export_results.json"
        if result_file.exists():
            print(f"- {result_type.replace('_', ' ').title()} results: {result_file}")

    if failed_exports:
        sys.exit(1)


if __name__ == "__main__":
    main()
