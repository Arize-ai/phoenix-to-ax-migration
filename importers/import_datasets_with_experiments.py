"""
Phoenix to Arize Datasets and Experiments Importer
"""

import inspect
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from arize.experimental.datasets import ArizeDatasetsClient
from dotenv import load_dotenv
from tqdm import tqdm

from .utils import (
    RESULTS_DIR,
    load_json_file,
    save_results_to_file,
)

load_dotenv()

logger = logging.getLogger(__name__)

EXCLUDED_FIELDS = {"created_at", "updated_at"}
EXAMPLE_ID_FIELDS = ["dataset_example_id", "example_id", "exampleId", "example", "id"]
OUTPUT_FIELDS = ["output", "result", "response", "answer"]


def _get_dataset_type_enum(client: ArizeDatasetsClient):
    sig = inspect.signature(client.create_dataset)
    return sig.parameters["dataset_type"].annotation


def _flatten_dict(source: Dict, target: Dict, excluded: set = None) -> None:
    if excluded is None:
        excluded = set()
    for key, value in source.items():
        if key in excluded:
            continue
        if isinstance(value, dict):
            _flatten_dict(value, target, excluded)
        else:
            target[key] = value


def convert_phoenix_example_to_arize(phoenix_example: Dict) -> Dict:
    arize_example = {}
    phoenix_id = phoenix_example.get("id")
    if phoenix_id:
        arize_example["id"] = phoenix_id

    for field in ["input", "output", "metadata"]:
        if field in phoenix_example:
            value = phoenix_example[field]
            if isinstance(value, dict):
                _flatten_dict(value, arize_example, EXCLUDED_FIELDS)
            else:
                arize_example[field] = value

    return arize_example if arize_example else {"id": phoenix_id} if phoenix_id else {}


def _extract_example_id(phoenix_run: Dict) -> Optional[str]:
    return next((str(phoenix_run[field]) for field in EXAMPLE_ID_FIELDS if field in phoenix_run), None)


def _extract_output(phoenix_run: Dict) -> Optional[str]:
    for field in OUTPUT_FIELDS:
        value = phoenix_run.get(field)
        if value is None or (isinstance(value, str) and value.startswith("<generator object")):
            continue
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def convert_task_run_to_arize(
    task_run: Dict, example_id_mapping: Dict[str, str]
) -> Optional[Dict]:
    phoenix_example_id = task_run.get("dataset_example_id")
    arize_example_id = example_id_mapping.get(phoenix_example_id) if phoenix_example_id else None
    output = _extract_output(task_run)
    
    if not (arize_example_id and output):
        return None

    arize_run = {"example_id": arize_example_id, "result": output}
    excluded = {"dataset_example_id", "experiment_id", "id", "output"}
    for key, value in task_run.items():
        if key not in excluded:
            arize_run[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
    return arize_run


def create_dataset(
    api_key: str,
    space_id: str,
    name: str,
    examples_df: pd.DataFrame,
) -> Dict[str, Any]:
    if not space_id or not space_id.strip():
        return {"status": "error", "error": "Space ID cannot be empty"}

    try:
        client = ArizeDatasetsClient(api_key=api_key)
        dataset_type = _get_dataset_type_enum(client).INFERENCES

        dataset_id = client.create_dataset(
            space_id=space_id.strip(),
            dataset_name=name,
            dataset_type=dataset_type,
            data=examples_df,
            convert_dict_to_json=True,
        )

        if dataset_id:
            return {"status": "created", "dataset": {"id": dataset_id, "name": name}}
        return {"status": "error", "error": "Dataset creation returned None"}

    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "duplicate" in error_msg:
            return {"status": "already_exists", "error": "Dataset name already exists in space"}
        return {"status": "error", "error": str(e)}


def _load_credentials(space_id: Optional[str], api_key: Optional[str]) -> Tuple[str, str]:
    import os
    space_id = (space_id or os.getenv("ARIZE_SPACE_ID") or "").strip()
    api_key = (api_key or os.getenv("ARIZE_API_KEY") or "").strip()
    
    if not space_id:
        raise ValueError("Space ID cannot be empty. Check ARIZE_SPACE_ID in .env file.")
    if not api_key:
        raise ValueError("API key cannot be empty. Check ARIZE_API_KEY in .env file.")
    
    return space_id, api_key


def _create_example_id_mapping(phoenix_examples: List[Dict]) -> Dict[str, str]:
    return {ex.get("id"): ex.get("id") for ex in phoenix_examples if ex.get("id")}


def _process_experiment(
    phoenix_experiment: Dict,
    example_id_mapping: Dict[str, str],
    arize_dataset_id: str,
    space_id: str,
    api_key: str,
) -> Dict[str, Any]:
    experiment_name = (
        phoenix_experiment.get("project_name")
        or phoenix_experiment.get("name")
        or phoenix_experiment.get("experiment_id", "Unnamed Experiment")
    )

    experiment_info = {"name": experiment_name, "status": "error"}
    task_runs = phoenix_experiment.get("task_runs", [])

    if not task_runs:
        experiment_info["error"] = "No task runs found"
        return experiment_info

    arize_task_runs = [run for run in (convert_task_run_to_arize(tr, example_id_mapping) for tr in task_runs) if run]

    if not arize_task_runs:
        experiment_info["error"] = "No valid task runs after conversion"
        return experiment_info

    try:
        experiment_runs = []
        for task_run in arize_task_runs:
            experiment_run = {"example_id": task_run.get("example_id"), "output": task_run.get("result")}
            experiment_run.update({
                k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v)
                for k, v in task_run.items()
                if k not in {"example_id", "result"} and v is not None
            })
            experiment_runs.append(experiment_run)

        payload = {
            "name": experiment_name,
            "dataset_id": arize_dataset_id,
            "experiment_runs": experiment_runs,
        }

        response = requests.post(
            "https://api.arize.com/v2/experiments",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

        if response.status_code == 201:
            response_data = response.json()
            experiment_id = response_data.get("id") or response_data.get("experiment_id")
            experiment_info.update({
                "status": "created",
                "arize_id": experiment_id,
                "runs_count": len(arize_task_runs),
            })
        elif response.status_code == 409:
            experiment_info["status"] = "already_exists"
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error") or error_data.get("message") or response.text
            except (json.JSONDecodeError, ValueError):
                error_msg = f"HTTP {response.status_code}: {response.text}"
            experiment_info["error"] = error_msg
            logger.error(f"Failed to create experiment '{experiment_name}': {error_msg}")

    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "duplicate" in error_msg:
            experiment_info["status"] = "already_exists"
        else:
            experiment_info["error"] = str(e)
            logger.error(f"Exception creating experiment '{experiment_name}': {e}")

    return experiment_info


def import_all(
    export_dir: Union[str, Path],
    space_id: Optional[str] = None,
    arize_api_key: Optional[str] = None,
    results_file: Optional[str] = None,
) -> Dict[str, Any]:

    try:
        space_id, arize_api_key = _load_credentials(space_id, arize_api_key)
    except ValueError as e:
        logger.error(str(e))
        return {"error": str(e)}

    export_path = Path(export_dir)
    if not export_path.exists():
        logger.error(f"Export directory does not exist: {export_path}")
        return {"error": f"Export directory does not exist: {export_path}"}

    datasets_path = export_path / "datasets" / "datasets.json"
    if not datasets_path.exists():
        logger.error(f"Datasets file not found: {datasets_path}")
        return {"error": "Datasets file not found"}

    phoenix_datasets = load_json_file(datasets_path) or []

    results_path = Path(results_file) if results_file else RESULTS_DIR / "dataset_experiment_import_results.json"

    imported_datasets = []
    experiment_stats = {"created": 0, "existing": 0, "errors": 0}

    for phoenix_dataset in tqdm(phoenix_datasets, desc="Importing datasets"):
        phoenix_dataset_id = phoenix_dataset.get("id")
        phoenix_dataset_name = phoenix_dataset.get("name", phoenix_dataset_id)

        dataset_info = {
            "phoenix_id": phoenix_dataset_id,
            "phoenix_name": phoenix_dataset_name,
            "status": "error",
            "experiments": [],
        }

        try:
            examples_path = export_path / "datasets" / f"dataset_{phoenix_dataset_id}_examples.json"
            if not examples_path.exists():
                dataset_info["error"] = "Examples file not found"
                imported_datasets.append(dataset_info)
                continue

            phoenix_examples = load_json_file(examples_path) or []
            if not phoenix_examples:
                dataset_info["error"] = "No examples in dataset"
                imported_datasets.append(dataset_info)
                continue

            examples_df = pd.DataFrame([convert_phoenix_example_to_arize(ex) for ex in phoenix_examples])

            result = create_dataset(
                api_key=arize_api_key,
                space_id=space_id,
                name=phoenix_dataset_name,
                examples_df=examples_df,
            )

            status = result.get("status")
            if status == "created":
                arize_dataset_id = result.get("dataset", {}).get("id")
                if not arize_dataset_id:
                    dataset_info["error"] = "Dataset created but no ID returned"
                    logger.error(f"Dataset '{phoenix_dataset_name}' created but no ID returned")
                    imported_datasets.append(dataset_info)
                    continue
                dataset_info.update({"arize_id": arize_dataset_id, "status": "created"})
                example_id_mapping = _create_example_id_mapping(phoenix_examples)
            elif status == "already_exists":
                dataset_info.update({
                    "status": "already_exists",
                    "error": "Dataset name already exists (experiments skipped - need dataset_id)",
                })
                imported_datasets.append(dataset_info)
                continue
            else:
                dataset_info["error"] = result.get("error", "Unknown error")
                logger.error(f"Failed to create dataset '{phoenix_dataset_name}': {dataset_info['error']}")
                imported_datasets.append(dataset_info)
                continue

            experiments_path = export_path / "datasets" / f"dataset_{phoenix_dataset_id}_experiments.json"
            if experiments_path.exists():
                phoenix_experiments = load_json_file(experiments_path) or []
                for phoenix_experiment in phoenix_experiments:
                    experiment_info = _process_experiment(
                        phoenix_experiment, example_id_mapping, arize_dataset_id, space_id, arize_api_key
                    )
                    dataset_info["experiments"].append(experiment_info)
                    exp_status = experiment_info.get("status")
                    if exp_status == "created":
                        experiment_stats["created"] += 1
                    elif exp_status == "already_exists":
                        experiment_stats["existing"] += 1
                    else:
                        experiment_stats["errors"] += 1

            imported_datasets.append(dataset_info)

        except Exception as e:
            logger.error(f"Error processing dataset {phoenix_dataset_name}: {e}")
            dataset_info["error"] = str(e)
            imported_datasets.append(dataset_info)

    from collections import Counter
    status_counts = Counter(d.get("status") for d in imported_datasets)
    results = {
        "total_datasets": len(phoenix_datasets),
        "datasets_created": status_counts.get("created", 0),
        "datasets_existing": status_counts.get("already_exists", 0),
        "datasets_errors": status_counts.get("error", 0),
        "total_experiments_created": experiment_stats["created"],
        "total_experiments_existing": experiment_stats["existing"],
        "total_experiments_errors": experiment_stats["errors"],
        "datasets": imported_datasets,
    }

    save_results_to_file(results, results_path, "dataset and experiment import")

    logger.info(
        f"Import complete: {results['datasets_created']} datasets created, "
        f"{experiment_stats['created']} experiments created, "
        f"{experiment_stats['errors']} experiment errors"
    )

    return results
