#!/usr/bin/env python3
"""
Consolidated Phoenix Datasets and Experiments Exporter
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from phoenix.client import Client as PhoenixClient
from tqdm import tqdm

from .utils import save_json

logger = logging.getLogger(__name__)


def _serialize_to_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_to_dict(item) for item in obj]
    if hasattr(obj, 'to_dict'):
        return _serialize_to_dict(obj.to_dict())
    if hasattr(obj, '__dict__'):
        return _serialize_to_dict(obj.__dict__)
    return str(obj)


def get_datasets(client: httpx.Client) -> List[Dict]:
    response = client.get("/v1/datasets")
    response.raise_for_status()
    return response.json().get("data", [])


def get_dataset_examples(client: httpx.Client, dataset_id: str) -> List[Dict]:
    response = client.get(f"/v1/datasets/{dataset_id}/examples")
    response.raise_for_status()
    return response.json().get("data", {}).get("examples", [])


def _extract_experiment_id(exp: Any) -> Optional[str]:
    if isinstance(exp, dict):
        return exp.get("id")
    return getattr(exp, "id", None) if hasattr(exp, "id") else None


def get_experiment_ids(phoenix_client: PhoenixClient, dataset_id: str) -> List[str]:
    try:
        experiments = phoenix_client.experiments.list(dataset_id=dataset_id)
        if not experiments:
            return []
        
        experiment_ids = []
        for exp in experiments:
            exp_id = _extract_experiment_id(exp)
            if exp_id:
                experiment_ids.append(str(exp_id))
            else:
                logger.warning(f"Could not extract ID from experiment object: {type(exp)}")
        
        return experiment_ids
    except Exception as e:
        logger.warning(f"Error getting experiments for dataset {dataset_id}: {e}")
        return []


def get_experiment(phoenix_client: PhoenixClient, experiment_id: str) -> Optional[Dict]:
    try:
        experiment = phoenix_client.experiments.get_experiment(experiment_id=str(experiment_id))
        experiment_dict = _serialize_to_dict(experiment)
        
        if not isinstance(experiment_dict, dict):
            return {"id": experiment_id, "data": experiment_dict}
        
        experiment_dict.setdefault("id", experiment_id)
        return experiment_dict
    except Exception as e:
        logger.warning(f"Error getting experiment {experiment_id}: {e}")
        return None


def export_all(
    client: httpx.Client,
    phoenix_client: PhoenixClient,
    output_dir: str,
    results_file: str,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    datasets = get_datasets(client)
    save_json(datasets, os.path.join(output_dir, "datasets.json"))

    results = {
        "datasets": len(datasets),
        "datasets_processed": [],
        "total_examples": 0,
        "total_experiments": 0,
    }

    for dataset in tqdm(datasets, desc="Exporting datasets"):
        dataset_id = dataset["id"]
        dataset_name = dataset.get("name", dataset_id)

        dataset_result = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "examples_count": 0,
            "experiments_count": 0,
            "status": "success",
        }

        try:
            examples = get_dataset_examples(client, dataset_id)
            examples_path = os.path.join(output_dir, f"dataset_{dataset_id}_examples.json")
            save_json(examples, examples_path)
            dataset_result["examples_count"] = len(examples)
            results["total_examples"] += len(examples)

            experiment_ids = get_experiment_ids(phoenix_client, dataset_id)
            experiments = [exp for exp_id in experiment_ids if (exp := get_experiment(phoenix_client, exp_id))]

            experiments_path = os.path.join(output_dir, f"dataset_{dataset_id}_experiments.json")
            save_json(experiments, experiments_path)
            dataset_result["experiments_count"] = len(experiments)
            results["total_experiments"] += len(experiments)

        except Exception as e:
            dataset_result["status"] = "error"
            dataset_result["error"] = str(e)
            logger.error(f"Error processing dataset {dataset_name}: {e}")

        results["datasets_processed"].append(dataset_result)

    save_json(results, results_file)
    return results
