#!/usr/bin/env python3
"""
Phoenix Traces Exporter
"""

import logging
import os
from typing import Dict, List, Optional, Union

import httpx
import pandas as pd
from phoenix.client import Client as PhoenixClient
from tqdm import tqdm

from .utils import get_projects, save_json

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_project_metadata(client: httpx.Client, project_name: str) -> Dict:
    response = client.get(f"/v1/projects/{project_name}")
    response.raise_for_status()
    return response.json()


def export_project_traces(
    client: httpx.Client,
    phoenix_client: PhoenixClient,
    project_name: str,
    output_dir: str,
) -> Dict[str, Union[str, int]]:
    project_dir = os.path.join(output_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    result = {
        "project_name": project_name,
        "trace_count": 0,
        "span_count": 0,
        "evaluation_count": 0,
        "annotation_count": 0,
        "status": "exported"
    }

    try:
        try:
            project_metadata = get_project_metadata(client, project_name)
            save_json(project_metadata, os.path.join(project_dir, "project_metadata.json"))
        except (httpx.HTTPError, IOError):
            pass

        df = phoenix_client.spans.get_spans_dataframe(
            project_identifier=project_name,
            limit=100000,
            timeout=120
        )
        
        if df.empty:
            return result

        span_count = len(df)
        result["span_count"] = span_count

        trace_id_col = next((col for col in df.columns if "trace_id" in col.lower()), None)
        if trace_id_col:
            result["trace_count"] = df[trace_id_col].nunique()

        traces_file = os.path.join(project_dir, "traces.json")
        df.to_json(traces_file, orient="records", indent=2)

        try:
            all_annotations_dfs = []
            batch_size = 100
            total_spans = len(df)

            for i in range(0, total_spans, batch_size):
                batch_spans_df = df.iloc[i:i + batch_size].copy()
                
                try:
                    batch_annotations_df = phoenix_client.spans.get_span_annotations_dataframe(
                        spans_dataframe=batch_spans_df,
                        project_identifier=project_name,
                        timeout=120
                    )
                    if not batch_annotations_df.empty:
                        if "context.span_id" not in batch_annotations_df.columns and batch_annotations_df.index.name == "span_id":
                            batch_annotations_df["context.span_id"] = batch_annotations_df.index
                        all_annotations_dfs.append(batch_annotations_df)
                except Exception as e:
                    logger.debug(f"Failed to get annotations for batch {i}: {e}")
                    continue

            if all_annotations_dfs:
                annotations_df = pd.concat(all_annotations_dfs, ignore_index=False, sort=False)
                
                if "context.span_id" not in annotations_df.columns:
                    if annotations_df.index.name == "span_id":
                        annotations_df["context.span_id"] = annotations_df.index
                    else:
                        logger.warning(f"context.span_id column missing for project {project_name}, skipping export")
                        return result
                
                if not annotations_df.empty:
                    annotations_df = annotations_df.reset_index(drop=True)
                    
                    human_mask = annotations_df.get("annotator_kind", "") == "HUMAN"
                    human_annotations_df = annotations_df[human_mask].copy()
                    non_human_evaluations_df = annotations_df[~human_mask].copy()
                    
                    # Save non-human evaluations to evaluations.json
                    if not non_human_evaluations_df.empty:
                        result["evaluation_count"] = len(non_human_evaluations_df)
                        evaluations_file = os.path.join(project_dir, "evaluations.json")
                        non_human_evaluations_df.to_json(evaluations_file, orient="records", indent=2)
                    
                    # Save human annotations to annotations.json
                    if not human_annotations_df.empty:
                        result["annotation_count"] = len(human_annotations_df)
                        annotations_file = os.path.join(project_dir, "annotations.json")
                        human_annotations_df.to_json(annotations_file, orient="records", indent=2)
        except Exception as e:
            logger.debug(f"Error processing annotations for {project_name}: {e}")
            pass

        return result

    except Exception as e:
        logger.error(f"Error exporting {project_name}: {e}")
        result["status"] = "error"
        result["error"] = str(e)
        return result


def export_traces(
    client: httpx.Client,
    phoenix_client: PhoenixClient,
    output_dir: str,
    project_names: Optional[List[str]] = None,
    results_file: Optional[str] = None,
) -> Dict[str, Dict]:
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    try:
        if project_names is None:
            projects = get_projects(client)
            project_names = [p["name"] for p in projects]

        if not project_names:
            return results

        for project_name in tqdm(project_names, desc="Exporting traces"):
            results[project_name] = export_project_traces(
                client=client,
                phoenix_client=phoenix_client,
                project_name=project_name,
                output_dir=output_dir,
            )

        if results_file:
            save_json(results, results_file)

        return results

    except Exception as e:
        logger.error(f"Error during traces export: {e}")
        if results_file:
            save_json({"error": str(e), "projects": results}, results_file)
        return results