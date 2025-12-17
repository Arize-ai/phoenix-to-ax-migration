"""
Phoenix to Arize Trace Importer
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from arize.pandas.logger import Client
from tqdm import tqdm

from .utils import (
    RESULTS_DIR,
    get_projects,
)


def get_project_traces_dataframe(export_dir: Union[str, Path], project_name: str) -> pd.DataFrame:
    traces_path = Path(export_dir) / "projects" / project_name / "traces.json"
    if not traces_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_json(traces_path, orient="records")
        
        if "status_code" in df.columns and "status" not in df.columns:
            df["status"] = df["status_code"]
        
        required_columns = ["context.span_id", "context.trace_id", "name"]
        for col in required_columns:
            if col not in df.columns:
                return pd.DataFrame()
        
        return df
    except (pd.errors.EmptyDataError, ValueError, json.JSONDecodeError):
        return pd.DataFrame()


def get_project_evaluations_dataframe(export_dir: Union[str, Path], project_name: str) -> pd.DataFrame:
    evaluations_path = Path(export_dir) / "projects" / project_name / "evaluations.json"
    if not evaluations_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_json(evaluations_path, orient="records")
    except (pd.errors.EmptyDataError, ValueError, json.JSONDecodeError):
        return pd.DataFrame()


def get_project_annotations_dataframe(export_dir: Union[str, Path], project_name: str) -> pd.DataFrame:
    annotations_path = Path(export_dir) / "projects" / project_name / "annotations.json"
    if not annotations_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_json(annotations_path, orient="records")
    except (pd.errors.EmptyDataError, ValueError, json.JSONDecodeError):
        return pd.DataFrame()


def convert_evaluations_to_arize_format(evaluations_df: pd.DataFrame) -> pd.DataFrame:
    if evaluations_df.empty:
        return pd.DataFrame()

    if "context.span_id" not in evaluations_df.columns:
        return pd.DataFrame()

    non_human_evaluations = evaluations_df[evaluations_df.get("annotator_kind", "") != "HUMAN"].copy()
    
    if non_human_evaluations.empty:
        return pd.DataFrame()

    formatted_rows = []
    for span_id, group_df in non_human_evaluations.groupby("context.span_id"):
        if pd.isna(span_id):
            continue
        row = {"context.span_id": str(span_id)}

        for _, eval_row in group_df.iterrows():
            annotation_name = eval_row.get("annotation_name", "")
            if not annotation_name:
                continue

            name = annotation_name.lower().replace(" ", "_").replace("-", "_")

            label = eval_row.get("result.label")
            score = eval_row.get("result.score")
            explanation = eval_row.get("result.explanation")

            has_label = pd.notna(label) and label is not None
            has_score = pd.notna(score) and score is not None

            if not has_label and not has_score:
                continue

            if has_label:
                row[f"eval.{name}.label"] = str(label)
            if has_score:
                row[f"eval.{name}.score"] = float(score)
            if pd.notna(explanation) and explanation:
                row[f"eval.{name}.explanation"] = str(explanation)

        if any(key.startswith("eval.") for key in row.keys()):
            formatted_rows.append(row)

    if not formatted_rows:
        return pd.DataFrame()

    df = pd.DataFrame(formatted_rows)
    
    if "context.span_id" not in df.columns:
        return pd.DataFrame()
    
    df["context.span_id"] = df["context.span_id"].astype(str)

    for col in list(df.columns):
        if col == "context.span_id":
            continue
        if df[col].isna().all():
            df = df.drop(columns=[col])

    if "context.span_id" not in df.columns:
        return pd.DataFrame()

    return df


def convert_human_annotations_to_arize_format(evaluations_df: pd.DataFrame) -> pd.DataFrame:
    if evaluations_df.empty or "context.span_id" not in evaluations_df.columns:
        return pd.DataFrame()

    human_df = evaluations_df[evaluations_df["annotator_kind"] == "HUMAN"].copy()
    
    if human_df.empty:
        return pd.DataFrame()

    formatted_rows = []
    for span_id, group_df in human_df.groupby("context.span_id"):
        row = {"context.span_id": span_id}
        notes = []

        for _, eval_row in group_df.iterrows():
            name = eval_row.get("annotation_name", "").lower().replace(" ", "_").replace("-", "_")
            if not name:
                continue

            label = eval_row.get("result.label")
            score = eval_row.get("result.score")
            explanation = eval_row.get("result.explanation")

            if pd.notna(label):
                row[f"annotation.{name}.label"] = str(label)
            if pd.notna(score):
                row[f"annotation.{name}.score"] = float(score)
            if pd.notna(explanation):
                notes.append(str(explanation))

        if any(key.startswith("annotation.") and key.endswith((".label", ".score")) for key in row.keys()):
            if notes:
                row["annotation.notes"] = " | ".join(notes)
            formatted_rows.append(row)

    if not formatted_rows:
        return pd.DataFrame()

    df = pd.DataFrame(formatted_rows)
    
    for col in list(df.columns):
        if col not in ["context.span_id", "annotation.notes"] and df[col].isna().all():
            df = df.drop(columns=[col])

    return df


def fix_dataframe_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types for columns that Arize expects in specific formats.
    - Columns ending in .documents should be lists of dicts with string keys
    - Columns ending in .parameters should be JSON strings
    """
    df = df.copy()
    
    def fix_documents(value):
        if pd.isna(value) or value is None:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(value, dict):
            return [{str(k): v for k, v in value.items()}]
        if isinstance(value, list):
            return [
                {str(k): v for k, v in item.items()} if isinstance(item, dict) else item
                for item in value if item is not None
            ]
        return None
    
    def fix_parameters(value):
        if pd.isna(value) or value is None:
            return None
        if isinstance(value, str):
            try:
                json.loads(value)  # Validate JSON
                return value
            except (json.JSONDecodeError, TypeError):
                pass
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return None
    
    for col in df.columns:
        if col.endswith(".documents"):
            df[col] = df[col].apply(fix_documents)
        elif col.endswith(".parameters"):
            df[col] = df[col].apply(fix_parameters)
    
    return df


def import_traces(
    export_dir: Union[str, Path],
    space_id: str,
    arize_api_key: str,
    results_file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:

    try:
        client = Client(space_id=space_id, api_key=arize_api_key)
    except Exception as e:
        print(f"Failed to initialize Arize client: {str(e)}")
        return {}

    projects = get_projects(export_dir)

    projects_with_traces = []
    for project_name in projects:
        traces_path = Path(export_dir) / "projects" / project_name / "traces.json"
        if traces_path.exists():
            projects_with_traces.append(project_name)

    print(f"Found {len(projects_with_traces)} projects to import traces from")

    if results_file:
        results_path = Path(results_file)
    else:
        results_path = RESULTS_DIR / "trace_import_results.json"

    previous_results = {}
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                previous_results = json.load(f)
        except Exception as e:
            print(f"Error loading previous results: {str(e)}")

    results = previous_results.copy()

    for project_name in tqdm(projects_with_traces, desc="Importing traces by project"):
        if project_name in results and results[project_name].get("status") == "imported":
            print(f"Project {project_name} already imported, skipping")
            continue

        import_info = {
            "project_name": project_name,
            "status": "pending",
            "message": "",
            "trace_count": 0,
            "import_date": datetime.now().isoformat(),
        }

        try:
            df = get_project_traces_dataframe(export_dir, project_name)

            if df.empty:
                print(f"No traces found for project {project_name}, skipping")
                import_info["status"] = "skipped"
                import_info["message"] = "No traces found"
                results[project_name] = import_info
                continue

            trace_count = len(df)
            print(f"Found {trace_count} traces for project {project_name}")

            # Fix data types for Arize compatibility
            df = fix_dataframe_column_types(df)

            arize_project_name = project_name

            # Read evaluations.json for non-human evaluations
            evals_df = get_project_evaluations_dataframe(export_dir, project_name)
            evals_dataframe = None
            
            if not evals_df.empty:
                evals_dataframe = convert_evaluations_to_arize_format(evals_df)
                
                if evals_dataframe.empty or "context.span_id" not in evals_dataframe.columns:
                    if "context.span_id" not in evals_dataframe.columns:
                        print(f"Warning: evals_dataframe missing context.span_id, skipping evaluations")
                    evals_dataframe = None
                else:
                    print(f"Found {len(evals_dataframe)} evaluation records to include with traces")
            
            # Read annotations.json for human annotations
            annotations_df = get_project_annotations_dataframe(export_dir, project_name)
            human_annotations_df = None
            
            if not annotations_df.empty:
                human_annotations_df = convert_human_annotations_to_arize_format(annotations_df)
                
                if human_annotations_df.empty:
                    human_annotations_df = None
                else:
                    print(f"Found {len(human_annotations_df)} human annotation records to import separately")

            try:
                client.log_spans(
                    dataframe=df,
                    project_name=arize_project_name,
                    evals_dataframe=evals_dataframe
                )

                print(f"Successfully imported {len(df)} traces to project {arize_project_name}")
                if evals_dataframe is not None:
                    print(f"Included {len(evals_dataframe)} evaluation records")
                
                # Import human annotations after user confirmation
                if human_annotations_df is not None and not human_annotations_df.empty:
                    if "context.span_id" not in human_annotations_df.columns:
                        print(f"Warning: human_annotations_df missing context.span_id, skipping")
                    else:
                        print(f"\nPlease verify your traces appear in Arize for project '{arize_project_name}'. This may take a few minutes.")
                        user_input = input("Type 'yes' to import annotations once traces are available: ").strip().lower()
                        
                        if user_input == "yes":
                            successful_annotations = 0
                            failed_annotations = 0
                            
                            for idx, row in human_annotations_df.iterrows():
                                cleaned_row = row.dropna().to_dict()
                                
                                cleaned_row = {k: v for k, v in cleaned_row.items() 
                                             if not isinstance(v, str) or v.strip()}
                                
                                annotation_keys = [k for k in cleaned_row.keys() 
                                                 if k.startswith("annotation.") and k != "annotation.notes"]
                                if not annotation_keys:
                                    continue
                                
                                row_df = pd.DataFrame([cleaned_row])
                                
                                try:
                                    client.log_annotations(
                                        dataframe=row_df,
                                        project_name=arize_project_name,
                                        validate=False,
                                    )
                                    successful_annotations += 1
                                except Exception as ann_e:
                                    failed_annotations += 1

                            if successful_annotations > 0:
                                print(f"Successfully imported human annotation on {successful_annotations} spans")
                                import_info["human_annotations_count"] = successful_annotations
                            if failed_annotations > 0:
                                print(f"Failed to import {failed_annotations} annotation records")
                        else:
                            print("Skipping human annotations import. You can import them later.")
                
                import_info["status"] = "imported"
                import_info["message"] = f"Imported {len(df)} traces"
                import_info["trace_count"] = len(df)
                import_info["arize_project_name"] = arize_project_name
                if evals_dataframe is not None:
                    import_info["evaluations_count"] = len(evals_dataframe)

            except Exception as e:
                error_message = str(e)
                print(f"Error importing traces for project {project_name}: {error_message}")
                import_info["status"] = "failed"
                import_info["message"] = f"Error: {error_message}"

        except Exception as e:
            error_message = str(e)
            print(f"Error processing project {project_name}: {error_message}")
            import_info["status"] = "failed"
            import_info["message"] = f"Error: {error_message}"

        results[project_name] = import_info

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    imported_count = sum(info.get("status") == "imported" for info in results.values())
    print(f"Successfully imported traces from {imported_count} projects")

    return results


