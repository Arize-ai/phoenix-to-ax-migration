#!/usr/bin/env python3
"""
Utility functions for Phoenix to Arize import operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PARENT_DIR / "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def load_json_file(file_path: Union[str, Path]) -> Optional[Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def get_projects(export_dir: Union[str, Path]) -> List[str]:
    projects_dir = Path(export_dir) / "projects"
    if not projects_dir.exists():
        logger.error(f"Projects directory not found: {projects_dir}")
        return []

    return [d.name for d in projects_dir.iterdir() if d.is_dir()]


def save_results_to_file(
    results: Any, file_path: Union[str, Path], description: str = "results"
) -> None:
    try:
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"{description.capitalize()} saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving {description} to {file_path}: {e}")
