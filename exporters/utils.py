#!/usr/bin/env python3
"""
Phoenix Exporters Utility Functions
"""

import json
from typing import Dict, List, Union

import httpx


def save_json(data: Union[Dict, List], filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def get_projects(client: httpx.Client) -> List[Dict]:
    response = client.get("/v1/projects")
    response.raise_for_status()
    return response.json().get("data", [])
