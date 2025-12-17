#!/usr/bin/env python3
"""
Utility functions for Phoenix data export/import.
"""

import argparse
import logging
import os
from typing import Dict

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def parse_export_args() -> argparse.Namespace:
    """Parse command line arguments for export."""
    parser = argparse.ArgumentParser(description="Export data from Phoenix server")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all data types: datasets-experiments and traces",
    )

    parser.add_argument(
        "--datasets-experiments",
        "--de",
        action="store_true",
        dest="datasets_experiments",
        help="Export datasets and experiments",
    )

    parser.add_argument("--traces", action="store_true", help="Export traces")

    return parser.parse_args()


def create_client_with_retry(
    base_url: str,
    headers: Dict[str, str],
    timeout: float = 30.0,
    max_attempts: int = 5,
) -> httpx.Client:
    """
    Create an HTTPX client with retry capabilities.

    Args:
        base_url: Base URL for the API
        headers: HTTP headers to include in requests
        timeout: Request timeout in seconds
        max_attempts: Maximum number of retry attempts

    Returns:
        HTTPX client with retry capabilities
    """
    transport = httpx.HTTPTransport(retries=max_attempts, verify=True)
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)


def parse_import_args() -> argparse.Namespace:
    """Parse command line arguments for import."""
    parser = argparse.ArgumentParser(description="Import Phoenix export data to Arize")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Import all data types: datasets-experiments and traces",
    )

    parser.add_argument(
        "--datasets-experiments",
        "--de",
        action="store_true",
        dest="datasets_experiments",
        help="Import datasets and experiments",
    )

    parser.add_argument("--traces", action="store_true", help="Import traces")

    return parser.parse_args()
