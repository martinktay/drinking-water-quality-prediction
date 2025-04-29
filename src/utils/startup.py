"""Startup check utilities for the application."""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple


def check_directories() -> List[str]:
    """
    Check if all required directories exist.

    Returns:
        List of missing directories
    """
    required_dirs = [
        'data/raw',
        'data/processed',
        'models',
        'reports/model_performance',
        'logs'
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            # Create the directory
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    return missing_dirs


def check_config() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if config file exists and is valid.

    Returns:
        Tuple of (is_valid, config_dict)
    """
    config_path = Path('config.yaml')
    if not config_path.exists():
        return False, {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['models', 'preprocessor', 'paths', 'api']
        for section in required_sections:
            if section not in config:
                return False, config

        return True, config
    except Exception:
        return False, {}


def check_processed_data() -> bool:
    """
    Check if processed data file exists.

    Returns:
        True if file exists, False otherwise
    """
    return Path('data/processed/processed_data.csv').exists()


def check_environment() -> List[str]:
    """
    Check if all required environment variables are set.

    Returns:
        List of missing environment variables
    """
    required_vars = [
        'PYTHONPATH'  # Add any other required environment variables
    ]

    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)

    return missing_vars


def perform_startup_checks() -> List[str]:
    """
    Perform all startup checks and return any warnings.

    Returns:
        List of warning messages
    """
    warnings = []

    # Check directories
    missing_dirs = check_directories()
    if missing_dirs:
        warnings.append(f"Created missing directories: {', '.join(missing_dirs)}")

    # Check config
    config_valid, _ = check_config()
    if not config_valid:
        warnings.append("Configuration file is missing or invalid")

    # Check processed data
    if not check_processed_data():
        warnings.append(
            "Processed data file is missing. Please run data processing pipeline first")

    # Check environment
    missing_vars = check_environment()
    if missing_vars:
        warnings.append(f"Missing environment variables: {', '.join(missing_vars)}")

    return warnings
