"""
Logging Setup Module
====================
Provides centralized logging configuration for the SGCC pipeline.
Uses the YAML-based logging config with a fallback to basic setup.
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


def _find_project_root() -> Path:
    """
    Locate the project root by searching for config/logging_config.yaml
    starting from this file's location and traversing upward.

    Returns:
        Path: The project root directory.
    """
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Traverse up to 5 levels
        if (current / "config" / "logging_config.yaml").exists():
            return current
        current = current.parent
    # Fallback: assume project root is 3 levels up from this file
    return Path(__file__).resolve().parent.parent.parent


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Initialize logging from the YAML configuration file.

    Args:
        config_path: Optional path to the logging YAML config.
                     If None, auto-discovers from project root.
    """
    project_root = _find_project_root()

    if config_path is None:
        config_path = str(project_root / "config" / "logging_config.yaml")

    # Ensure log directory exists
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "r") as f:
            log_config = yaml.safe_load(f)

        # Fix file handler paths to be absolute
        for handler_name, handler_cfg in log_config.get("handlers", {}).items():
            if "filename" in handler_cfg:
                handler_cfg["filename"] = str(
                    project_root / handler_cfg["filename"]
                )

        logging.config.dictConfig(log_config)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        # Fallback to basic logging if config is unavailable
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger(__name__).warning(
            "Could not load logging config from '%s': %s. Using basic config.",
            config_path,
            e,
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance. Automatically sets up logging on first call.

    Args:
        name: Logger name, typically the module's __name__ or a dotted path
              like 'sgcc.data.loader'.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger("sgcc.models.trainer")
        >>> logger.info("Training started with %d samples", n_samples)
    """
    # Ensure logging is configured (idempotent after first call)
    if not logging.getLogger("sgcc").handlers:
        setup_logging()

    return logging.getLogger(name)
