"""
Logging configuration for the workflow system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    console_level: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format_string: Custom format string

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Determine log file path: default to logs/aqflow.log
    if log_file is None:
        log_path = Path("logs") / "aqflow.log"
    else:
        log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    # Clear existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler (always on)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(format_string))
    fh.setLevel(getattr(logging, level.upper()))
    root.addHandler(fh)

    # Console handler (quiet by default)
    ch = logging.StreamHandler(sys.stdout)
    ch_level = console_level or "WARNING"
    ch.setLevel(getattr(logging, ch_level.upper()))
    ch.setFormatter(logging.Formatter(format_string))
    root.addHandler(ch)

    return logging.getLogger("atlas-qe-workflow")
