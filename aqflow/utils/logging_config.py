"""
Logging configuration for the workflow system.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional

LOG_PATH = Path("aqflow_data") / "aqflow.log"

class _FastFileHandler(logging.FileHandler):
    """File handler that can fsync on flush to minimize latency.

    Note: fsync improves visibility at the cost of I/O overhead. Use when
    realtime log tailing is desired and volume is low.
    """

    def __init__(self, filename, mode="a", encoding=None, delay=False, *, fsync=False):
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        self._fsync = bool(fsync)

    def flush(self):
        try:
            super().flush()
            if self._fsync and self.stream and hasattr(self.stream, "fileno"):
                os.fsync(self.stream.fileno())
        except Exception:
            # Never let logging flush raise
            pass


def _env_flag(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v not in ("0", "false", "False", "no", "NO", "")


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

    # Determine log file path: default to aqflow_data/aqflow.log
    if log_file is None:
        log_path = LOG_PATH
    else:
        log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    # Clear existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler (always on)
    low_latency = _env_flag("AQFLOW_LOG_LOW_LATENCY")
    force_fsync = _env_flag("AQFLOW_LOG_FSYNC")
    line_buffered = _env_flag("AQFLOW_LOG_LINE_BUFFERED")

    if low_latency and line_buffered:
        # Use a line-buffered stream with StreamHandler for minimal delay
        try:
            stream = open(log_path, mode="a", buffering=1, encoding="utf-8")
        except Exception:
            stream = open(log_path, mode="a", encoding="utf-8")
        fh = logging.StreamHandler(stream)
    else:
        # Default FileHandler; optionally fsync on flush
        fh = _FastFileHandler(log_path, fsync=force_fsync or low_latency)

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
