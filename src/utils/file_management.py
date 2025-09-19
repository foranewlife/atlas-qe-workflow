"""
File management utilities for the workflow system.
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FileManager:
    """Utilities for file and directory management."""

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists, create if necessary."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def copy_files(source_patterns: List[str], dest_dir: Path):
        """Copy files matching patterns to destination."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for pattern in source_patterns:
            source_path = Path(pattern)
            if source_path.exists():
                if source_path.is_file():
                    shutil.copy2(source_path, dest_dir)
                elif source_path.is_dir():
                    shutil.copytree(source_path, dest_dir / source_path.name, dirs_exist_ok=True)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Path):
        """Save data as JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)