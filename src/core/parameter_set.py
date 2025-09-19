"""
Parameter set management system for ATLAS-QE workflow.

This module provides comprehensive parameter set management including:
- Unique fingerprinting for parameter combinations
- State tracking (pending, running, completed, failed)
- Intelligent parameter space enumeration
- Database persistence and querying
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ParameterSetStatus(Enum):
    """Status of a parameter set calculation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ParameterSet:
    """
    Represents a complete parameter set for a materials calculation.

    This includes all parameters needed to uniquely identify and execute
    a calculation: software, structure, computational parameters, and
    pseudopotentials.
    """

    # Core identification
    system: str  # e.g., "Mg", "MgAl"
    software: str  # "atlas" or "qe"
    structure: str  # "fcc", "bcc", "diamond", etc.

    # Software-specific parameters
    parameters: Dict[str, Any]  # Software-specific computation parameters
    pseudopotential_set: str  # Name of the pseudopotential combination
    pseudopotential_files: List[str]  # Actual pseudopotential files

    # Structural parameters
    volume_points: int = 11
    volume_range: Tuple[float, float] = (0.8, 1.2)

    # Metadata
    fingerprint: Optional[str] = None
    status: ParameterSetStatus = ParameterSetStatus.PENDING
    created_at: Optional[float] = None
    updated_at: Optional[float] = None

    # Results storage
    workspace_path: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Generate fingerprint and set timestamps if not provided."""
        if self.fingerprint is None:
            self.fingerprint = self.generate_fingerprint()
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = self.created_at

    def generate_fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this parameter set.

        The fingerprint is based on all calculation-relevant parameters,
        ensuring that identical calculations can be detected and cached.

        Returns:
            str: SHA256 hash of the parameter combination
        """
        # Create a canonical representation of all parameters
        fingerprint_data = {
            "system": self.system,
            "software": self.software,
            "structure": self.structure,
            "parameters": self.parameters,
            "pseudopotential_set": self.pseudopotential_set,
            "pseudopotential_files": sorted(self.pseudopotential_files),
            "volume_points": self.volume_points,
            "volume_range": self.volume_range,
        }

        # Convert to canonical JSON and hash
        canonical_json = json.dumps(fingerprint_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def get_workspace_name(self) -> str:
        """
        Generate a human-readable workspace name.

        Format: {software}_{structure}_{parameters_summary}_{pseudopotential_set}
        Example: atlas_fcc_kedf701_gap020_lda_combo
        """
        # Extract key parameters for naming
        param_summary = self._get_parameter_summary()

        workspace_name = f"{self.software}_{self.structure}_{param_summary}_{self.pseudopotential_set}"

        # Clean up name (replace problematic characters)
        workspace_name = workspace_name.replace(".", "").replace(" ", "_").replace("/", "_")

        return workspace_name

    def _get_parameter_summary(self) -> str:
        """Extract key parameters for workspace naming."""
        if self.software == "atlas":
            # ATLAS: functional, gap
            functional = self.parameters.get("functional", "")
            gap = self.parameters.get("gap", "")
            gap_str = str(gap).replace('.', '')
            # Ensure at least 3 digits for gap
            if len(gap_str) == 2:  # e.g., "02" -> "020"
                gap_str += "0"
            return f"{functional}_gap{gap_str}"

        elif self.software == "qe":
            # QE: configuration, k_points
            config = self.parameters.get("configuration", "")
            k_points = self.parameters.get("k_points", [])
            if k_points:
                k_str = "k" + "".join(map(str, k_points[0:3]))  # Use first 3 values
            else:
                k_str = "k888"
            return f"{config}_{k_str}"

        return "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSet":
        """Create from dictionary (deserialization)."""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ParameterSetStatus(data["status"])
        return cls(**data)


class ParameterSetManager:
    """
    Manages parameter sets with database persistence and intelligent querying.

    Features:
    - SQLite database for parameter set storage
    - Efficient querying and filtering
    - Parameter space enumeration
    - Missing calculation detection
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the parameter set manager.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            db_path = Path("data/parameter_sets.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_sets (
                    fingerprint TEXT PRIMARY KEY,
                    system TEXT NOT NULL,
                    software TEXT NOT NULL,
                    structure TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    pseudopotential_set TEXT NOT NULL,
                    pseudopotential_files TEXT NOT NULL,
                    volume_points INTEGER NOT NULL,
                    volume_range TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    workspace_path TEXT,
                    results TEXT
                )
            """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system ON parameter_sets(system)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_software ON parameter_sets(software)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON parameter_sets(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pseudopotential_set ON parameter_sets(pseudopotential_set)")

            conn.commit()

    def add_parameter_set(self, param_set: ParameterSet) -> bool:
        """
        Add a parameter set to the database.

        Args:
            param_set: The parameter set to add

        Returns:
            bool: True if added successfully, False if already exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = param_set.to_dict()

                conn.execute("""
                    INSERT INTO parameter_sets (
                        fingerprint, system, software, structure, parameters,
                        pseudopotential_set, pseudopotential_files, volume_points,
                        volume_range, status, created_at, updated_at,
                        workspace_path, results
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["fingerprint"],
                    data["system"],
                    data["software"],
                    data["structure"],
                    json.dumps(data["parameters"]),
                    data["pseudopotential_set"],
                    json.dumps(data["pseudopotential_files"]),
                    data["volume_points"],
                    json.dumps(data["volume_range"]),
                    data["status"],
                    data["created_at"],
                    data["updated_at"],
                    data.get("workspace_path"),
                    json.dumps(data.get("results")) if data.get("results") else None
                ))

                conn.commit()
                logger.info(f"Added parameter set {param_set.fingerprint[:8]}")
                return True

        except sqlite3.IntegrityError:
            logger.warning(f"Parameter set {param_set.fingerprint[:8]} already exists")
            return False

    def get_parameter_set(self, fingerprint: str) -> Optional[ParameterSet]:
        """
        Retrieve a parameter set by fingerprint.

        Args:
            fingerprint: The parameter set fingerprint

        Returns:
            ParameterSet if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM parameter_sets WHERE fingerprint = ?",
                (fingerprint,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_parameter_set(row)

    def update_parameter_set(self, param_set: ParameterSet):
        """
        Update an existing parameter set in the database.

        Args:
            param_set: The parameter set to update
        """
        param_set.updated_at = time.time()

        with sqlite3.connect(self.db_path) as conn:
            data = param_set.to_dict()

            conn.execute("""
                UPDATE parameter_sets SET
                    status = ?, updated_at = ?, workspace_path = ?, results = ?
                WHERE fingerprint = ?
            """, (
                data["status"],
                data["updated_at"],
                data.get("workspace_path"),
                json.dumps(data.get("results")) if data.get("results") else None,
                data["fingerprint"]
            ))

            conn.commit()
            logger.info(f"Updated parameter set {param_set.fingerprint[:8]}")

    def query_parameter_sets(self, filters: Optional[Dict[str, Any]] = None) -> List[ParameterSet]:
        """
        Query parameter sets with flexible filtering.

        Args:
            filters: Dictionary of field:value filters

        Returns:
            List of matching parameter sets
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM parameter_sets"
            params = []

            if filters:
                conditions = []
                for field, value in filters.items():
                    if field in ["system", "software", "structure", "pseudopotential_set", "status"]:
                        conditions.append(f"{field} = ?")
                        params.append(value)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_parameter_set(row) for row in rows]

    def get_missing_parameter_sets(self, target_space: List[ParameterSet]) -> List[ParameterSet]:
        """
        Identify parameter sets that are missing from the database.

        Args:
            target_space: List of all desired parameter sets

        Returns:
            List of parameter sets that need to be calculated
        """
        existing_fingerprints = set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT fingerprint FROM parameter_sets")
            existing_fingerprints = {row[0] for row in cursor.fetchall()}

        missing = []
        for param_set in target_space:
            if param_set.fingerprint not in existing_fingerprints:
                missing.append(param_set)

        logger.info(f"Found {len(missing)} missing parameter sets out of {len(target_space)} total")
        return missing

    def _row_to_parameter_set(self, row: sqlite3.Row) -> ParameterSet:
        """Convert database row to ParameterSet object."""
        return ParameterSet(
            system=row["system"],
            software=row["software"],
            structure=row["structure"],
            parameters=json.loads(row["parameters"]),
            pseudopotential_set=row["pseudopotential_set"],
            pseudopotential_files=json.loads(row["pseudopotential_files"]),
            volume_points=row["volume_points"],
            volume_range=tuple(json.loads(row["volume_range"])),
            fingerprint=row["fingerprint"],
            status=ParameterSetStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            workspace_path=row["workspace_path"],
            results=json.loads(row["results"]) if row["results"] else None
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    status,
                    COUNT(*) as count
                FROM parameter_sets
                GROUP BY status
            """)

            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = conn.execute("SELECT COUNT(*) FROM parameter_sets")
            total_count = cursor.fetchone()[0]

            return {
                "total_parameter_sets": total_count,
                "status_breakdown": status_counts,
                "database_path": str(self.db_path)
            }