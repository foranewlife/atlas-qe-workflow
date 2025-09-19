"""
Intelligent caching system for parameter set calculations.

This module provides:
- Exact parameter set matching for 100% cache hits
- Approximate matching for similar parameter sets
- Version-aware caching with software upgrade detection
- Cache space management and cleanup
- Performance analytics and hit rate tracking
"""

import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging

from .parameter_set import ParameterSet, ParameterSetManager, ParameterSetStatus

logger = logging.getLogger(__name__)


class CacheMatchType(Enum):
    """Types of cache matches available."""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class CacheMatch:
    """Represents a cache match result."""
    match_type: CacheMatchType
    parameter_set: Optional[ParameterSet]
    similarity_score: float
    reusable_data: Optional[Dict[str, Any]] = None
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_queries: int = 0
    exact_hits: int = 0
    approximate_hits: int = 0
    partial_hits: int = 0
    misses: int = 0
    space_used_gb: float = 0.0
    oldest_entry: Optional[float] = None
    newest_entry: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        hits = self.exact_hits + self.approximate_hits + self.partial_hits
        return hits / self.total_queries

    @property
    def exact_hit_rate(self) -> float:
        """Calculate exact hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.exact_hits / self.total_queries


class ParameterSetCache:
    """
    Intelligent caching system for parameter set calculations.

    Features:
    - Exact matching for identical parameter sets
    - Approximate matching for similar configurations
    - Partial matching for reusable intermediate results
    - Version tracking for software upgrades
    - Intelligent space management
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        parameter_manager: Optional[ParameterSetManager] = None,
        max_cache_size_gb: float = 50.0
    ):
        """
        Initialize the cache system.

        Args:
            cache_dir: Directory for cache storage
            parameter_manager: Parameter set manager instance
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.parameter_manager = parameter_manager or ParameterSetManager()
        self.max_cache_size_gb = max_cache_size_gb

        # Statistics tracking
        self.stats_file = self.cache_dir / "cache_stats.json"
        self.stats = self._load_statistics()

        # Software version tracking
        self.version_file = self.cache_dir / "software_versions.json"
        self.software_versions = self._load_software_versions()

    def check_computation_cache(self, param_set: ParameterSet) -> CacheMatch:
        """
        Check if a computation exists in cache.

        Args:
            param_set: Parameter set to check

        Returns:
            CacheMatch with details about cache availability
        """
        self.stats.total_queries += 1

        # First, try exact match
        exact_match = self._check_exact_match(param_set)
        if exact_match.match_type == CacheMatchType.EXACT:
            self.stats.exact_hits += 1
            self._save_statistics()
            return exact_match

        # Try approximate match
        approx_match = self._check_approximate_match(param_set)
        if approx_match.match_type == CacheMatchType.APPROXIMATE:
            self.stats.approximate_hits += 1
            self._save_statistics()
            return approx_match

        # Try partial match
        partial_match = self._check_partial_match(param_set)
        if partial_match.match_type == CacheMatchType.PARTIAL:
            self.stats.partial_hits += 1
            self._save_statistics()
            return partial_match

        # No match found
        self.stats.misses += 1
        self._save_statistics()
        return CacheMatch(
            match_type=CacheMatchType.NONE,
            parameter_set=None,
            similarity_score=0.0,
            reasons=["No similar parameter sets found in cache"]
        )

    def _check_exact_match(self, param_set: ParameterSet) -> CacheMatch:
        """Check for exact parameter set match."""
        existing = self.parameter_manager.get_parameter_set(param_set.fingerprint)

        if existing and existing.status == ParameterSetStatus.COMPLETED:
            # Verify cache files still exist
            if self._verify_cache_files(existing):
                return CacheMatch(
                    match_type=CacheMatchType.EXACT,
                    parameter_set=existing,
                    similarity_score=1.0,
                    reasons=["Exact parameter set match found"]
                )
            else:
                # Mark as failed if cache files are missing
                existing.status = ParameterSetStatus.FAILED
                self.parameter_manager.update_parameter_set(existing)

        return CacheMatch(
            match_type=CacheMatchType.NONE,
            parameter_set=None,
            similarity_score=0.0
        )

    def _check_approximate_match(self, param_set: ParameterSet) -> CacheMatch:
        """
        Check for approximate matches with similar parameter sets.

        Approximate matches can accelerate convergence by providing good
        initial guesses from similar calculations.
        """
        # Get all completed parameter sets for the same system/software/structure
        filters = {
            "system": param_set.system,
            "software": param_set.software,
            "structure": param_set.structure,
            "status": ParameterSetStatus.COMPLETED.value
        }

        candidates = self.parameter_manager.query_parameter_sets(filters)

        if not candidates:
            return CacheMatch(
                match_type=CacheMatchType.NONE,
                parameter_set=None,
                similarity_score=0.0
            )

        # Calculate similarity scores
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self._calculate_similarity_score(param_set, candidate)
            if score > best_score and score >= 0.7:  # Threshold for approximate match
                best_match = candidate
                best_score = score

        if best_match:
            return CacheMatch(
                match_type=CacheMatchType.APPROXIMATE,
                parameter_set=best_match,
                similarity_score=best_score,
                reasons=[f"Similar parameter set found (similarity: {best_score:.2f})"]
            )

        return CacheMatch(
            match_type=CacheMatchType.NONE,
            parameter_set=None,
            similarity_score=0.0
        )

    def _check_partial_match(self, param_set: ParameterSet) -> CacheMatch:
        """
        Check for partial matches where some intermediate results can be reused.

        For example, optimized structures from similar parameter sets.
        """
        # Look for structure optimizations with same system and similar parameters
        if param_set.software == "qe":  # QE can reuse optimized structures
            filters = {
                "system": param_set.system,
                "software": "qe",
                "status": ParameterSetStatus.COMPLETED.value
            }

            candidates = self.parameter_manager.query_parameter_sets(filters)

            for candidate in candidates:
                # Check if structure optimization data is available
                if (candidate.results and
                    "optimized_structure" in candidate.results and
                    self._structure_parameters_similar(param_set, candidate)):

                    return CacheMatch(
                        match_type=CacheMatchType.PARTIAL,
                        parameter_set=candidate,
                        similarity_score=0.5,
                        reusable_data={"optimized_structure": candidate.results["optimized_structure"]},
                        reasons=["Optimized structure available from similar calculation"]
                    )

        return CacheMatch(
            match_type=CacheMatchType.NONE,
            parameter_set=None,
            similarity_score=0.0
        )

    def _calculate_similarity_score(self, param_set1: ParameterSet, param_set2: ParameterSet) -> float:
        """
        Calculate similarity score between two parameter sets.

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        score = 0.0
        total_weight = 0.0

        # Basic matching (required)
        if (param_set1.system != param_set2.system or
            param_set1.software != param_set2.software or
            param_set1.structure != param_set2.structure):
            return 0.0

        # Volume parameters (weight: 0.2)
        volume_score = self._compare_volume_parameters(param_set1, param_set2)
        score += volume_score * 0.2
        total_weight += 0.2

        # Pseudopotential similarity (weight: 0.3)
        pp_score = self._compare_pseudopotentials(param_set1, param_set2)
        score += pp_score * 0.3
        total_weight += 0.3

        # Software-specific parameters (weight: 0.5)
        software_score = self._compare_software_parameters(param_set1, param_set2)
        score += software_score * 0.5
        total_weight += 0.5

        return score / total_weight if total_weight > 0 else 0.0

    def _compare_volume_parameters(self, param_set1: ParameterSet, param_set2: ParameterSet) -> float:
        """Compare volume range and points parameters."""
        # Volume range comparison
        range1 = param_set1.volume_range
        range2 = param_set2.volume_range

        range_overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
        range_union = max(range1[1], range2[1]) - min(range1[0], range2[0])
        range_score = range_overlap / range_union if range_union > 0 else 0.0

        # Volume points comparison
        points_score = 1.0 - abs(param_set1.volume_points - param_set2.volume_points) / max(param_set1.volume_points, param_set2.volume_points)

        return (range_score + points_score) / 2.0

    def _compare_pseudopotentials(self, param_set1: ParameterSet, param_set2: ParameterSet) -> float:
        """Compare pseudopotential sets."""
        if param_set1.pseudopotential_set == param_set2.pseudopotential_set:
            return 1.0

        # Compare individual files
        files1 = set(param_set1.pseudopotential_files)
        files2 = set(param_set2.pseudopotential_files)

        if not files1 or not files2:
            return 0.0

        intersection = len(files1.intersection(files2))
        union = len(files1.union(files2))

        return intersection / union if union > 0 else 0.0

    def _compare_software_parameters(self, param_set1: ParameterSet, param_set2: ParameterSet) -> float:
        """Compare software-specific parameters."""
        params1 = param_set1.parameters
        params2 = param_set2.parameters

        if param_set1.software == "atlas":
            return self._compare_atlas_parameters(params1, params2)
        elif param_set1.software == "qe":
            return self._compare_qe_parameters(params1, params2)

        return 0.0

    def _compare_atlas_parameters(self, params1: Dict, params2: Dict) -> float:
        """Compare ATLAS-specific parameters."""
        score = 0.0
        count = 0

        # Functional comparison
        if "functional" in params1 and "functional" in params2:
            score += 1.0 if params1["functional"] == params2["functional"] else 0.0
            count += 1

        # Gap parameter comparison
        if "gap" in params1 and "gap" in params2:
            gap_diff = abs(params1["gap"] - params2["gap"])
            gap_score = max(0.0, 1.0 - gap_diff / 0.1)  # 0.1 is considered significant difference
            score += gap_score
            count += 1

        return score / count if count > 0 else 0.0

    def _compare_qe_parameters(self, params1: Dict, params2: Dict) -> float:
        """Compare QE-specific parameters."""
        score = 0.0
        count = 0

        # Configuration comparison
        if "configuration" in params1 and "configuration" in params2:
            score += 1.0 if params1["configuration"] == params2["configuration"] else 0.0
            count += 1

        # K-points comparison
        if "k_points" in params1 and "k_points" in params2:
            k1 = params1["k_points"]
            k2 = params2["k_points"]
            if len(k1) >= 3 and len(k2) >= 3:
                k_diff = sum(abs(k1[i] - k2[i]) for i in range(3)) / 3.0
                k_score = max(0.0, 1.0 - k_diff / 4.0)  # 4 k-point difference is significant
                score += k_score
                count += 1

        return score / count if count > 0 else 0.0

    def _structure_parameters_similar(self, param_set1: ParameterSet, param_set2: ParameterSet) -> bool:
        """Check if structure-related parameters are similar enough for reuse."""
        # Volume ranges should overlap significantly
        range1 = param_set1.volume_range
        range2 = param_set2.volume_range

        overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
        total_range = max(range1[1], range2[1]) - min(range1[0], range2[0])

        return overlap / total_range > 0.8 if total_range > 0 else False

    def _verify_cache_files(self, param_set: ParameterSet) -> bool:
        """Verify that cached calculation files still exist."""
        if not param_set.workspace_path:
            return False

        workspace = Path(param_set.workspace_path)
        if not workspace.exists():
            return False

        # Check for essential result files
        essential_files = []
        if param_set.software == "atlas":
            essential_files = ["atlas.out", "DENSFILE"]
        elif param_set.software == "qe":
            essential_files = ["scf.out", "*.save"]

        for pattern in essential_files:
            if not list(workspace.glob(pattern)):
                return False

        return True

    def intelligent_cache_reuse(self, param_set: ParameterSet) -> Optional[Dict[str, Any]]:
        """
        Implement intelligent cache reuse strategies.

        Args:
            param_set: Target parameter set

        Returns:
            Dictionary with reusable data or None
        """
        cache_match = self.check_computation_cache(param_set)

        if cache_match.match_type == CacheMatchType.EXACT:
            return {
                "type": "exact",
                "parameter_set": cache_match.parameter_set,
                "data": cache_match.parameter_set.results
            }

        elif cache_match.match_type == CacheMatchType.APPROXIMATE:
            return {
                "type": "approximate",
                "parameter_set": cache_match.parameter_set,
                "similarity": cache_match.similarity_score,
                "data": cache_match.parameter_set.results,
                "suggestions": ["Use as initial guess", "Accelerated convergence expected"]
            }

        elif cache_match.match_type == CacheMatchType.PARTIAL:
            return {
                "type": "partial",
                "parameter_set": cache_match.parameter_set,
                "reusable_data": cache_match.reusable_data,
                "suggestions": ["Reuse optimized structure", "Skip structure optimization"]
            }

        return None

    def cleanup_cache(self, max_age_days: Optional[float] = None) -> Dict[str, Any]:
        """
        Clean up old cache entries to manage disk space.

        Args:
            max_age_days: Maximum age in days for cache entries

        Returns:
            Cleanup statistics
        """
        if max_age_days is None:
            max_age_days = 30.0  # Default: 30 days

        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        # Get old parameter sets
        all_param_sets = self.parameter_manager.query_parameter_sets()
        old_param_sets = [
            ps for ps in all_param_sets
            if ps.created_at and ps.created_at < cutoff_time
        ]

        cleaned_count = 0
        freed_space = 0.0

        for param_set in old_param_sets:
            if param_set.workspace_path:
                workspace = Path(param_set.workspace_path)
                if workspace.exists():
                    # Calculate space before deletion
                    space = sum(f.stat().st_size for f in workspace.rglob('*') if f.is_file())
                    freed_space += space

                    # Remove workspace
                    shutil.rmtree(workspace, ignore_errors=True)
                    cleaned_count += 1

                    logger.info(f"Cleaned cache for parameter set {param_set.fingerprint[:8]}")

        self._update_cache_statistics()

        return {
            "cleaned_entries": cleaned_count,
            "freed_space_gb": freed_space / (1024**3),
            "remaining_entries": len(all_param_sets) - cleaned_count
        }

    def get_cache_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        self._update_cache_statistics()
        return self.stats

    def _update_cache_statistics(self):
        """Update cache space usage statistics."""
        if self.cache_dir.exists():
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            self.stats.space_used_gb = total_size / (1024**3)

        # Update entry timestamps
        all_param_sets = self.parameter_manager.query_parameter_sets()
        if all_param_sets:
            timestamps = [ps.created_at for ps in all_param_sets if ps.created_at]
            if timestamps:
                self.stats.oldest_entry = min(timestamps)
                self.stats.newest_entry = max(timestamps)

    def _load_statistics(self) -> CacheStatistics:
        """Load cache statistics from file."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                return CacheStatistics(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache statistics: {e}")

        return CacheStatistics()

    def _save_statistics(self):
        """Save cache statistics to file."""
        try:
            with open(self.stats_file, 'w') as f:
                # Convert to dict, handling non-serializable fields
                data = {
                    "total_queries": self.stats.total_queries,
                    "exact_hits": self.stats.exact_hits,
                    "approximate_hits": self.stats.approximate_hits,
                    "partial_hits": self.stats.partial_hits,
                    "misses": self.stats.misses,
                    "space_used_gb": self.stats.space_used_gb,
                    "oldest_entry": self.stats.oldest_entry,
                    "newest_entry": self.stats.newest_entry
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache statistics: {e}")

    def _load_software_versions(self) -> Dict[str, str]:
        """Load tracked software versions."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load software versions: {e}")

        return {}

    def _save_software_versions(self):
        """Save current software versions."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.software_versions, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save software versions: {e}")

    def invalidate_cache_for_software_upgrade(self, software: str, new_version: str):
        """
        Invalidate cache entries when software is upgraded.

        Args:
            software: Software name ("atlas" or "qe")
            new_version: New software version
        """
        old_version = self.software_versions.get(software)

        if old_version and old_version != new_version:
            # Mark all parameter sets for this software as requiring recalculation
            filters = {"software": software, "status": ParameterSetStatus.COMPLETED.value}
            affected_param_sets = self.parameter_manager.query_parameter_sets(filters)

            for param_set in affected_param_sets:
                param_set.status = ParameterSetStatus.PENDING
                self.parameter_manager.update_parameter_set(param_set)

            logger.info(f"Invalidated {len(affected_param_sets)} cache entries for {software} upgrade {old_version} -> {new_version}")

        # Update version tracking
        self.software_versions[software] = new_version
        self._save_software_versions()