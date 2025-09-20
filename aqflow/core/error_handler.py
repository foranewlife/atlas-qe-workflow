"""
Comprehensive error handling and reporting system.

This module provides intelligent error detection, classification,
recovery strategies, and detailed error reporting for the
distributed computing system.
"""

import re
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors in the system."""
    SOFTWARE_EXECUTION = "software_execution"
    CONVERGENCE_FAILURE = "convergence_failure"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    FILE_SYSTEM_ERROR = "file_system_error"
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Possible recovery actions."""
    RETRY = "retry"
    RETRY_WITH_DIFFERENT_RESOURCE = "retry_different_resource"
    RETRY_WITH_MODIFIED_PARAMETERS = "retry_modified_parameters"
    SKIP_TASK = "skip_task"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class ErrorPattern:
    """Error pattern for detection."""
    category: ErrorCategory
    severity: ErrorSeverity
    patterns: List[str]
    recovery_action: RecoveryAction
    description: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str
    timestamp: float = field(default_factory=time.time)
    category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    description: str = ""

    # Context information
    task_id: Optional[str] = None
    software: Optional[str] = None
    resource_name: Optional[str] = None
    working_directory: Optional[str] = None

    # Error details
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    log_files: List[str] = field(default_factory=list)

    # Analysis results
    matched_patterns: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    recovery_action: RecoveryAction = RecoveryAction.RETRY
    suggestions: List[str] = field(default_factory=list)

    # Recovery tracking
    retry_count: int = 0
    resolved: bool = False
    resolution_notes: Optional[str] = None


class ErrorHandler:
    """
    Comprehensive error handling and recovery system.

    Provides intelligent error detection, classification, and
    automated recovery strategies for distributed computing tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error handler.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.error_patterns: List[ErrorPattern] = []
        self.error_reports: Dict[str, ErrorReport] = {}

        # Error handling settings
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delays = self.config.get('retry_delays', [60, 300, 900])  # 1min, 5min, 15min
        self.enable_auto_recovery = self.config.get('enable_auto_recovery', True)

        # Initialize error patterns
        self._load_error_patterns()

        logger.info("ErrorHandler initialized")

    def analyze_error(
        self,
        task_id: str,
        software: str,
        resource_name: str,
        working_directory: str,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
        log_files: Optional[List[str]] = None
    ) -> ErrorReport:
        """
        Analyze an error and generate comprehensive error report.

        Args:
            task_id: Task identifier
            software: Software that failed
            resource_name: Resource where failure occurred
            working_directory: Working directory
            exit_code: Process exit code
            stdout: Standard output
            stderr: Standard error
            log_files: Additional log files to analyze

        Returns:
            Comprehensive error report
        """
        try:
            # Generate unique error ID
            error_id = f"error_{task_id}_{int(time.time() * 1000)}"

            # Create initial error report
            error_report = ErrorReport(
                error_id=error_id,
                task_id=task_id,
                software=software,
                resource_name=resource_name,
                working_directory=working_directory,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                log_files=log_files or []
            )

            # Collect additional log content
            self._collect_log_content(error_report)

            # Analyze error patterns
            self._classify_error(error_report)

            # Determine recovery strategy
            self._determine_recovery_strategy(error_report)

            # Store error report
            self.error_reports[error_id] = error_report

            logger.error(f"Error analyzed: {error_id} - {error_report.category.value} - {error_report.description}")

            return error_report

        except Exception as e:
            logger.error(f"Error analyzing failure: {e}")
            # Return minimal error report
            return ErrorReport(
                error_id=f"error_{task_id}_fallback",
                task_id=task_id,
                software=software,
                resource_name=resource_name,
                description=f"Error analysis failed: {str(e)}"
            )

    def get_recovery_strategy(self, error_report: ErrorReport) -> Dict[str, Any]:
        """
        Get detailed recovery strategy for an error.

        Args:
            error_report: Error report to analyze

        Returns:
            Recovery strategy details
        """
        strategy = {
            'action': error_report.recovery_action.value,
            'retry_recommended': error_report.recovery_action in [
                RecoveryAction.RETRY,
                RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
                RecoveryAction.RETRY_WITH_MODIFIED_PARAMETERS
            ],
            'retry_delay': self._get_retry_delay(error_report.retry_count),
            'max_retries_reached': error_report.retry_count >= self.max_retries,
            'suggestions': error_report.suggestions,
            'parameter_modifications': self._get_parameter_modifications(error_report),
            'resource_recommendations': self._get_resource_recommendations(error_report)
        }

        return strategy

    def should_retry(self, error_report: ErrorReport) -> bool:
        """
        Determine if a task should be retried based on error analysis.

        Args:
            error_report: Error report

        Returns:
            True if task should be retried
        """
        if not self.enable_auto_recovery:
            return False

        if error_report.retry_count >= self.max_retries:
            return False

        # Never retry certain error types
        no_retry_categories = [
            ErrorCategory.CONFIGURATION_ERROR,
            ErrorCategory.FILE_SYSTEM_ERROR
        ]

        if error_report.category in no_retry_categories:
            return False

        # Check if recovery action allows retry
        retry_actions = [
            RecoveryAction.RETRY,
            RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
            RecoveryAction.RETRY_WITH_MODIFIED_PARAMETERS
        ]

        return error_report.recovery_action in retry_actions

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        try:
            total_errors = len(self.error_reports)

            if total_errors == 0:
                return {
                    'total_errors': 0,
                    'by_category': {},
                    'by_severity': {},
                    'by_software': {},
                    'resolution_rate': 0.0,
                    'avg_retry_count': 0.0
                }

            # Count by category
            category_counts = {}
            for report in self.error_reports.values():
                category = report.category.value
                category_counts[category] = category_counts.get(category, 0) + 1

            # Count by severity
            severity_counts = {}
            for report in self.error_reports.values():
                severity = report.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Count by software
            software_counts = {}
            for report in self.error_reports.values():
                if report.software:
                    software_counts[report.software] = software_counts.get(report.software, 0) + 1

            # Calculate resolution rate
            resolved_count = sum(1 for report in self.error_reports.values() if report.resolved)
            resolution_rate = (resolved_count / total_errors) * 100

            # Calculate average retry count
            total_retries = sum(report.retry_count for report in self.error_reports.values())
            avg_retry_count = total_retries / total_errors

            return {
                'total_errors': total_errors,
                'by_category': category_counts,
                'by_severity': severity_counts,
                'by_software': software_counts,
                'resolution_rate': resolution_rate,
                'avg_retry_count': avg_retry_count,
                'resolved_errors': resolved_count,
                'unresolved_errors': total_errors - resolved_count
            }

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}

    def mark_error_resolved(self, error_id: str, resolution_notes: str = ""):
        """Mark an error as resolved."""
        if error_id in self.error_reports:
            self.error_reports[error_id].resolved = True
            self.error_reports[error_id].resolution_notes = resolution_notes
            logger.info(f"Marked error {error_id} as resolved")

    def _collect_log_content(self, error_report: ErrorReport):
        """Collect content from log files."""
        try:
            work_dir = Path(error_report.working_directory)

            # Common log files to check
            common_logs = [
                'job.log', 'job.out', 'job.err',
                'atlas.out', 'atlas.log',
                'pw.out', 'scf.out',
                'error.log', 'stderr.txt', 'stdout.txt'
            ]

            for log_name in common_logs:
                log_file = work_dir / log_name
                if log_file.exists():
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()

                        # Add to stderr for analysis
                        if log_name not in ['stdout.txt']:
                            error_report.stderr += f"\n--- {log_name} ---\n{content}\n"
                        else:
                            error_report.stdout += f"\n--- {log_name} ---\n{content}\n"

                        error_report.log_files.append(str(log_file))

                    except Exception as e:
                        logger.warning(f"Could not read log file {log_file}: {e}")

        except Exception as e:
            logger.error(f"Error collecting log content: {e}")

    def _classify_error(self, error_report: ErrorReport):
        """Classify error based on patterns and content."""
        try:
            # Combine all error text for analysis
            error_text = f"{error_report.stderr}\n{error_report.stdout}".lower()

            # Check against known patterns
            for pattern in self.error_patterns:
                for pattern_text in pattern.patterns:
                    if re.search(pattern_text.lower(), error_text):
                        error_report.category = pattern.category
                        error_report.severity = pattern.severity
                        error_report.recovery_action = pattern.recovery_action
                        error_report.description = pattern.description
                        error_report.suggestions.extend(pattern.suggestions)
                        error_report.matched_patterns.append(pattern_text)
                        return

            # Fallback classification based on exit code
            self._classify_by_exit_code(error_report)

        except Exception as e:
            logger.error(f"Error classifying error: {e}")

    def _classify_by_exit_code(self, error_report: ErrorReport):
        """Classify error based on exit code."""
        exit_code = error_report.exit_code

        if exit_code == 0:
            # Success code but marked as error - likely convergence issue
            error_report.category = ErrorCategory.CONVERGENCE_FAILURE
            error_report.severity = ErrorSeverity.MEDIUM
            error_report.description = "Calculation completed but did not converge"
            error_report.recovery_action = RecoveryAction.RETRY_WITH_MODIFIED_PARAMETERS

        elif exit_code in [1, 2]:
            # Common error codes
            error_report.category = ErrorCategory.SOFTWARE_EXECUTION
            error_report.severity = ErrorSeverity.MEDIUM
            error_report.description = f"Software execution failed with exit code {exit_code}"
            error_report.recovery_action = RecoveryAction.RETRY

        elif exit_code in [124, 125, 126, 127]:
            # System-level errors
            error_report.category = ErrorCategory.RESOURCE_ERROR
            error_report.severity = ErrorSeverity.HIGH
            error_report.description = f"System error: exit code {exit_code}"
            error_report.recovery_action = RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE

        elif exit_code == 137:
            # SIGKILL - likely out of memory
            error_report.category = ErrorCategory.RESOURCE_ERROR
            error_report.severity = ErrorSeverity.HIGH
            error_report.description = "Process killed (likely out of memory)"
            error_report.recovery_action = RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE
            error_report.suggestions.append("Try using a resource with more memory")

        elif exit_code == 139:
            # SIGSEGV - segmentation fault
            error_report.category = ErrorCategory.SOFTWARE_EXECUTION
            error_report.severity = ErrorSeverity.HIGH
            error_report.description = "Segmentation fault in software"
            error_report.recovery_action = RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE

        else:
            # Unknown exit code
            error_report.category = ErrorCategory.UNKNOWN_ERROR
            error_report.severity = ErrorSeverity.MEDIUM
            error_report.description = f"Unknown error with exit code {exit_code}"
            error_report.recovery_action = RecoveryAction.RETRY

    def _determine_recovery_strategy(self, error_report: ErrorReport):
        """Determine detailed recovery strategy."""
        try:
            # Add specific suggestions based on software and error type
            if error_report.software == 'atlas':
                self._add_atlas_specific_suggestions(error_report)
            elif error_report.software == 'qe':
                self._add_qe_specific_suggestions(error_report)

            # Add general suggestions based on error category
            self._add_category_specific_suggestions(error_report)

        except Exception as e:
            logger.error(f"Error determining recovery strategy: {e}")

    def _add_atlas_specific_suggestions(self, error_report: ErrorReport):
        """Add ATLAS-specific recovery suggestions."""
        if error_report.category == ErrorCategory.CONVERGENCE_FAILURE:
            error_report.suggestions.extend([
                "Reduce GAP parameter for finer grid",
                "Increase ScfIter for more iterations",
                "Try different KEDF functional",
                "Check initial guess quality"
            ])
        elif error_report.category == ErrorCategory.RESOURCE_ERROR:
            error_report.suggestions.extend([
                "Increase GAP parameter to reduce memory usage",
                "Use fewer grid points",
                "Try on resource with more memory"
            ])

    def _add_qe_specific_suggestions(self, error_report: ErrorReport):
        """Add QE-specific recovery suggestions."""
        if error_report.category == ErrorCategory.CONVERGENCE_FAILURE:
            error_report.suggestions.extend([
                "Reduce mixing_beta parameter",
                "Increase electron_maxstep",
                "Try different mixing_mode",
                "Reduce degauss parameter",
                "Check k-point sampling"
            ])
        elif error_report.category == ErrorCategory.RESOURCE_ERROR:
            error_report.suggestions.extend([
                "Reduce ecutwfc parameter",
                "Use fewer k-points",
                "Try on resource with more cores",
                "Enable disk_io = 'low'"
            ])

    def _add_category_specific_suggestions(self, error_report: ErrorReport):
        """Add suggestions based on error category."""
        if error_report.category == ErrorCategory.NETWORK_ERROR:
            error_report.suggestions.extend([
                "Check network connectivity",
                "Verify SSH key permissions",
                "Try on local resource instead"
            ])
        elif error_report.category == ErrorCategory.FILE_SYSTEM_ERROR:
            error_report.suggestions.extend([
                "Check disk space availability",
                "Verify file permissions",
                "Check filesystem mount status"
            ])

    def _get_retry_delay(self, retry_count: int) -> int:
        """Get delay for retry based on retry count."""
        if retry_count < len(self.retry_delays):
            return self.retry_delays[retry_count]
        else:
            return self.retry_delays[-1]  # Use last delay for further retries

    def _get_parameter_modifications(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Get recommended parameter modifications."""
        modifications = {}

        if error_report.software == 'atlas':
            if error_report.category == ErrorCategory.CONVERGENCE_FAILURE:
                modifications = {
                    'GAP': 'reduce by 0.02',
                    'ScfIter': 'increase to 500',
                    'scftol': 'relax to 1e-6'
                }
            elif error_report.category == ErrorCategory.RESOURCE_ERROR:
                modifications = {
                    'GAP': 'increase by 0.05',
                    'LCELL': 'reduce by 2'
                }

        elif error_report.software == 'qe':
            if error_report.category == ErrorCategory.CONVERGENCE_FAILURE:
                modifications = {
                    'mixing_beta': 'reduce to 0.1',
                    'electron_maxstep': 'increase to 200',
                    'conv_thr': 'relax to 1e-6'
                }
            elif error_report.category == ErrorCategory.RESOURCE_ERROR:
                modifications = {
                    'ecutwfc': 'reduce by 10 Ry',
                    'disk_io': 'set to minimal'
                }

        return modifications

    def _get_resource_recommendations(self, error_report: ErrorReport) -> List[str]:
        """Get resource recommendations for retry."""
        recommendations = []

        if error_report.category == ErrorCategory.RESOURCE_ERROR:
            recommendations.extend([
                "Use resource with more memory",
                "Use resource with more CPU cores",
                "Prefer local over remote resources"
            ])
        elif error_report.category == ErrorCategory.NETWORK_ERROR:
            recommendations.extend([
                "Use local resource only",
                "Check network connectivity before retry"
            ])
        elif error_report.category == ErrorCategory.CONVERGENCE_FAILURE:
            recommendations.extend([
                "Try different resource type",
                "Use resource with better numerical libraries"
            ])

        return recommendations

    def _load_error_patterns(self):
        """Load error patterns from configuration."""
        # ATLAS error patterns
        atlas_patterns = [
            ErrorPattern(
                category=ErrorCategory.CONVERGENCE_FAILURE,
                severity=ErrorSeverity.MEDIUM,
                patterns=[
                    r"Maximum SCF iterations exceeded",
                    r"SCF not converged",
                    r"Energy not decreasing"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_MODIFIED_PARAMETERS,
                description="ATLAS SCF convergence failure",
                suggestions=["Increase ScfIter", "Adjust mixing parameters", "Reduce GAP"]
            ),
            ErrorPattern(
                category=ErrorCategory.RESOURCE_ERROR,
                severity=ErrorSeverity.HIGH,
                patterns=[
                    r"Memory allocation failed",
                    r"Out of memory",
                    r"Cannot allocate memory"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
                description="ATLAS memory allocation failure",
                suggestions=["Use resource with more memory", "Increase GAP parameter"]
            ),
            ErrorPattern(
                category=ErrorCategory.SOFTWARE_EXECUTION,
                severity=ErrorSeverity.HIGH,
                patterns=[
                    r"FATAL ERROR",
                    r"Segmentation fault",
                    r"Bus error"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
                description="ATLAS fatal execution error",
                suggestions=["Try different resource", "Check input file validity"]
            )
        ]

        # QE error patterns
        qe_patterns = [
            ErrorPattern(
                category=ErrorCategory.CONVERGENCE_FAILURE,
                severity=ErrorSeverity.MEDIUM,
                patterns=[
                    r"convergence NOT achieved",
                    r"charge is wrong",
                    r"too many bands are not converged"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_MODIFIED_PARAMETERS,
                description="QE SCF convergence failure",
                suggestions=["Reduce mixing_beta", "Increase electron_maxstep", "Try different mixing_mode"]
            ),
            ErrorPattern(
                category=ErrorCategory.RESOURCE_ERROR,
                severity=ErrorSeverity.HIGH,
                patterns=[
                    r"not enough memory",
                    r"allocate.*failed",
                    r"virtual memory exhausted"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
                description="QE memory allocation failure",
                suggestions=["Use resource with more memory", "Reduce ecutwfc", "Use fewer k-points"]
            ),
            ErrorPattern(
                category=ErrorCategory.FILE_SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                patterns=[
                    r"No space left on device",
                    r"Permission denied",
                    r"cannot create.*file"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
                description="QE file system error",
                suggestions=["Check disk space", "Check file permissions", "Try different resource"]
            )
        ]

        # Network and system error patterns
        system_patterns = [
            ErrorPattern(
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.HIGH,
                patterns=[
                    r"ssh.*connection refused",
                    r"ssh.*timeout",
                    r"rsync.*error",
                    r"Connection closed by remote host"
                ],
                recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_RESOURCE,
                description="Network/SSH connection error",
                suggestions=["Check network connectivity", "Verify SSH configuration", "Use local resource"]
            ),
            ErrorPattern(
                category=ErrorCategory.TIMEOUT_ERROR,
                severity=ErrorSeverity.MEDIUM,
                patterns=[
                    r"timeout",
                    r"time limit exceeded",
                    r"wall time exceeded"
                ],
                recovery_action=RecoveryAction.RETRY,
                description="Job timeout",
                suggestions=["Increase timeout limit", "Optimize calculation parameters"]
            )
        ]

        # Combine all patterns
        self.error_patterns.extend(atlas_patterns)
        self.error_patterns.extend(qe_patterns)
        self.error_patterns.extend(system_patterns)

        logger.info(f"Loaded {len(self.error_patterns)} error patterns")