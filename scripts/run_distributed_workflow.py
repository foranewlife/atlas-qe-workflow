#!/usr/bin/env python3
"""
Distributed ATLAS-QE Workflow Executor (refactored thin CLI)

This script now acts as a small façade that delegates to core modules:
- EosController: config validation, task creation, orchestration
- TaskProcessor: local/ssh execution of prepared tasks

Usage:
    python run_distributed_workflow.py workflow_config.yaml --resources config/resources.yaml
    python run_distributed_workflow.py workflow_config.yaml --dry-run
"""

import argparse
import logging
import sys
from typing import Optional

from src.utils.logging_config import setup_logging
from src.core.eos_controller import EosController

logger = logging.getLogger("atlas-qe-workflow")


class DistributedWorkflowExecutor:
    """Thin wrapper using new core controller."""

    def __init__(self, workflow_config_file: str, resource_config_file: Optional[str], dry_run: bool):
        self.workflow_config_file = workflow_config_file
        self.resource_config_file = resource_config_file or "config/resources.yaml"
        self.dry_run = dry_run
        self.controller = EosController(self.workflow_config_file, self.resource_config_file)

    def run(self) -> bool:
        try:
            # Validate and prepare tasks
            self.controller.validate_inputs()
            tasks = self.controller.generate_tasks()
            if self.dry_run:
                logger.info("Dry run: tasks generated; skipping execution")
                return True

            # Non-blocking execution with polling (single-thread), parallelism configurable later
            results = self.controller.execute(tasks)
            failures = [r for r in results if r.returncode != 0]
            if failures:
                logger.error(f"Completed with {len(failures)} failures out of {len(results)} tasks")
                return False
            logger.info(f"All {len(results)} tasks completed successfully")
            return True
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Distributed ATLAS-QE Workflow (simplified)")
    parser.add_argument("workflow_config", help="Path to workflow YAML (e.g., examples/test_small/test_small.yaml)")

    parser.add_argument(
        "--resources",
        default="config/resources.yaml",
        help="Path to resource configuration YAML file (default: config/resources.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate tasks and input files without executing calculations",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--log-file", help="Log file path (default: logs/distributed_workflow.log)")

    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file or "logs/distributed_workflow.log"
    setup_logging(level=args.log_level, log_file=log_file)

    try:
        executor = DistributedWorkflowExecutor(
            workflow_config_file=args.workflow_config,
            resource_config_file=args.resources,
            dry_run=args.dry_run,
        )
        success = executor.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Distributed workflow execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
