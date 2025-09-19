#!/usr/bin/env python3
"""
ATLAS-QE EOS Workflow Runner (Modular Version)

Multi-structure, multi-method equation of state calculation workflow
using the modular src architecture.

Usage:
    python run_eos_workflow_new.py examples/gaas_eos_study/gaas_eos_study.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.eos_workflow import EOSWorkflowRunner
from utils.logging_config import setup_logging


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ATLAS-QE EOS Workflow Runner (Modular)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_eos_workflow_new.py examples/gaas_eos_study/gaas_eos_study.yaml
    python run_eos_workflow_new.py config.yaml --max-workers 8 --verbose
    python run_eos_workflow_new.py config.yaml --dry-run
        """
    )

    parser.add_argument(
        'config_file',
        type=Path,
        help='YAML configuration file for EOS study'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Results output directory (overrides config)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate inputs but do not execute calculations'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    try:
        # Initialize and run workflow
        runner = EOSWorkflowRunner(
            config_path=args.config_file,
            results_dir=args.results_dir
        )

        if args.dry_run:
            logger.info("Dry run mode: validating configuration and inputs")
            summary = runner.get_calculation_summary()

            print(f"\n📊 Calculation Summary")
            print(f"═══════════════════════")
            print(f"Total calculations: {summary['total_calculations']}")
            print(f"Structure-combination pairs: {summary['total_combinations']}")
            print(f"Structures: {summary['total_structures']}")

            for struct_name, struct_info in summary['structures'].items():
                print(f"\n🏗️  {struct_name}:")
                print(f"   Volume points: {struct_info['volume_points']}")
                print(f"   Volume range: {struct_info['volume_range']}")
                print(f"   Combinations: {len(struct_info['combinations'])}")

            # Generate inputs only
            results = runner.run_workflow(
                max_workers=args.max_workers,
                dry_run=True
            )

            print(f"\n✅ Dry run completed!")
            print(f"Input files generated for {results.completed_tasks} calculations")

        else:
            results = runner.run_workflow(max_workers=args.max_workers)

            print(f"\n🎉 EOS Workflow Completed!")
            print(f"═══════════════════════════")
            print(f"Total calculations: {results.total_tasks}")
            print(f"Successful: {results.completed_tasks}")
            print(f"Failed: {results.failed_tasks}")
            print(f"Execution time: {results.execution_time:.1f} seconds")
            print(f"Results saved in: {runner.results_dir}")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()