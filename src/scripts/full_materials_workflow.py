#!/usr/bin/env python3
"""
Full materials workflow entry point.

This script provides the main entry point for complete ATLAS-QE materials
calculation workflows, including EOS studies, structure optimization,
and convergence testing.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path for development mode
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.eos_workflow import EOSWorkflowRunner
from utils.logging_config import setup_logging


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="ATLAS-QE Full Materials Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete EOS workflow
    atlas-qe-workflow eos examples/gaas_eos_study/gaas_eos_study.yaml

    # Run with custom settings
    atlas-qe-workflow eos config.yaml --max-workers 8 --results-dir ./my_results

    # Dry run to validate configuration
    atlas-qe-workflow eos config.yaml --dry-run
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path (optional)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Workflow commands')

    # EOS workflow subcommand
    eos_parser = subparsers.add_parser(
        'eos',
        help='Run equation of state workflow'
    )
    eos_parser.add_argument(
        'config_file',
        type=Path,
        help='YAML configuration file for EOS study'
    )
    eos_parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    eos_parser.add_argument(
        '--results-dir',
        type=Path,
        help='Results output directory (overrides config)'
    )
    eos_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate inputs but do not execute calculations'
    )

    return parser


def run_eos_workflow(args) -> int:
    """Run EOS workflow with given arguments."""
    try:
        # Initialize workflow runner
        runner = EOSWorkflowRunner(
            config_path=args.config_file,
            results_dir=args.results_dir
        )

        if args.dry_run:
            # Dry run mode
            logger.info("Dry run mode: validating configuration and generating inputs")
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
                for combo in struct_info['combinations']:
                    print(f"     - {combo}")

            # Run dry validation
            results = runner.run_workflow(
                max_workers=args.max_workers,
                dry_run=True
            )

            print(f"\n✅ Dry run completed successfully!")
            print(f"Generated input files for {results.completed_tasks} calculations")

        else:
            # Real execution
            results = runner.run_workflow(max_workers=args.max_workers)

            print(f"\n🎉 EOS Workflow Completed!")
            print(f"═══════════════════════════")
            print(f"Total calculations: {results.total_tasks}")
            print(f"Successful: {results.completed_tasks}")
            print(f"Failed: {results.failed_tasks}")
            print(f"Execution time: {results.execution_time:.1f} seconds")
            print(f"Results saved in: {runner.results_dir}")
            print(f"Summary file: {results.summary_file}")

        return 0

    except Exception as e:
        logger.error(f"EOS workflow failed: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    global logger
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Handle commands
    if args.command == 'eos':
        return run_eos_workflow(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())