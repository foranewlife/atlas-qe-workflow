#!/usr/bin/env python3
"""
Resource Management Tool for Distributed Workflow System

Provides tools for managing computational resources including:
- Resource health monitoring
- SSH connectivity testing
- Software availability checking
- Resource capacity planning
- Load balancing analysis

Usage:
    python manage_resources.py --test-all
    python manage_resources.py --test-resource server_6101
    python manage_resources.py --check-software atlas qe
    python manage_resources.py --capacity-planning
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.core.configuration import ConfigurationManager
from src.core.resource_manager import ResourceManager, LocalResource, RemoteResource, ResourceType
from src.utils.logging_config import setup_logging

import logging
logger = logging.getLogger(__name__)


class ResourceManagerTool:
    """
    Tool for managing and testing computational resources.

    Provides comprehensive resource management capabilities including
    connectivity testing, software validation, and capacity planning.
    """

    def __init__(self, resource_config_file: str = "config/resources.yaml"):
        """Initialize resource management tool."""
        self.config_manager = ConfigurationManager(
            resource_config_file=resource_config_file
        )

        self.resource_manager = ResourceManager()
        self._setup_resources()

        logger.info("ResourceManagerTool initialized")

    def _setup_resources(self):
        """Setup computational resources from configuration."""
        for config in self.config_manager.resource_configs:
            if config.resource_type == ResourceType.LOCAL:
                resource = LocalResource(
                    name=config.name,
                    capability=config.capability,
                    software_paths=config.software_paths
                )
            else:
                resource = RemoteResource(
                    name=config.name,
                    hostname=config.hostname,
                    capability=config.capability,
                    software_paths=config.software_paths,
                    username=config.username,
                    ssh_key=config.ssh_key,
                    remote_work_dir=config.remote_work_dir
                )
            self.resource_manager.add_resource(resource)

    def test_all_resources(self) -> Dict[str, bool]:
        """Test connectivity and availability of all resources."""
        print("\n" + "="*80)
        print("TESTING ALL RESOURCES")
        print("="*80)

        results = {}

        for name, resource in self.resource_manager.resources.items():
            print(f"\nTesting {name} ({resource.resource_type.value})...")

            # Basic availability test
            available = resource.check_availability()
            results[name] = available

            if available:
                print(f"  ✅ {name} is available")

                # Additional tests for different resource types
                if resource.resource_type == ResourceType.REMOTE:
                    self._test_remote_resource_details(resource)
                else:
                    self._test_local_resource_details(resource)

            else:
                print(f"  ❌ {name} is not available")
                self._diagnose_resource_issues(resource)

        # Summary
        available_count = sum(results.values())
        total_count = len(results)

        print(f"\n" + "="*80)
        print(f"RESOURCE TEST SUMMARY: {available_count}/{total_count} resources available")
        print("="*80)

        return results

    def test_specific_resource(self, resource_name: str) -> bool:
        """Test a specific resource in detail."""
        if resource_name not in self.resource_manager.resources:
            print(f"❌ Resource '{resource_name}' not found")
            print(f"Available resources: {list(self.resource_manager.resources.keys())}")
            return False

        resource = self.resource_manager.resources[resource_name]

        print(f"\n" + "="*80)
        print(f"DETAILED TEST: {resource_name}")
        print("="*80)

        # Basic info
        print(f"Type: {resource.resource_type.value}")
        print(f"Cores: {resource.capability.cores}")
        print(f"Memory: {resource.capability.memory_gb} GB")
        print(f"Max Jobs: {resource.capability.max_concurrent_jobs}")

        if hasattr(resource, 'hostname'):
            print(f"Hostname: {resource.hostname}")
            print(f"Username: {resource.username}")

        # Availability test
        print(f"\n🔍 Testing availability...")
        available = resource.check_availability()

        if available:
            print(f"✅ Resource is available")

            # Detailed tests
            if resource.resource_type == ResourceType.REMOTE:
                success = self._test_remote_resource_comprehensive(resource)
            else:
                success = self._test_local_resource_comprehensive(resource)

            return success
        else:
            print(f"❌ Resource is not available")
            self._diagnose_resource_issues(resource)
            return False

    def _test_remote_resource_details(self, resource):
        """Test remote resource with basic checks."""
        try:
            # Test SSH connectivity
            cmd = resource.ssh_base_cmd + ['echo', 'test']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                print(f"    ✅ SSH connectivity")
            else:
                print(f"    ❌ SSH connectivity failed")

            # Test uptime
            cmd = resource.ssh_base_cmd + ['uptime']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                uptime_output = result.stdout.decode().strip()
                print(f"    ℹ️  Uptime: {uptime_output}")

        except Exception as e:
            print(f"    ❌ Remote testing failed: {e}")

    def _test_local_resource_details(self, resource):
        """Test local resource with basic checks."""
        try:
            import psutil

            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            print(f"    ℹ️  CPU usage: {cpu_percent:.1f}%")
            print(f"    ℹ️  Memory usage: {memory.percent:.1f}%")
            print(f"    ℹ️  Available memory: {memory.available / (1024**3):.1f} GB")

        except Exception as e:
            print(f"    ⚠️  Could not get system metrics: {e}")

    def _test_remote_resource_comprehensive(self, resource) -> bool:
        """Comprehensive testing of remote resource."""
        success = True

        try:
            print(f"\n🔍 Testing SSH connection...")
            cmd = resource.ssh_base_cmd + ['echo', 'connection_test']
            result = subprocess.run(cmd, capture_output=True, timeout=15)

            if result.returncode == 0:
                print(f"  ✅ SSH connection successful")
            else:
                print(f"  ❌ SSH connection failed")
                success = False

            print(f"\n🔍 Testing remote directory creation...")
            test_dir = f"{resource.remote_work_dir}/test_{int(time.time())}"
            cmd = resource.ssh_base_cmd + ['mkdir', '-p', test_dir]
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                print(f"  ✅ Directory creation successful")

                # Cleanup test directory
                cmd = resource.ssh_base_cmd + ['rmdir', test_dir]
                subprocess.run(cmd, capture_output=True, timeout=10)
            else:
                print(f"  ❌ Directory creation failed")
                success = False

            print(f"\n🔍 Testing file transfer capabilities...")
            success &= self._test_file_transfer(resource)

            print(f"\n🔍 Testing software availability...")
            success &= self._test_software_on_resource(resource)

            print(f"\n🔍 Testing system resources...")
            self._check_remote_system_resources(resource)

        except Exception as e:
            print(f"❌ Comprehensive testing failed: {e}")
            success = False

        return success

    def _test_local_resource_comprehensive(self, resource) -> bool:
        """Comprehensive testing of local resource."""
        success = True

        try:
            print(f"\n🔍 Testing software availability...")
            success &= self._test_software_on_resource(resource)

            print(f"\n🔍 Testing system resources...")
            self._check_local_system_resources(resource)

            print(f"\n🔍 Testing directory access...")
            test_dir = Path("results/test_local")
            try:
                test_dir.mkdir(parents=True, exist_ok=True)
                test_file = test_dir / "test.txt"
                test_file.write_text("test")
                test_file.unlink()
                test_dir.rmdir()
                print(f"  ✅ Directory access successful")
            except Exception as e:
                print(f"  ❌ Directory access failed: {e}")
                success = False

        except Exception as e:
            print(f"❌ Comprehensive testing failed: {e}")
            success = False

        return success

    def _test_file_transfer(self, resource) -> bool:
        """Test file transfer capabilities for remote resource."""
        try:
            import tempfile

            # Create test file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("test file content")
                local_test_file = f.name

            remote_test_file = f"{resource.remote_work_dir}/transfer_test.txt"

            # Test upload
            cmd = [
                'rsync', '-av', '--timeout=30',
                local_test_file,
                f"{resource.username}@{resource.hostname}:{remote_test_file}"
            ]

            if resource.ssh_key:
                cmd.extend(['-e', f'ssh -i {resource.ssh_key}'])

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0:
                print(f"  ✅ File upload successful")

                # Test download
                download_file = local_test_file + ".download"
                cmd = [
                    'rsync', '-av', '--timeout=30',
                    f"{resource.username}@{resource.hostname}:{remote_test_file}",
                    download_file
                ]

                if resource.ssh_key:
                    cmd.extend(['-e', f'ssh -i {resource.ssh_key}'])

                result = subprocess.run(cmd, capture_output=True, timeout=60)

                if result.returncode == 0:
                    print(f"  ✅ File download successful")

                    # Cleanup
                    Path(local_test_file).unlink()
                    Path(download_file).unlink()

                    # Cleanup remote file
                    cmd = resource.ssh_base_cmd + ['rm', remote_test_file]
                    subprocess.run(cmd, capture_output=True, timeout=10)

                    return True
                else:
                    print(f"  ❌ File download failed")
            else:
                print(f"  ❌ File upload failed: {result.stderr.decode()}")

            # Cleanup on failure
            try:
                Path(local_test_file).unlink()
            except:
                pass

            return False

        except Exception as e:
            print(f"  ❌ File transfer test failed: {e}")
            return False

    def _test_software_on_resource(self, resource) -> bool:
        """Test software availability on resource."""
        all_available = True

        for software, path in resource.software_paths.items():
            if resource.resource_type == ResourceType.LOCAL:
                # Test local software
                if Path(path).exists():
                    print(f"  ✅ {software}: {path}")
                else:
                    print(f"  ❌ {software}: {path} (not found)")
                    all_available = False
            else:
                # Test remote software
                cmd = resource.ssh_base_cmd + ['test', '-f', path]
                result = subprocess.run(cmd, capture_output=True, timeout=10)

                if result.returncode == 0:
                    print(f"  ✅ {software}: {path}")
                else:
                    print(f"  ❌ {software}: {path} (not found)")
                    all_available = False

        return all_available

    def _check_remote_system_resources(self, resource):
        """Check system resources on remote machine."""
        try:
            # Memory info
            cmd = resource.ssh_base_cmd + ['free', '-h']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                memory_output = result.stdout.decode().strip()
                print(f"  ℹ️  Memory info:")
                for line in memory_output.split('\n'):
                    if 'Mem:' in line:
                        print(f"    {line}")

            # Disk space
            cmd = resource.ssh_base_cmd + ['df', '-h', resource.remote_work_dir]
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                disk_output = result.stdout.decode().strip()
                print(f"  ℹ️  Disk space ({resource.remote_work_dir}):")
                for line in disk_output.split('\n')[1:]:  # Skip header
                    print(f"    {line}")

            # Load average
            cmd = resource.ssh_base_cmd + ['uptime']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                uptime_output = result.stdout.decode().strip()
                if 'load average:' in uptime_output:
                    load_part = uptime_output.split('load average:')[1].strip()
                    print(f"  ℹ️  Load average: {load_part}")

        except Exception as e:
            print(f"  ⚠️  Could not check remote system resources: {e}")

    def _check_local_system_resources(self, resource):
        """Check local system resources."""
        try:
            import psutil

            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"  ℹ️  CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")

            # Memory info
            memory = psutil.virtual_memory()
            print(f"  ℹ️  Memory: {memory.total / (1024**3):.1f} GB total, "
                  f"{memory.percent:.1f}% used")

            # Disk space
            workspace = Path("results")
            if workspace.exists():
                total, used, free = psutil.disk_usage(workspace)
                free_percent = (free / total) * 100
                print(f"  ℹ️  Disk: {free / (1024**3):.1f} GB free "
                      f"({free_percent:.1f}%)")

        except Exception as e:
            print(f"  ⚠️  Could not check local system resources: {e}")

    def _diagnose_resource_issues(self, resource):
        """Diagnose common resource issues."""
        print(f"  🔍 Diagnosing issues for {resource.name}...")

        if resource.resource_type == ResourceType.REMOTE:
            # Test basic network connectivity
            try:
                cmd = ['ping', '-c', '3', resource.hostname]
                result = subprocess.run(cmd, capture_output=True, timeout=15)

                if result.returncode == 0:
                    print(f"    ✅ Network connectivity to {resource.hostname}")
                else:
                    print(f"    ❌ Cannot ping {resource.hostname}")
            except:
                print(f"    ❌ Network connectivity test failed")

            # Test SSH port
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((resource.hostname, 22))
                sock.close()

                if result == 0:
                    print(f"    ✅ SSH port (22) is open")
                else:
                    print(f"    ❌ SSH port (22) is not accessible")
            except:
                print(f"    ❌ SSH port test failed")

        else:
            # Local resource diagnostics
            print(f"    ℹ️  Local resource - check software paths and permissions")

    def check_software_availability(self, software_list: List[str]):
        """Check availability of specific software across all resources."""
        print(f"\n" + "="*80)
        print(f"SOFTWARE AVAILABILITY CHECK")
        print("="*80)

        for software in software_list:
            print(f"\n🔍 Checking {software}:")

            available_on = []
            unavailable_on = []

            for name, resource in self.resource_manager.resources.items():
                if software in resource.software_paths:
                    path = resource.software_paths[software]

                    if resource.resource_type == ResourceType.LOCAL:
                        if Path(path).exists():
                            available_on.append(name)
                            print(f"  ✅ {name}: {path}")
                        else:
                            unavailable_on.append(name)
                            print(f"  ❌ {name}: {path} (not found)")
                    else:
                        # Test remote software
                        cmd = resource.ssh_base_cmd + ['test', '-f', path]
                        result = subprocess.run(cmd, capture_output=True, timeout=10)

                        if result.returncode == 0:
                            available_on.append(name)
                            print(f"  ✅ {name}: {path}")
                        else:
                            unavailable_on.append(name)
                            print(f"  ❌ {name}: {path} (not found)")
                else:
                    print(f"  ➖ {name}: not configured")

            print(f"  📊 Summary: Available on {len(available_on)} resources, "
                  f"unavailable on {len(unavailable_on)}")

    def capacity_planning_analysis(self):
        """Perform capacity planning analysis."""
        print(f"\n" + "="*80)
        print(f"CAPACITY PLANNING ANALYSIS")
        print("="*80)

        # Update all resource metrics
        self.resource_manager.update_all_resources()

        total_cores = 0
        total_memory = 0
        available_cores = 0
        available_memory = 0

        print(f"\nResource Inventory:")
        for name, resource in self.resource_manager.resources.items():
            status = "🟢" if resource.status.value == 'available' else "🔴"

            total_cores += resource.capability.cores
            total_memory += resource.capability.memory_gb

            if resource.status.value == 'available':
                available_cores += resource.capability.cores
                available_memory += resource.capability.memory_gb

            load = resource.get_load_score()
            load_icon = "🟢" if load < 0.5 else "🟡" if load < 0.8 else "🔴"

            print(f"  {status} {name}:")
            print(f"    Cores: {resource.capability.cores}")
            print(f"    Memory: {resource.capability.memory_gb} GB")
            print(f"    Max Jobs: {resource.capability.max_concurrent_jobs}")
            print(f"    Load: {load_icon} {load:.2f}")
            print(f"    Active Jobs: {resource.metrics.active_jobs}")

        print(f"\nCapacity Summary:")
        print(f"  Total Cores: {total_cores}")
        print(f"  Available Cores: {available_cores}")
        print(f"  Total Memory: {total_memory} GB")
        print(f"  Available Memory: {available_memory} GB")

        # Utilization analysis
        if total_cores > 0:
            core_utilization = ((total_cores - available_cores) / total_cores) * 100
            print(f"  Core Utilization: {core_utilization:.1f}%")

        if total_memory > 0:
            memory_utilization = ((total_memory - available_memory) / total_memory) * 100
            print(f"  Memory Utilization: {memory_utilization:.1f}%")

        # Recommendations
        print(f"\nRecommendations:")

        if available_cores == 0:
            print(f"  ⚠️  No computational resources available")
        elif available_cores < 4:
            print(f"  ⚠️  Low computational capacity ({available_cores} cores)")
        else:
            print(f"  ✅ Adequate computational capacity")

        # Software distribution analysis
        software_dist = {}
        for name, resource in self.resource_manager.resources.items():
            for software in resource.software_paths.keys():
                if software not in software_dist:
                    software_dist[software] = []
                software_dist[software].append(name)

        print(f"\nSoftware Distribution:")
        for software, resources in software_dist.items():
            print(f"  {software}: {len(resources)} resources ({', '.join(resources)})")


def main():
    """Main entry point for resource management tool."""
    parser = argparse.ArgumentParser(
        description="Resource Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--resources",
        default="config/resources.yaml",
        help="Resource configuration file"
    )

    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all resources"
    )

    parser.add_argument(
        "--test-resource",
        help="Test specific resource by name"
    )

    parser.add_argument(
        "--check-software",
        nargs="+",
        help="Check availability of specific software"
    )

    parser.add_argument(
        "--capacity-planning",
        action="store_true",
        help="Perform capacity planning analysis"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    try:
        # Initialize resource manager
        manager = ResourceManagerTool(args.resources)

        # Execute requested actions
        if args.test_all:
            manager.test_all_resources()

        if args.test_resource:
            manager.test_specific_resource(args.test_resource)

        if args.check_software:
            manager.check_software_availability(args.check_software)

        if args.capacity_planning:
            manager.capacity_planning_analysis()

        # Default action if no specific action specified
        if not any([args.test_all, args.test_resource, args.check_software, args.capacity_planning]):
            print("No action specified. Use --help for available options.")
            manager.test_all_resources()

    except Exception as e:
        logger.error(f"Resource management failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())