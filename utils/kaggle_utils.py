import json
import os
import subprocess
import time
from pathlib import Path
import yaml
from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from utils.db_utils import get_logger
from utils.discord_utils import notify_script_complete, notify_script_failed, notify_batch_summary

logger = get_logger(__name__)

class KernelStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class KernelJob:
    script_path: str
    kernel_slug: str
    status: KernelStatus
    retry_count: int = 0


load_dotenv()
def execute_single_script(
        script_path: str,
        username: str = str,
        title: str = "Kaggle Execution",
        enable_gpu: bool = False,
        enable_internet: bool = True,
        is_private: bool = True
):
    """
    Execute a Python script on Kaggle using the Kaggle Kernels API.

    Args:
        script_path: Path to the .py file to execute
        username: Your Kaggle username
        title: Title for the Kaggle kernel
        enable_gpu: Whether to enable GPU
        enable_internet: Whether to enable internet
        is_private: Whether the kernel should be private
    """
    script_path = Path(script_path)

    if not script_path.exists() or script_path.suffix != '.py':
        raise ValueError("Invalid Python file path")

    # Create kernel metadata
    kernel_slug = title.lower().replace(' ', '-')
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": script_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": str(is_private).lower(),
        "enable_gpu": str(enable_gpu).lower(),
        "enable_tpu": "false",
        "enable_internet": str(enable_internet).lower(),
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    # Create temp directory for kernel files
    temp_dir = Path("temp_kaggle_kernel")
    temp_dir.mkdir(exist_ok=True)

    # Write metadata
    metadata_path = temp_dir / "kernel-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy script to temp directory
    import shutil
    shutil.copy(script_path, temp_dir / script_path.name)

    # Push to Kaggle
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(temp_dir)],
        capture_output=True,
        text=True
    )

    # Cleanup
    shutil.rmtree(temp_dir)

    if result.returncode != 0:
        raise RuntimeError(f"Kaggle push failed: {result.stderr}")

    return result.stdout


def get_kaggle_kernel_status(kernel_slug: str) -> dict:
    """
    Get the current status of a Kaggle kernel.

    Args:
        kernel_slug: The kernel slug (derived from title)

    Returns:
        dict: Dictionary containing status information
    """
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    kernel_ref = f"{KAGGLE_USERNAME}/{kernel_slug}"

    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_ref],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to get kernel status: {result.stderr}")

    # Parse the output
    output = result.stdout.strip()

    # Determine status from output
    status_info = {
        "kernel_ref": kernel_ref,
        "raw_output": output,
        "is_complete": "complete" in output.lower(),
        "has_error": "error" in output.lower(),
        "is_running": "running" in output.lower() or "queued" in output.lower()
    }

    return status_info



def run_kaggle_scripts_from_yaml(
        yaml_path: str,
        max_concurrent: int = 5,
        max_retries: int = 3,
        check_interval: int = 30
):
    """
    Execute multiple Kaggle scripts concurrently based on YAML configuration.

    Args:
        yaml_path: Path to YAML file with Kaggle settings
        max_concurrent: Maximum number of concurrent executions (default: 5)
        max_retries: Maximum retry attempts for failed scripts
        check_interval: Seconds between status checks
    """
    # Load YAML configuration
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')

    # Extract settings
    scripts = config.get('scripts', [])
    username = config.get('username', KAGGLE_USERNAME)
    title_prefix = config.get('title_prefix', 'Script')
    enable_gpu = config.get('enable_gpu', True)
    enable_internet = config.get('enable_internet', True)
    is_private = config.get('is_private', True)

    # Initialize jobs
    jobs: Dict[str, KernelJob] = {}
    for idx, script_path in enumerate(scripts):
        kernel_slug = f"{title_prefix}-{idx}".lower().replace(' ', '-')
        jobs[script_path] = KernelJob(
            script_path=script_path,
            kernel_slug=kernel_slug,
            status=KernelStatus.PENDING
        )

    running: Set[str] = set()
    completed: Set[str] = set()
    failed: Set[str] = set()

    def submit_job(script_path: str) -> bool:
        """Submit a job to Kaggle. Returns True if successful."""
        job = jobs[script_path]
        try:
            execute_single_script(
                script_path=script_path,
                username = username,
                title=f"{job.kernel_slug}",
                enable_gpu=enable_gpu,
                enable_internet=enable_internet,
                is_private=is_private
            )
            job.status = KernelStatus.RUNNING
            running.add(script_path)
            print(f"✓ Started: {script_path} (slug: {job.kernel_slug})")
            return True
        except Exception as e:
            print(f"✗ Failed to submit {script_path}: {e}")
            return False

    def check_job_status(script_path: str) -> KernelStatus:
        """Check status of a running job."""
        job = jobs[script_path]
        try:
            status_info = get_kaggle_kernel_status(job.kernel_slug)

            if status_info['is_complete']:
                return KernelStatus.COMPLETE
            elif status_info['has_error']:
                return KernelStatus.FAILED
            elif status_info['is_running']:
                return KernelStatus.RUNNING
            else:
                return KernelStatus.PENDING
        except Exception as e:
            print(f"✗ Error checking status for {script_path}: {e}")
            return KernelStatus.FAILED

    # Main execution loop
    while len(completed) + len(failed) < len(jobs):
        # Check status of running jobs
        for script_path in list(running):
            job = jobs[script_path]
            status = check_job_status(script_path)

            if status == KernelStatus.COMPLETE:
                job.status = KernelStatus.COMPLETE
                running.remove(script_path)
                completed.add(script_path)
                print(f"✓ Completed: {script_path}")
                notify_script_complete(script_path, job.kernel_slug)

            elif status == KernelStatus.FAILED:
                job.status = KernelStatus.FAILED
                running.remove(script_path)

                if job.retry_count < max_retries:
                    job.retry_count += 1
                    job.status = KernelStatus.PENDING
                    print(f"⟳ Retry {job.retry_count}/{max_retries}: {script_path}")
                    notify_script_failed(script_path, job.kernel_slug, job.retry_count, max_retries)
                else:
                    failed.add(script_path)
                    print(f"✗ Failed permanently: {script_path}")
                    notify_script_failed(script_path, job.kernel_slug, job.retry_count, max_retries)

        # Submit new jobs if slots available
        available_slots = max_concurrent - len(running)
        if available_slots > 0:
            pending_jobs = [
                sp for sp, job in jobs.items()
                if job.status == KernelStatus.PENDING and sp not in running
            ]

            for script_path in pending_jobs[:available_slots]:
                submit_job(script_path)

        # Wait before next check
        if running:
            time.sleep(check_interval)

    # Summary
    print(f"\n{'='*50}")
    print(f"Execution Summary:")
    print(f"Completed: {len(completed)}/{len(jobs)}")
    print(f"Failed: {len(failed)}/{len(jobs)}")

    if failed:
        print(f"\nFailed scripts:")
        for script_path in failed:
            print(f"  - {script_path}")

    notify_batch_summary(len(completed), len(failed), len(jobs), list(failed))

    return {
        'completed': list(completed),
        'failed': list(failed),
        'jobs': jobs
    }

# Example usage:
result = run_kaggle_scripts_from_yaml(
    yaml_path='../configs/kaggle_config.yaml',
    max_concurrent=2,
    max_retries=3,
    check_interval=10
)