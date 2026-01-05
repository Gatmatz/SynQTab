import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")

def execute_on_kaggle(
        script_path: str,
        username: str,
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
        "id": f"{KAGGLE_USERNAME}/{kernel_slug}",
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


def get_kaggle_results(
        username: str,
        kernel_slug: str,
        output_dir: str = "kaggle_output",
        timeout: int = 3600
):
    """
    Wait for kernel completion and download results.

    Args:
        username: Your Kaggle username
        kernel_slug: The kernel slug (derived from title)
        output_dir: Directory to save output files
        timeout: Maximum wait time in seconds
    """
    kernel_ref = f"{username}/{kernel_slug}"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Wait for kernel to complete
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check kernel status
        status_result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True,
            text=True
        )

        if "complete" in status_result.stdout.lower():
            break
        elif "error" in status_result.stdout.lower():
            raise RuntimeError(f"Kernel execution failed: {status_result.stdout}")

        time.sleep(30)  # Check every 30 seconds

    # Download output
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download output: {result.stderr}")

    return output_path


def get_kaggle_kernel_status(kernel_slug: str) -> dict:
    """
    Get the current status of a Kaggle kernel.

    Args:
        username: Your Kaggle username
        kernel_slug: The kernel slug (derived from title)

    Returns:
        dict: Dictionary containing status information
    """
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