"""
Shared utilities for the MR-RATE pipeline runner scripts.
"""

import argparse
import subprocess
from pathlib import Path
import yaml


def parse_args(description: str) -> argparse.Namespace:
    """Parse the single --config argument shared by all pipeline runners."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to the batch YAML config file (e.g. run/configs/mri_batch00.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load and return a YAML batch config file."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def run_step(step_label: str, cmd: list) -> None:
    """Execute a pipeline step as a subprocess; raises on non-zero exit."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  {step_label}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print(f"{separator}\n")

    subprocess.run([str(c) for c in cmd], check=True)

    print(f"\n  {step_label} — done.")
