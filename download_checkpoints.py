#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone script to download all required checkpoints for Cosmos-Transfer2.5 inference
# This script triggers all checkpoint downloads that would normally happen during inference.

"""
Download all checkpoints required for Cosmos-Transfer2.5 inference.

This script downloads:
1. Model checkpoints (depth, edge, seg, blur control models)
2. T5 text encoder checkpoint (google-t5/t5-11b)
3. Guardrail checkpoints (text and video safety filters)
4. VAE tokenizer checkpoint

Usage:
    python download_checkpoints.py [OPTIONS]

Options:
    --skip-guardrails     Skip downloading guardrail checkpoints
    --models MODEL_LIST   Comma-separated list of models to download
                         (depth,edge,seg,blur,all). Default: all
    --help               Show this help message
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is not installed or the virtual environment is not activated.")
    print("Please activate your conda/venv environment first:")
    print("  conda activate cosmos-transfer2.5")
    print("Or install the package:")
    print("  pip install -e .")
    sys.exit(1)

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid


# Checkpoint UUIDs for all required models
CHECKPOINT_UUIDS = {
    # Transfer2.5 model checkpoints (control models)
    "depth": "0f214f66-ae98-43cf-ab25-d65d09a7e68f",
    "edge": "ecd0ba00-d598-4f94-aa09-e8627899c431",
    "seg": "fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab",
    "blur": "20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1",

    # Text encoder (T5)
    "t5": "4dbf13c6-1d30-4b02-99d6-75780dd8b744",

    # VAE tokenizer
    "vae": "685afcaa-4de2-42fe-b7b9-69f7a2dee4d8",

    # Guardrail checkpoints
    "guardrail": "9c7b7da4-2d95-45bb-9cb8-2eed954e9736",
}

# Groupings for convenience
MODEL_GROUPS = {
    "control_models": ["depth", "edge", "seg", "blur"],
    "text_models": ["t5"],
    "tokenizer": ["vae"],
    "guardrails": ["guardrail"],
    "all": ["depth", "edge", "seg", "blur", "t5", "vae", "guardrail"],
}


def download_checkpoint(name: str, uuid: str) -> bool:
    """
    Download a checkpoint by UUID.

    Args:
        name: Human-readable name for the checkpoint
        uuid: UUID of the checkpoint

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        log.info(f"Downloading checkpoint: {name} (UUID: {uuid})")
        checkpoint_config = get_checkpoint_by_uuid(uuid)
        # Accessing .path triggers the download
        checkpoint_path = checkpoint_config.path
        log.success(f"✓ Successfully downloaded {name} to: {checkpoint_path}")
        return True
    except Exception as e:
        log.error(f"✗ Failed to download {name}: {e}")
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download all checkpoints required for Cosmos-Transfer2.5 inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--skip-guardrails",
        action="store_true",
        help="Skip downloading guardrail checkpoints (safety filters)"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models to download (depth,edge,seg,blur,all). Default: all"
    )

    return parser.parse_args()


def main():
    """Main entry point for checkpoint downloading."""
    args = parse_args()

    # Parse model selection
    requested_models = set()
    for model_spec in args.models.split(","):
        model_spec = model_spec.strip().lower()
        if model_spec in MODEL_GROUPS:
            requested_models.update(MODEL_GROUPS[model_spec])
        elif model_spec in CHECKPOINT_UUIDS:
            requested_models.add(model_spec)
        else:
            log.warning(f"Unknown model: {model_spec}")

    # Remove guardrails if requested
    if args.skip_guardrails:
        requested_models.discard("guardrail")
        log.info("Skipping guardrail checkpoints")

    # Sort for consistent ordering
    requested_models = sorted(requested_models)

    log.info("=" * 80)
    log.info("Cosmos-Transfer2.5 Checkpoint Downloader")
    log.info("=" * 80)
    log.info(f"Downloading {len(requested_models)} checkpoint(s): {', '.join(requested_models)}")
    log.info("")

    # Download all requested checkpoints
    results = {}
    for model in requested_models:
        uuid = CHECKPOINT_UUIDS[model]
        success = download_checkpoint(model, uuid)
        results[model] = success
        log.info("")

    # Print summary
    log.info("=" * 80)
    log.info("Download Summary")
    log.info("=" * 80)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    log.info(f"✓ Successful: {len(successful)}/{len(results)}")
    if successful:
        for name in successful:
            log.info(f"  - {name}")

    if failed:
        log.error(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            log.error(f"  - {name}")
        sys.exit(1)
    else:
        log.success("\n🎉 All checkpoints downloaded successfully!")
        log.info("\nYou can now run inference with run_inferencev2.sh")
        sys.exit(0)


if __name__ == "__main__":
    main()
