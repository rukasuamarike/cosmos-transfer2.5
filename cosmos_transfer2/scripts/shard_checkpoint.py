#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Checkpoint Sharding Script for Cosmos Transfer2.5

This script shards a full model checkpoint into N pipeline-parallel shards,
enabling deployment on cheaper GPUs with lower memory requirements.

Sharding Strategy:
- DiT blocks are evenly distributed across PP ranks
- Embedders (x_embedder, pos_embedder, t_embedder) go to rank 0
- Final layer goes to last rank
- VAE is kept as a separate file (shared across all ranks)

Usage:
    python -m cosmos_transfer2.scripts.shard_checkpoint \
        --input-checkpoint /path/to/full_checkpoint.pt \
        --output-dir /path/to/sharded/ \
        --pp-size 4

Output:
    /path/to/sharded/
    ├── dit_rank_0.pt  (blocks 0-6 + embedders)
    ├── dit_rank_1.pt  (blocks 7-13)
    ├── dit_rank_2.pt  (blocks 14-20)
    ├── dit_rank_3.pt  (blocks 21-27 + final_layer)
    ├── vae.pt         (tokenizer/VAE - shared)
    └── metadata.json  (sharding config)
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_state_dict_structure(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Analyze checkpoint structure to identify components."""
    structure = {
        "dit_blocks": set(),
        "embedders": [],
        "final_layer": [],
        "vae": [],
        "other": [],
    }

    block_pattern = re.compile(r"^(?:net\.)?blocks\.(\d+)\.")

    for key in state_dict.keys():
        # Check for DiT blocks
        match = block_pattern.match(key)
        if match:
            block_idx = int(match.group(1))
            structure["dit_blocks"].add(block_idx)
            continue

        # Categorize other components
        if any(x in key for x in ["x_embedder", "pos_embedder", "t_embedder", "t_embedding"]):
            structure["embedders"].append(key)
        elif "final_layer" in key:
            structure["final_layer"].append(key)
        elif any(x in key for x in ["tokenizer", "vae", "encoder", "decoder"]):
            structure["vae"].append(key)
        else:
            structure["other"].append(key)

    structure["dit_blocks"] = sorted(structure["dit_blocks"])
    structure["num_blocks"] = len(structure["dit_blocks"])
    return structure


def shard_checkpoint(
    input_path: str,
    output_dir: str,
    pp_size: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Shard a full checkpoint into pp_size pipeline-parallel shards.

    Args:
        input_path: Path to full checkpoint
        output_dir: Directory to save sharded checkpoints
        pp_size: Number of pipeline parallel ranks
        verbose: Print progress information

    Returns:
        Metadata dict describing the sharding
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Loading checkpoint from {input_path}...")

    # Load checkpoint
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Analyze structure
    structure = parse_state_dict_structure(state_dict)
    num_blocks = structure["num_blocks"]

    if verbose:
        print(f"Found {num_blocks} DiT blocks")
        print(f"Embedder keys: {len(structure['embedders'])}")
        print(f"Final layer keys: {len(structure['final_layer'])}")
        print(f"VAE keys: {len(structure['vae'])}")
        print(f"Other keys: {len(structure['other'])}")

    if num_blocks == 0:
        raise ValueError("No DiT blocks found in checkpoint. Check key naming convention.")

    # Calculate block distribution
    blocks_per_rank = num_blocks // pp_size
    remainder = num_blocks % pp_size

    block_ranges = []
    start = 0
    for rank in range(pp_size):
        # Distribute remainder blocks to first ranks
        extra = 1 if rank < remainder else 0
        end = start + blocks_per_rank + extra
        block_ranges.append((start, end))
        start = end

    if verbose:
        print(f"\nBlock distribution (PP size = {pp_size}):")
        for rank, (start, end) in enumerate(block_ranges):
            print(f"  Rank {rank}: blocks {start}-{end - 1} ({end - start} blocks)")

    # Create shards
    shards = [defaultdict(dict) for _ in range(pp_size)]
    vae_shard = {}

    block_pattern = re.compile(r"^(?:net\.)?blocks\.(\d+)\.(.*)")

    for key, value in state_dict.items():
        # Handle DiT blocks
        match = block_pattern.match(key)
        if match:
            block_idx = int(match.group(1))
            rest_of_key = match.group(2)

            # Find which rank this block belongs to
            for rank, (start, end) in enumerate(block_ranges):
                if start <= block_idx < end:
                    # Renumber block within this shard
                    new_block_idx = block_idx - start
                    # Preserve net. prefix if present
                    prefix = "net." if key.startswith("net.") else ""
                    new_key = f"{prefix}blocks.{new_block_idx}.{rest_of_key}"
                    shards[rank][new_key] = value
                    break
            continue

        # Embedders go to rank 0
        if any(x in key for x in ["x_embedder", "pos_embedder", "t_embedder", "t_embedding"]):
            shards[0][key] = value
            continue

        # Final layer goes to last rank
        if "final_layer" in key:
            shards[pp_size - 1][key] = value
            continue

        # VAE/tokenizer goes to separate shard
        if any(x in key for x in ["tokenizer", "vae", "encoder", "decoder"]):
            vae_shard[key] = value
            continue

        # Other keys (like crossattn_proj, img_context_proj) go to rank 0
        shards[0][key] = value

    # Save shards
    metadata = {
        "pp_size": pp_size,
        "num_blocks": num_blocks,
        "block_ranges": block_ranges,
        "original_checkpoint": os.path.basename(input_path),
        "shards": [],
    }

    for rank in range(pp_size):
        shard_path = os.path.join(output_dir, f"dit_rank_{rank}.pt")
        torch.save(dict(shards[rank]), shard_path)
        shard_info = {
            "rank": rank,
            "path": f"dit_rank_{rank}.pt",
            "block_range": block_ranges[rank],
            "num_keys": len(shards[rank]),
            "has_embedders": rank == 0,
            "has_final_layer": rank == pp_size - 1,
        }
        metadata["shards"].append(shard_info)
        if verbose:
            size_mb = os.path.getsize(shard_path) / (1024 * 1024)
            print(f"Saved rank {rank} shard: {shard_path} ({size_mb:.1f} MB, {len(shards[rank])} keys)")

    # Save VAE shard
    if vae_shard:
        vae_path = os.path.join(output_dir, "vae.pt")
        torch.save(vae_shard, vae_path)
        metadata["vae"] = {
            "path": "vae.pt",
            "num_keys": len(vae_shard),
        }
        if verbose:
            size_mb = os.path.getsize(vae_path) / (1024 * 1024)
            print(f"Saved VAE shard: {vae_path} ({size_mb:.1f} MB, {len(vae_shard)} keys)")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"Saved metadata: {metadata_path}")

    return metadata


def validate_shards(output_dir: str, verbose: bool = True) -> bool:
    """Validate that sharded checkpoints can be loaded and have expected structure."""
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    total_keys = 0
    for shard_info in metadata["shards"]:
        shard_path = os.path.join(output_dir, shard_info["path"])
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)

        if verbose:
            print(f"Rank {shard_info['rank']}: {len(shard)} keys")
            # Show sample keys
            sample_keys = list(shard.keys())[:3]
            for key in sample_keys:
                print(f"  - {key}")

        total_keys += len(shard)

    if "vae" in metadata:
        vae_path = os.path.join(output_dir, metadata["vae"]["path"])
        vae = torch.load(vae_path, map_location="cpu", weights_only=False)
        if verbose:
            print(f"VAE: {len(vae)} keys")
        total_keys += len(vae)

    if verbose:
        print(f"\nTotal keys across all shards: {total_keys}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Shard a Cosmos Transfer2.5 checkpoint for pipeline parallelism"
    )
    parser.add_argument(
        "--input-checkpoint",
        "-i",
        type=str,
        required=True,
        help="Path to full checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save sharded checkpoints",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=2,
        help="Pipeline parallel size (number of shards)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate shards after creation",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Shard checkpoint
    metadata = shard_checkpoint(
        input_path=args.input_checkpoint,
        output_dir=args.output_dir,
        pp_size=args.pp_size,
        verbose=not args.quiet,
    )

    # Validate if requested
    if args.validate:
        print("\nValidating shards...")
        validate_shards(args.output_dir, verbose=not args.quiet)

    print("\nDone!")
    print(f"Sharded checkpoint saved to: {args.output_dir}")
    print(f"Use with: --sharded-checkpoint-dir {args.output_dir} --pp-size {args.pp_size}")


if __name__ == "__main__":
    main()
