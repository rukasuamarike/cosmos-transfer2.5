#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sharded Checkpoint Loader for Cosmos Transfer2.5

This module provides utilities to load pipeline-parallel sharded checkpoints
created by shard_checkpoint.py.

Usage:
    from cosmos_transfer2.scripts.load_sharded_checkpoint import (
        load_sharded_dit_checkpoint,
        ShardedCheckpointConfig,
    )

    # Load DiT shard for this rank
    config = ShardedCheckpointConfig(
        sharded_dir="/path/to/sharded/",
        pp_rank=distributed.get_rank(),
        pp_size=4,
    )
    state_dict = load_sharded_dit_checkpoint(config)
    model.load_state_dict(state_dict, strict=False)
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class ShardedCheckpointConfig:
    """Configuration for loading sharded checkpoints."""

    sharded_dir: str
    pp_rank: int
    pp_size: int
    load_vae: bool = True
    device: str = "cpu"


def load_metadata(sharded_dir: str) -> dict[str, Any]:
    """Load sharding metadata from directory."""
    metadata_path = os.path.join(sharded_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            "Ensure the checkpoint was sharded with shard_checkpoint.py"
        )
    with open(metadata_path) as f:
        return json.load(f)


def load_sharded_dit_checkpoint(
    config: ShardedCheckpointConfig,
    remap_block_indices: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Load the DiT shard for this pipeline parallel rank.

    Args:
        config: Sharding configuration
        remap_block_indices: If True, remap block indices to match full model structure

    Returns:
        State dict for this rank's DiT portion
    """
    metadata = load_metadata(config.sharded_dir)

    # Validate PP size matches
    if metadata["pp_size"] != config.pp_size:
        raise ValueError(
            f"PP size mismatch: checkpoint was sharded with pp_size={metadata['pp_size']}, "
            f"but loading with pp_size={config.pp_size}"
        )

    if config.pp_rank >= config.pp_size:
        raise ValueError(
            f"Invalid pp_rank={config.pp_rank} for pp_size={config.pp_size}"
        )

    # Find shard info for this rank
    shard_info = metadata["shards"][config.pp_rank]
    shard_path = os.path.join(config.sharded_dir, shard_info["path"])

    # Load shard
    shard = torch.load(shard_path, map_location=config.device, weights_only=False)

    if remap_block_indices:
        # Remap block indices back to original numbering
        block_start, block_end = shard_info["block_range"]
        remapped = {}
        for key, value in shard.items():
            if "blocks." in key:
                # Parse local block index
                parts = key.split(".")
                blocks_idx = parts.index("blocks")
                local_block_idx = int(parts[blocks_idx + 1])
                # Convert to global block index
                global_block_idx = local_block_idx + block_start
                parts[blocks_idx + 1] = str(global_block_idx)
                new_key = ".".join(parts)
                remapped[new_key] = value
            else:
                remapped[key] = value
        shard = remapped

    return shard


def load_vae_checkpoint(config: ShardedCheckpointConfig) -> Optional[dict[str, torch.Tensor]]:
    """
    Load the VAE checkpoint (shared across all ranks).

    Args:
        config: Sharding configuration

    Returns:
        VAE state dict, or None if not present
    """
    metadata = load_metadata(config.sharded_dir)

    if "vae" not in metadata:
        return None

    vae_path = os.path.join(config.sharded_dir, metadata["vae"]["path"])
    return torch.load(vae_path, map_location=config.device, weights_only=False)


def get_block_range_for_rank(sharded_dir: str, pp_rank: int) -> tuple[int, int]:
    """Get the (start, end) block range for a given PP rank."""
    metadata = load_metadata(sharded_dir)
    return tuple(metadata["shards"][pp_rank]["block_range"])


def get_total_blocks(sharded_dir: str) -> int:
    """Get total number of DiT blocks in the sharded model."""
    metadata = load_metadata(sharded_dir)
    return metadata["num_blocks"]


class ShardedModelLoader:
    """
    Helper class for loading sharded models in distributed settings.

    This integrates with the existing model loading infrastructure to
    automatically load the correct shard based on the current rank.

    Usage:
        loader = ShardedModelLoader(
            sharded_dir="/path/to/sharded/",
            pp_size=4,
        )

        # In your model initialization
        if loader.is_sharded:
            state_dict = loader.load_dit_shard_for_current_rank()
            model.load_state_dict(state_dict, strict=False)
    """

    def __init__(
        self,
        sharded_dir: Optional[str] = None,
        pp_size: int = 1,
        device: str = "cpu",
    ):
        self.sharded_dir = sharded_dir
        self.pp_size = pp_size
        self.device = device
        self._metadata = None

        if self.is_sharded:
            self._metadata = load_metadata(sharded_dir)

    @property
    def is_sharded(self) -> bool:
        """Check if using sharded checkpoints."""
        return self.sharded_dir is not None and self.pp_size > 1

    @property
    def num_blocks(self) -> Optional[int]:
        """Get total number of DiT blocks."""
        if self._metadata:
            return self._metadata["num_blocks"]
        return None

    def get_pp_rank(self) -> int:
        """Get current pipeline parallel rank from distributed context."""
        try:
            from megatron.core import parallel_state
            if parallel_state.is_initialized():
                # Use pipeline parallel rank if available
                if hasattr(parallel_state, "get_pipeline_model_parallel_rank"):
                    return parallel_state.get_pipeline_model_parallel_rank()
                # Fall back to context parallel rank
                return parallel_state.get_context_parallel_rank()
        except ImportError:
            pass

        # Fall back to torch distributed rank
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() % self.pp_size

        return 0

    def load_dit_shard_for_current_rank(
        self,
        remap_block_indices: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Load DiT shard for current distributed rank."""
        if not self.is_sharded:
            raise RuntimeError("Not using sharded checkpoints")

        pp_rank = self.get_pp_rank()
        config = ShardedCheckpointConfig(
            sharded_dir=self.sharded_dir,
            pp_rank=pp_rank,
            pp_size=self.pp_size,
            device=self.device,
        )
        return load_sharded_dit_checkpoint(config, remap_block_indices)

    def load_vae(self) -> Optional[dict[str, torch.Tensor]]:
        """Load VAE checkpoint."""
        if not self.is_sharded:
            raise RuntimeError("Not using sharded checkpoints")

        config = ShardedCheckpointConfig(
            sharded_dir=self.sharded_dir,
            pp_rank=0,  # VAE is rank-independent
            pp_size=self.pp_size,
            device=self.device,
        )
        return load_vae_checkpoint(config)

    def get_block_range(self) -> tuple[int, int]:
        """Get block range for current rank."""
        if not self.is_sharded:
            raise RuntimeError("Not using sharded checkpoints")

        pp_rank = self.get_pp_rank()
        return get_block_range_for_rank(self.sharded_dir, pp_rank)

    def should_have_embedders(self) -> bool:
        """Check if current rank should have embedder layers."""
        return self.get_pp_rank() == 0

    def should_have_final_layer(self) -> bool:
        """Check if current rank should have final layer."""
        return self.get_pp_rank() == self.pp_size - 1


# Convenience function for integration with existing code
def create_sharded_loader_from_args(args) -> Optional[ShardedModelLoader]:
    """
    Create a ShardedModelLoader from command-line arguments.

    Expects args to have:
    - sharded_checkpoint_dir: Path to sharded checkpoint directory
    - pipeline_parallel_size: Number of PP ranks

    Returns None if not using sharded checkpoints.
    """
    sharded_dir = getattr(args, "sharded_checkpoint_dir", None)
    pp_size = getattr(args, "pipeline_parallel_size", 1)

    if sharded_dir is None or pp_size <= 1:
        return None

    return ShardedModelLoader(
        sharded_dir=sharded_dir,
        pp_size=pp_size,
    )
