#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper script to download all checkpoints for Cosmos-Transfer2.5
# This script ensures the right Python environment is used

set -e

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    cat << 'EOF'
Cosmos-Transfer2.5 Checkpoint Downloader
=========================================

Downloads all required model checkpoints before running inference.

USAGE:
    bash download_checkpoints.sh [OPTIONS]

OPTIONS:
    --skip-guardrails     Skip downloading guardrail checkpoints
    --models MODEL_LIST   Comma-separated list of models to download
                         Options: depth, edge, seg, blur, all
                         Default: all
    --help, -h           Show this help message

EXAMPLES:
    # Download all checkpoints (recommended)
    bash download_checkpoints.sh

    # Download only specific control models
    bash download_checkpoints.sh --models depth,edge

    # Download all models except guardrails
    bash download_checkpoints.sh --skip-guardrails

WHAT THIS DOWNLOADS:
    1. Control models (depth, edge, seg, blur) - ~8GB total
    2. T5 text encoder (google-t5/t5-11b) - ~42GB
    3. VAE tokenizer - ~1GB
    4. Guardrail models (safety filters) - ~15GB

Total download size: ~66GB

The checkpoints are cached in ~/.cache/huggingface/ and will be
reused across inference runs.
EOF
    exit 0
fi

echo "============================================"
echo "Cosmos-Transfer2.5 Checkpoint Downloader"
echo "============================================"
echo ""

# Run the Python script
python -m download_checkpoints "$@"
