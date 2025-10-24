#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

# ============================================================================
# Cosmos-Transfer2.5 Inference Script v2
# ============================================================================
# This script provides a simplified interface to the Cosmos-Transfer2.5
# inference pipeline, generating JSON configs and calling the Python CLI.
#
# USAGE:
#   bash run_inferencev2.sh INPUT_VIDEO CONTROLS PROMPT [OPTIONS]
#
# ARGUMENTS:
#   INPUT_VIDEO     Path to input video file
#   CONTROLS        Comma-separated list of control types with optional weights
#                   Format: "control1=weight1,control2=weight2"
#                   Available controls: depth, edge, seg, vis
#                   Default weight is 1.0 if not specified
#   PROMPT          Text prompt describing desired output
#
# OPTIONS:
#   -o, --output DIR         Output directory (default: outputs_TIMESTAMP)
#   --guidance N             Guidance scale 0-7 (default: 3)
#   --seed N                 Random seed (default: 2025)
#   --resolution SIZE        Resolution: 480 or 720 (default: 720)
#   --num-prompts N          Generate N variations (default: 1)
#   --negative-prompt TEXT   Negative prompt (uses default if not specified)
#   --disable-guardrails     DISABLE SAFETY FILTERS (use at your own risk)
#   --multi-gpu              Use all available GPUs
#   --help                   Show this help message
#
# EXAMPLES:
#   # Basic usage with edge control
#   bash run_inferencev2.sh input.mp4 "edge=0.8" "a robot walking"
#
#   # Multiple controls with custom output directory
#   bash run_inferencev2.sh input.mp4 "depth=0.8,edge=0.5" "robot in factory" -o my_output
#
#   # Multiple prompt variations with multi-GPU
#   bash run_inferencev2.sh input.mp4 "depth" "industrial robot" --num-prompts 3 --multi-gpu
#
# PIPELINE FLOW:
#   1. Parse command-line arguments
#   2. Create timestamped config directory
#   3. For each prompt variation:
#      a. Generate JSON config file with all parameters
#      b. Call Python inference CLI (examples/inference.py)
#      c. Python loads model, processes video with controls, generates output
#   4. Outputs saved to: {output_dir}/{sample_name}/output.mp4
# ============================================================================

# Default values
OUTPUT_DIR=""
GUIDANCE=3
SEED=2025
RESOLUTION="720"
NUM_PROMPTS=1
NEGATIVE_PROMPT=""
DISABLE_GUARDRAILS=false
USE_MULTI_GPU=false

# Help message
show_help() {
    grep "^#" "$0" | grep -E "^# (USAGE|ARGUMENTS|OPTIONS|EXAMPLES|PIPELINE)" -A 100 | sed 's/^# //' | sed 's/^#//'
}

# Parse arguments
if [ "$#" -lt 3 ]; then
    echo "Error: Missing required arguments"
    echo ""
    show_help
    exit 1
fi

INPUT_VIDEO="$1"
CONTROLS_STR="$2"
PROMPT="$3"
shift 3

# Parse optional arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --negative-prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --disable-guardrails)
            DISABLE_GUARDRAILS=true
            shift
            ;;
        --multi-gpu)
            USE_MULTI_GPU=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video not found: $INPUT_VIDEO"
    exit 1
fi

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs_${TIMESTAMP}"
fi

# Create timestamped config directory
CONFIG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR="cosmos_inference_${CONFIG_TIMESTAMP}"
mkdir -p "$CONFIG_DIR"

# Parse control string (e.g., "depth=0.8,edge=0.5")
echo "Parsing controls: $CONTROLS_STR"
IFS=',' read -ra CONTROL_ARRAY <<< "$CONTROLS_STR"
CONTROL_JSON=""
for ctrl in "${CONTROL_ARRAY[@]}"; do
    # Split on '=' to get control type and weight
    IFS='=' read -r control_type control_weight <<< "$ctrl"

    # Default weight to 1.0 if not specified
    if [ -z "$control_weight" ]; then
        control_weight="1.0"
    fi

    echo "  - Using control: $control_type with weight: $control_weight"

    # Build JSON for this control
    if [ -n "$CONTROL_JSON" ]; then
        CONTROL_JSON="${CONTROL_JSON},"
    fi
    CONTROL_JSON="${CONTROL_JSON}
    \"${control_type}\": {
        \"control_weight\": ${control_weight}
    }"
done

# Convert absolute path for video
VIDEO_PATH=$(realpath "$INPUT_VIDEO")

echo ""
echo "Starting Cosmos-Transfer2.5 inference..."
echo "Video: $INPUT_VIDEO"
echo "Output directory: $OUTPUT_DIR"
echo "Number of prompts: $NUM_PROMPTS"
echo ""

# Generate config files and run inference for each prompt
for i in $(seq 1 $NUM_PROMPTS); do
    SAMPLE_NAME=$(printf "sample_%02d" $i)
    CONFIG_FILE="${CONFIG_DIR}/${SAMPLE_NAME}_config.json"

    echo "Processing prompt $i/$NUM_PROMPTS: $PROMPT"

    # Build negative prompt JSON field
    NEGATIVE_PROMPT_JSON=""
    if [ -n "$NEGATIVE_PROMPT" ]; then
        NEGATIVE_PROMPT_JSON=",
    \"negative_prompt\": \"$NEGATIVE_PROMPT\""
    fi

    # Create JSON config file
    cat > "$CONFIG_FILE" << EOF
{
    "name": "${SAMPLE_NAME}",
    "prompt": "$PROMPT",
    "video_path": "$VIDEO_PATH",
    "guidance": $GUIDANCE,
    "seed": $SEED,
    "resolution": "$RESOLUTION"${NEGATIVE_PROMPT_JSON},
${CONTROL_JSON}
}
EOF

    echo "Created config: $CONFIG_FILE"
    cat "$CONFIG_FILE"
    echo ""

    # Detect number of GPUs and setup command
    if [ "$USE_MULTI_GPU" = true ]; then
        NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
        if [ "$NUM_GPUS" -gt 1 ]; then
            echo "Running inference with $NUM_GPUS GPUs..."
            PYTHON_CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500"
        else
            echo "Only 1 GPU detected, running single GPU mode..."
            PYTHON_CMD="python"
        fi
    else
        echo "Running inference with single GPU..."
        PYTHON_CMD="python"
    fi

    # Run the inference CLI
    # The examples/inference.py script uses tyro.cli(Args) where Args contains:
    #   - input_files (-i): list of JSON config files
    #   - setup.output_dir (-o/--output-dir): output directory (REQUIRED)
    #   - setup.model: model variant (default based on controls)
    #   - setup.disable_guardrails: disable safety filters
    #   - Other optional setup args (guidance, seed, etc.)

    # Build guardrail flag
    GUARDRAIL_FLAG=""
    if [ "$DISABLE_GUARDRAILS" = true ]; then
        echo "⚠️  WARNING: Guardrails are DISABLED - safety filters will not run"
        GUARDRAIL_FLAG="--setup.disable-guardrails"
    fi

    $PYTHON_CMD -m examples.inference \
        -i "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        $GUARDRAIL_FLAG \
        || {
            echo "Error: Inference failed for sample $i"
            exit 1
        }

    echo ""
    echo "Completed sample $i"
    echo "---"
done

echo ""
echo "============================================"
echo "All inference tasks completed!"
echo "Outputs saved to: $OUTPUT_DIR"
echo "Configs saved to: $CONFIG_DIR"
echo "============================================"
