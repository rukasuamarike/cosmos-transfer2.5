# Quick Start: Running Inference with run_inferencev2.sh

## TL;DR

```bash
# Basic usage
bash run_inferencev2.sh input.mp4 "depth=0.8,edge=0.5" "your prompt here" -o outputs

# With custom options
bash run_inferencev2.sh assets/robot.mp4 "edge=0.8" "robot in factory" \
    -o my_output \
    --guidance 4 \
    --seed 42 \
    --num-prompts 3 \
    --multi-gpu
```

## What was fixed from the original run_inference.sh?

### The Problem
```bash
# Old script error:
╭─ Required options ────────────────────────────────────╮
│ The following arguments are required: -o/--output-dir │
╰───────────────────────────────────────────────────────╯
```

### The Root Cause

The Python CLI (`examples/inference.py`) requires `--output-dir` as a **mandatory argument**:

```python
# From config.py line 180
class CommonSetupArguments:
    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory."""  # NO DEFAULT - REQUIRED!
```

The old script was either:
1. Not passing `-o` at all, OR
2. Passing it incorrectly to the Python module

### The Fix

**run_inferencev2.sh always passes `--output-dir` correctly:**

```bash
# Default to timestamped directory if user doesn't specify
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs_${TIMESTAMP}"
fi

# Always pass it to Python CLI
python -m examples.inference \
    -i "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR"  # <-- This is REQUIRED!
```

## Usage Examples

### Example 1: Basic Single Control
```bash
bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
    "edge=0.8" \
    "a robot picks up a blue towel"
```

**Output:**
- Creates `outputs_20251024_173045/` (timestamped)
- Generates `sample_01/output.mp4`
- Saves control visualization `sample_01_control_edge.mp4`

### Example 2: Multiple Controls with Custom Output
```bash
bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
    "depth=0.8,edge=0.5" \
    "a robot picks up a blue towel" \
    -o outputs/robot_demo
```

**Output:**
- Creates `outputs/robot_demo/`
- Uses both depth and edge controls
- Generates control visualizations for both

### Example 3: Multiple Prompt Variations
```bash
bash run_inferencev2.sh assets/video.mp4 \
    "depth" \
    "industrial warehouse scene" \
    --num-prompts 5 \
    --guidance 4 \
    --seed 2025 \
    -o outputs/warehouse_variations
```

**Output:**
- Generates 5 different variations:
  - `sample_01/output.mp4`
  - `sample_02/output.mp4`
  - ... (each with different seed)
- All with same prompt but different random seeds

### Example 4: Multi-GPU for Faster Generation
```bash
bash run_inferencev2.sh assets/long_video.mp4 \
    "edge=0.7" \
    "futuristic city streets" \
    --multi-gpu \
    -o outputs/city_fast
```

**What happens:**
- Detects all available GPUs (e.g., 8 GPUs)
- Uses `torchrun --nproc_per_node=8`
- Enables context parallelism for faster processing

### Example 5: Full Custom Parameters
```bash
bash run_inferencev2.sh assets/input.mp4 \
    "depth=0.9,seg=0.6" \
    "modern apartment interior" \
    -o outputs/apartment \
    --guidance 5 \
    --seed 12345 \
    --resolution 720 \
    --num-prompts 3 \
    --negative-prompt "blurry, low quality, pixelated" \
    --multi-gpu
```

## Control Types

| Control | Description | Auto-computed? | Parameters |
|---------|-------------|----------------|------------|
| `edge` | Canny edge detection | Yes | `preset_edge_threshold`: very_low, low, medium, high, very_high |
| `depth` | Depth map (Depth-Anything) | Yes | - |
| `seg` | Segmentation mask (SAM2) | Yes (needs prompt) | `control_prompt`: text describing what to segment |
| `vis` | Visibility/blur mask | Yes | `preset_blur_strength`: very_low, low, medium, high, very_high |

### Using Pre-computed Controls

If you have pre-computed control videos, use the JSON config directly:

```json
{
    "name": "sample_01",
    "prompt": "robot in factory",
    "video_path": "input.mp4",
    "depth": {
        "control_weight": 0.8,
        "control_path": "precomputed/depth_map.mp4"
    }
}
```

Then call:
```bash
python -m examples.inference -i config.json --output-dir outputs
```

## Direct Python CLI Usage (Without Shell Script)

If you prefer to skip the shell wrapper:

```bash
# Create JSON config
cat > my_config.json << EOF
{
    "name": "my_sample",
    "prompt": "a robot in a modern factory",
    "video_path": "/workspace/assets/robot_input.mp4",
    "guidance": 3,
    "seed": 2025,
    "depth": {"control_weight": 0.8},
    "edge": {"control_weight": 0.5}
}
EOF

# Run inference directly
python -m examples.inference \
    -i my_config.json \
    --output-dir outputs/my_test

# Or with multiple configs
python -m examples.inference \
    -i config1.json config2.json config3.json \
    --output-dir outputs/batch_test

# With model override
python -m examples.inference \
    -i my_config.json \
    --output-dir outputs \
    --setup.model depth \
    --setup.guidance 5
```

## Troubleshooting

### 1. "Input video not found"
```bash
# Use absolute paths
VIDEO=$(realpath assets/robot_input.mp4)
bash run_inferencev2.sh "$VIDEO" "edge" "robot"
```

### 2. "No controls provided"
```bash
# Always specify at least one control
bash run_inferencev2.sh input.mp4 "edge" "prompt"  # ✓ Good
bash run_inferencev2.sh input.mp4 "" "prompt"      # ✗ Bad
```

### 3. Out of memory
```bash
# Try lower resolution
bash run_inferencev2.sh input.mp4 "edge" "prompt" --resolution 480

# Or enable multi-GPU
bash run_inferencev2.sh input.mp4 "edge" "prompt" --multi-gpu
```

### 4. Want to see all options
```bash
# Show help
bash run_inferencev2.sh --help

# Or inspect generated config
cat cosmos_inference_*/sample_01_config.json
```

## Comparing Old vs New Script

| Feature | run_inference.sh (old) | run_inferencev2.sh (new) |
|---------|------------------------|--------------------------|
| Output dir handling | ❌ Missing/broken | ✓ Required with default |
| Help message | ❌ None | ✓ Comprehensive |
| Multi-GPU support | ❌ Manual | ✓ Automatic detection |
| Control parsing | ✓ Basic | ✓ Enhanced |
| Error handling | ❌ Limited | ✓ Validation & checks |
| Config visibility | ✓ Prints JSON | ✓ Prints + saves to file |
| Documentation | ❌ None | ✓ Inline + guide |

## Next Steps

1. **Read the full pipeline guide:**
   - See `INFERENCE_PIPELINE_GUIDE.md` for architecture deep-dive

2. **Experiment with parameters:**
   - Try different control weights
   - Test various guidance scales
   - Generate multiple variations

3. **Optimize for your use case:**
   - Multi-GPU for speed
   - Lower resolution for prototyping
   - Higher guidance for stronger prompt adherence

4. **Integrate into workflows:**
   - Use JSON configs for reproducibility
   - Script batch processing
   - Automate with CI/CD

## Summary

**The key fix:** `run_inferencev2.sh` always ensures `--output-dir` is passed to the Python CLI, either from user input (`-o my_dir`) or from a sensible default (`outputs_TIMESTAMP`).

This was the missing piece that caused the "required arguments" error in your original run.
