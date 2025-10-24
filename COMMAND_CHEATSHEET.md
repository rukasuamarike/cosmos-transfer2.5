# Cosmos-Transfer2.5 Command Cheatsheet

## 🚨 Your Exact Fix

### What Was Failing
```bash
bash run_inference.sh assets/robot_example/robot_input.mp4 \
    'depth=0.8,edge=0.5' \
    'a robot picks up a blue towel' \
    --o outputs/robo_example

# ERROR: Guardrail blocked video2world generation.
```

### What Will Work
```bash
bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
    'depth=0.8,edge=0.5' \
    'a robot picks up a blue towel' \
    -o outputs/robo_example \
    --disable-guardrails  # ← THE FIX
```

---

## Quick Command Templates

### 1. Basic Single Control
```bash
bash run_inferencev2.sh INPUT.mp4 "edge=0.8" "your prompt" -o OUTPUT_DIR
```

### 2. Multiple Controls
```bash
bash run_inferencev2.sh INPUT.mp4 "depth=0.8,edge=0.5" "your prompt" -o OUTPUT_DIR
```

### 3. With Guardrails Disabled (For Your Use Case)
```bash
bash run_inferencev2.sh INPUT.mp4 "depth=0.8,edge=0.5" "your prompt" \
    -o OUTPUT_DIR \
    --disable-guardrails
```

### 4. Full Options
```bash
bash run_inferencev2.sh INPUT.mp4 "depth=0.8,edge=0.5" "your prompt" \
    -o OUTPUT_DIR \
    --guidance 4 \
    --seed 42 \
    --resolution 720 \
    --num-prompts 3 \
    --disable-guardrails \
    --multi-gpu
```

### 5. Direct Python CLI
```bash
# Create JSON config first
cat > config.json << EOF
{
    "name": "sample_01",
    "prompt": "your prompt here",
    "video_path": "/workspace/INPUT.mp4",
    "depth": {"control_weight": 0.8},
    "edge": {"control_weight": 0.5}
}
EOF

# Run inference
python -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.disable-guardrails
```

---

## Control Syntax

| Control Type | Syntax | Auto-computed? |
|-------------|---------|----------------|
| Edge detection | `"edge=0.8"` | Yes |
| Depth map | `"depth=0.9"` | Yes |
| Segmentation | `"seg=0.7"` | Yes (needs prompt) |
| Visibility/blur | `"vis=0.6"` | Yes |

### Multiple Controls
```bash
# Just comma-separate
"depth=0.8,edge=0.5,seg=0.6"

# Or omit weights (defaults to 1.0)
"depth,edge"
```

---

## Common Flags Reference

| Flag | Description | Example |
|------|-------------|---------|
| `-o, --output` | Output directory | `-o outputs/test` |
| `--guidance N` | CFG scale (0-7) | `--guidance 4` |
| `--seed N` | Random seed | `--seed 42` |
| `--resolution` | 480 or 720 | `--resolution 720` |
| `--num-prompts N` | Generate N variations | `--num-prompts 5` |
| `--negative-prompt` | Negative prompt | `--negative-prompt "blurry"` |
| `--disable-guardrails` | **Disable safety filters** | `--disable-guardrails` |
| `--multi-gpu` | Use all GPUs | `--multi-gpu` |
| `--help` | Show help | `--help` |

---

## Python CLI Setup Flags

For `python -m examples.inference`:

| Flag | Description | Example |
|------|-------------|---------|
| `-i, --input-files` | JSON config files | `-i config.json` |
| `--output-dir` | Output directory (required) | `--output-dir outputs` |
| `--setup.model` | Model variant | `--setup.model depth` |
| `--setup.disable-guardrails` | **Disable guardrails** | `--setup.disable-guardrails` |
| `--setup.keep-going` | Continue on errors | `--setup.keep-going` |
| `--setup.context-parallel-size` | Multi-GPU | `--setup.context-parallel-size 8` |
| `--overrides.guidance` | Override guidance | `--overrides.guidance 5` |
| `--overrides.seed` | Override seed | `--overrides.seed 999` |

---

## Troubleshooting Commands

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

### View Logs
```bash
# Console log
cat outputs/OUTPUT_DIR/console.log

# Debug log (verbose)
cat outputs/OUTPUT_DIR/debug.log

# Tail logs in real-time
tail -f outputs/OUTPUT_DIR/console.log
```

### Test Guardrails Disabled
```bash
# Look for these warnings in output
grep "Guardrails are DISABLED" outputs/*/console.log
grep "Guardrail checks.*disabled" outputs/*/console.log
```

### Check Outputs
```bash
# List generated files
ls -lh outputs/sample_01/

# Should see:
# - output.mp4 (main video)
# - sample_01_control_depth.mp4
# - sample_01_control_edge.mp4
# - sample_01.json (config)
# - sample_01.txt (prompt)
```

---

## Multi-GPU Examples

### Automatic Detection
```bash
bash run_inferencev2.sh INPUT.mp4 "edge" "prompt" \
    --multi-gpu  # Automatically detects all GPUs
```

### Manual torchrun
```bash
# 8 GPUs
torchrun --nproc_per_node=8 \
    -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.context-parallel-size 8 \
    --setup.disable-guardrails
```

---

## JSON Config Examples

### Minimal
```json
{
    "name": "sample_01",
    "prompt": "a robot in a factory",
    "video_path": "/workspace/input.mp4",
    "edge": {"control_weight": 0.8}
}
```

### Full Options
```json
{
    "name": "robot_demo",
    "prompt": "a humanoid robot picking up a blue towel",
    "video_path": "/workspace/assets/robot_input.mp4",
    "negative_prompt": "blurry, low quality, pixelated",
    "guidance": 4,
    "seed": 2025,
    "resolution": "720",
    "num_conditional_frames": 1,
    "num_video_frames_per_chunk": 93,
    "num_steps": 35,
    "depth": {
        "control_weight": 0.8
    },
    "edge": {
        "control_weight": 0.5,
        "preset_edge_threshold": "medium"
    }
}
```

### Multiple Configs for Batch Processing
```bash
# Create multiple configs
cat > config1.json << EOF
{"name": "test1", "prompt": "prompt1", ...}
EOF

cat > config2.json << EOF
{"name": "test2", "prompt": "prompt2", ...}
EOF

# Process all at once
python -m examples.inference \
    -i config1.json config2.json \
    --output-dir outputs \
    --setup.disable-guardrails
```

---

## File Locations Quick Reference

| File | Location |
|------|----------|
| Main inference script | `examples/inference.py` |
| Config definitions | `cosmos_transfer2/config.py` |
| Inference orchestrator | `cosmos_transfer2/inference.py` |
| Diffusion pipeline | `cosmos_transfer2/_src/transfer2/inference/inference_pipeline.py` |
| Guardrail presets | `cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/common/presets.py` |
| Shell wrapper v2 | `run_inferencev2.sh` |

---

## Default Values

| Parameter | Default | Location |
|-----------|---------|----------|
| Guidance | 3 | `config.py:261` |
| Seed | 2025 | `config.py:414` |
| Resolution | "720" | `config.py:397` |
| Num conditional frames | 1 | `config.py:399` |
| Num steps | 35 | `config.py:401` |
| Disable guardrails | False | `config.py:195` |
| Keep going | False | `config.py:200` |
| Offload guardrails | True | `config.py:198` |

---

## Performance Optimization

### Faster (Lower Quality)
```bash
bash run_inferencev2.sh INPUT.mp4 "edge" "prompt" \
    -o outputs \
    --resolution 480 \      # Lower res
    --disable-guardrails    # Skip safety checks
```

### Best Quality (Slower)
```bash
bash run_inferencev2.sh INPUT.mp4 "depth=0.9,edge=0.8" "prompt" \
    -o outputs \
    --resolution 720 \
    --guidance 5 \          # Higher guidance
    --multi-gpu             # Use all GPUs
```

### Balanced
```bash
bash run_inferencev2.sh INPUT.mp4 "edge=0.8" "prompt" \
    -o outputs \
    --resolution 720 \
    --guidance 3 \
    --disable-guardrails \
    --multi-gpu
```

---

## Help Commands

```bash
# Shell script help
bash run_inferencev2.sh --help

# Python CLI help
python -m examples.inference --help

# View specific config schema
python -c "from cosmos_transfer2.config import InferenceArguments; help(InferenceArguments)"
```

---

## Copy-Paste Solutions

### For Your Specific Robot Example
```bash
cd /workspace  # Or your project root

bash run_inferencev2.sh \
    assets/robot_example/robot_input.mp4 \
    'depth=0.8,edge=0.5' \
    'a robot picks up a blue towel' \
    -o outputs/robo_example \
    --disable-guardrails
```

### Generic Template (Fill in the blanks)
```bash
bash run_inferencev2.sh \
    PATH/TO/INPUT.mp4 \
    'CONTROL1=WEIGHT1,CONTROL2=WEIGHT2' \
    'YOUR PROMPT HERE' \
    -o outputs/YOUR_OUTPUT_DIR \
    --disable-guardrails
```

---

## Quick Checks

### Did guardrails disable work?
```bash
grep -i "guardrails are disabled" outputs/*/console.log
# Should see: ⚠️  WARNING: Guardrails are DISABLED
```

### Did generation complete?
```bash
ls outputs/sample_01/output.mp4
# If exists, success!
```

### Check generation time
```bash
grep "time per chunk" outputs/*/console.log
```

### Check VRAM usage during run
```bash
watch -n 0.5 nvidia-smi
# Monitor memory usage in real-time
```

---

## Documentation Tree

```
QUICK_FIX_GUARDRAIL.md          ← Start here for immediate fix
    ↓
COMMAND_CHEATSHEET.md           ← This file (quick commands)
    ↓
QUICK_START_INFERENCE.md        ← Usage examples and tutorials
    ↓
GUARDRAIL_GUIDE.md              ← Deep dive on guardrails
    ↓
INFERENCE_PIPELINE_GUIDE.md     ← Full architecture docs
    ↓
GUARDRAIL_SUMMARY.md            ← Technical reference
```

---

**Bottom Line:** Add `--disable-guardrails` to your command and you're done! 🎉
