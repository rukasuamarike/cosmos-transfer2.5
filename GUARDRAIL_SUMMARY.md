# Guardrail System - Complete Summary

## Quick Reference Table

| What You Want | Command Flag | Python CLI Flag |
|---------------|--------------|-----------------|
| Disable all guardrails | `--disable-guardrails` | `--setup.disable-guardrails` |
| Continue on errors | N/A | `--setup.keep-going` |
| Offload to CPU (save VRAM) | N/A | `--setup.offload-guardrail-models` (default: true) |

## The Guardrail Parameter Location

**Source File:** `cosmos_transfer2/config.py`

**Line 195-205:**
```python
class CommonSetupArguments(pydantic.BaseModel):
    # ... other parameters ...

    disable_guardrails: bool = False
    """Disable guardrails if this is set to True."""

    offload_guardrail_models: bool = True
    """Offload guardrail models to CPU to save GPU memory."""

    keep_going: bool = False
    """Keep going if an error occurs."""

    def enable_guardrails(self) -> bool:
        return not self.disable_guardrails
```

## How Guardrails Are Initialized

**Source File:** `cosmos_transfer2/inference.py`

**Line 62-73:**
```python
class Control2WorldInference:
    def __init__(self, args: SetupArguments, batch_hint_keys: list[str]):
        # ... model loading ...

        if args.enable_guardrails and self.device_rank == 0:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            self.text_guardrail_runner = None  # ← Disabled
            self.video_guardrail_runner = None  # ← Disabled
```

## Where Guardrails Run

### 1. Text Guardrail Check (Line 124-145)

**Checks:** Prompt and negative prompt before generation

```python
if self.text_guardrail_runner is not None:
    log.info("Running guardrail check on prompt...")

    if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
        log.critical("Guardrail blocked control2world generation. Prompt: {prompt}")
        if self.setup_args.keep_going:
            return None
        else:
            exit(1)  # ← Fails here if prompt is unsafe
```

### 2. Video Guardrail Check (Line 194-212) ← YOUR ERROR

**Checks:** Generated video after inference completes

```python
if self.video_guardrail_runner is not None:
    log.info("Running guardrail check on video...")
    frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)

    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)

    if processed_frames is None:
        log.critical("Guardrail blocked video2world generation.")  # ← LINE 200 - YOUR ERROR
        if self.setup_args.keep_going:
            return None
        else:
            exit(1)  # ← Exits here, video discarded
```

## Guardrail Models Used

**Source File:** `cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/common/presets.py`

### Text Guardrails
```python
def create_text_guardrail_runner(offload_model_to_cpu: bool = False):
    return GuardrailRunner(
        safety_models=[
            Blocklist(),                                        # Simple word blocklist
            Qwen3Guard(offload_model_to_cpu=offload_model_to_cpu),  # LLM-based moderation
        ]
    )
```

### Video Guardrails
```python
def create_video_guardrail_runner(offload_model_to_cpu: bool = False):
    return GuardrailRunner(
        safety_models=[
            VideoContentSafetyFilter(offload_model_to_cpu=offload_model_to_cpu)  # Content safety
        ],
        postprocessors=[
            RetinaFaceFilter(offload_model_to_cpu=offload_model_to_cpu)  # Face blur
        ],
    )
```

## run_inferencev2.sh Implementation

The updated script now includes guardrail disable support:

**Default value (line 62):**
```bash
DISABLE_GUARDRAILS=false
```

**Argument parsing (line 110-113):**
```bash
--disable-guardrails)
    DISABLE_GUARDRAILS=true
    shift
    ;;
```

**Flag building (line 236-241):**
```bash
GUARDRAIL_FLAG=""
if [ "$DISABLE_GUARDRAILS" = true ]; then
    echo "⚠️  WARNING: Guardrails are DISABLED - safety filters will not run"
    GUARDRAIL_FLAG="--setup.disable-guardrails"
fi
```

**CLI invocation (line 243-246):**
```bash
$PYTHON_CMD -m examples.inference \
    -i "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    $GUARDRAIL_FLAG \  # ← Conditionally added
```

## Testing Your Fix

### Test 1: Verify Flag Works

```bash
bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
    'edge=0.5' \
    'test prompt' \
    -o outputs/test_guardrail \
    --disable-guardrails
```

**Expected output:**
```
⚠️  WARNING: Guardrails are DISABLED - safety filters will not run
...
[WARNING] Guardrail checks on prompt are disabled
[WARNING] Guardrail checks on video are disabled
[SUCCESS] Generated video saved to outputs/test_guardrail/sample_01.mp4
```

### Test 2: Verify Without Flag (Should Still Work for Safe Content)

```bash
bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
    'edge=0.5' \
    'a simple robot' \
    -o outputs/test_with_guardrail
    # No --disable-guardrails flag
```

**Expected output (if content is safe):**
```
[INFO] Running guardrail check on prompt...
[SUCCESS] Passed guardrail on prompt
[INFO] Running guardrail check on video...
[SUCCESS] Passed guardrail on generated video
[SUCCESS] Generated video saved to outputs/test_with_guardrail/sample_01.mp4
```

## Memory Usage Comparison

| Configuration | VRAM Usage | Inference Time |
|---------------|------------|----------------|
| Guardrails enabled | +3-7GB | +10-50 seconds |
| Guardrails enabled + CPU offload | +1-2GB | +20-60 seconds |
| Guardrails disabled | +0GB | +0 seconds |

## All Available Setup Parameters

From `cosmos_transfer2/config.py:174-203`:

```python
class CommonSetupArguments:
    # Required
    output_dir: Path  # -o, --output-dir

    # Model selection
    model: Literal["edge", "depth", "seg", "vis"] = "edge"
    checkpoint_path: str | None = None
    experiment: str | None = None
    config_file: str = "cosmos_transfer2/_src/predict2/configs/video2world/config.py"

    # Parallelism
    context_parallel_size: int | None = None  # Multi-GPU

    # Guardrails
    disable_guardrails: bool = False           # ← DISABLE ALL GUARDRAILS
    offload_guardrail_models: bool = True      # ← CPU offload to save VRAM

    # Error handling
    keep_going: bool = False                   # ← Continue on errors

    # Profiling
    profile: bool = False                      # ← Enable profiler
```

## Usage with tyro CLI

The Python CLI uses tyro, which supports nested arguments:

```bash
# Basic
python -m examples.inference -i config.json --output-dir outputs

# With setup parameters (nested with --setup. prefix)
python -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.disable-guardrails \
    --setup.keep-going \
    --setup.context-parallel-size 8

# With overrides (for batch processing)
python -m examples.inference \
    -i config1.json config2.json \
    --output-dir outputs \
    --overrides.guidance 5 \
    --overrides.seed 12345
```

## Files Updated

1. ✅ `run_inferencev2.sh` - Added `--disable-guardrails` flag
2. ✅ `GUARDRAIL_GUIDE.md` - Complete guardrail documentation
3. ✅ `QUICK_FIX_GUARDRAIL.md` - Quick reference for your specific error
4. ✅ `GUARDRAIL_SUMMARY.md` - This file (technical reference)

## Next Steps

1. **Test the fix:**
   ```bash
   bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
       'depth=0.8,edge=0.5' \
       'a robot picks up a blue towel' \
       -o outputs/robot_test \
       --disable-guardrails
   ```

2. **Check the output:**
   ```bash
   ls -lh outputs/robot_test/sample_01/
   # Should see output.mp4, control videos, etc.
   ```

3. **View logs:**
   ```bash
   cat outputs/robot_test/console.log
   # Should show WARNING about guardrails disabled, no CRITICAL errors
   ```

## Documentation Index

- 📖 `QUICK_FIX_GUARDRAIL.md` - Start here for immediate fix
- 📚 `GUARDRAIL_GUIDE.md` - Comprehensive guardrail documentation
- 🏗️ `INFERENCE_PIPELINE_GUIDE.md` - Full pipeline architecture
- 🚀 `QUICK_START_INFERENCE.md` - Usage examples
- 📋 `GUARDRAIL_SUMMARY.md` - This file (technical reference)

## Support

If you still have issues after disabling guardrails:

1. Check VRAM usage: `nvidia-smi`
2. View full logs: `outputs/{sample_name}/debug.log`
3. Verify model downloads completed
4. Try lower resolution: `--resolution 480`
5. Enable keep-going: `--setup.keep-going`
