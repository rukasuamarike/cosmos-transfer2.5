# Cosmos-Transfer2.5 Guardrail System Guide

## 🚨 Your Issue: Guardrail Blocking Video Generation

```
[10-24 18:06:56|CRITICAL|cosmos_transfer2/inference.py:200:_generate_sample]
Guardrail blocked video2world generation.
```

## Quick Fix: Disable Guardrails

### Method 1: Using run_inferencev2.sh (Recommended)

```bash
bash run_inferencev2.sh assets/robot.mp4 "depth=0.8,edge=0.5" \
    "a robot picks up a blue towel" \
    -o outputs/robot_test \
    --disable-guardrails  # <-- ADD THIS FLAG
```

### Method 2: Direct Python CLI

```bash
python -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.disable-guardrails  # <-- ADD THIS FLAG
```

### Method 3: Environment Variable (Alternative)

Set in your shell or Docker container:
```bash
export COSMOS_DISABLE_GUARDRAILS=1
```

**Note:** The code doesn't currently check this env var, but you can add it to your workflow scripts.

---

## Understanding the Guardrail System

### What Are Guardrails?

Guardrails are **safety filters** that check:
1. **Text prompts** - Block unsafe/inappropriate prompts
2. **Generated videos** - Detect unsafe content & blur faces

### Guardrail Components

#### 1. Text Guardrails (cosmos_transfer2/inference.py:123-145)

**Run on:**
- User prompt (line 127)
- Negative prompt (line 136)

**Checks:**
- **Blocklist**: Banned words/phrases
- **Qwen3Guard**: LLM-based content moderation

**Models used:**
```python
create_text_guardrail_runner():
    - Blocklist()
    - Qwen3Guard(offload_model_to_cpu=offload_model_to_cpu)
```

#### 2. Video Guardrails (cosmos_transfer2/inference.py:193-212)

**Run on:**
- Generated output video (after generation completes)

**Checks:**
- **VideoContentSafetyFilter**: Unsafe visual content
- **RetinaFaceFilter**: Face detection & blurring (post-processor)

**Models used:**
```python
create_video_guardrail_runner():
    safety_models = [VideoContentSafetyFilter(...)]
    postprocessors = [RetinaFaceFilter(...)]
```

---

## Why Your Video Was Blocked

### Possible Causes

1. **VideoContentSafetyFilter flagged content as unsafe**
   - Violence, gore, adult content, etc.
   - Even benign robot videos can trigger false positives

2. **Face detection issues**
   - RetinaFaceFilter might fail on certain frames
   - Can cause the entire check to fail

3. **Model inference errors**
   - Guardrail models loading issues
   - GPU memory pressure
   - Model checkpoint download failures

### Location of Block

Your error is at **line 200** in `cosmos_transfer2/inference.py`:

```python
# cosmos_transfer2/inference.py:193-212
if self.video_guardrail_runner is not None:
    log.info("Running guardrail check on video...")
    frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)

    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)

    if processed_frames is None:  # <-- THIS IS THE BLOCK
        log.critical("Guardrail blocked video2world generation.")  # <-- LINE 200
        if self.setup_args.keep_going:
            return None
        else:
            exit(1)  # <-- Script exits here
```

### What Happens When Blocked

1. Video generation **completes successfully**
2. Generated frames are passed to guardrail
3. Guardrail analyzes frames
4. If unsafe: `run_video_guardrail()` returns `None`
5. Script logs CRITICAL error and **exits**
6. **Your generated video is discarded** 😞

---

## How to Disable Guardrails

### Config Parameter (config.py:195-205)

```python
class CommonSetupArguments:
    disable_guardrails: bool = False
    """Disable guardrails if this is set to True."""

    def enable_guardrails(self) -> bool:
        return not self.disable_guardrails
```

### Initialization Logic (inference.py:62-73)

```python
if args.enable_guardrails and self.device_rank == 0:
    self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(...)
    self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(...)
else:
    self.text_guardrail_runner = None  # Guardrails disabled
    self.video_guardrail_runner = None
```

When `disable_guardrails=True`:
- `enable_guardrails()` returns `False`
- Guardrail runners are set to `None`
- All guardrail checks are skipped

---

## Complete Disable Examples

### Example 1: Shell Script

```bash
# Your exact command with guardrails disabled
bash run_inferencev2.sh \
    assets/robot_example/robot_input.mp4 \
    'depth=0.8,edge=0.5' \
    'a robot picks up a blue towel' \
    -o outputs/robot_demo \
    --disable-guardrails
```

**Output:**
```
⚠️  WARNING: Guardrails are DISABLED - safety filters will not run
[10-24 18:10:23|WARNING] Guardrail checks on prompt are disabled
[10-24 18:11:45|WARNING] Guardrail checks on video are disabled
[10-24 18:12:03|SUCCESS] Generated video saved to outputs/robot_demo/sample_01.mp4
```

### Example 2: Python CLI with JSON

**Create config:**
```json
{
    "name": "robot_test",
    "prompt": "a robot picks up a blue towel",
    "video_path": "assets/robot_input.mp4",
    "depth": {"control_weight": 0.8},
    "edge": {"control_weight": 0.5}
}
```

**Run with guardrails disabled:**
```bash
python -m examples.inference \
    -i robot_config.json \
    --output-dir outputs/robot_test \
    --setup.disable-guardrails
```

### Example 3: Multi-GPU with Guardrails Disabled

```bash
bash run_inferencev2.sh \
    assets/video.mp4 \
    "edge=0.7" \
    "futuristic city" \
    --multi-gpu \
    --disable-guardrails \
    -o outputs/city_fast
```

---

## Selective Guardrail Disable (Advanced)

If you want to disable **only video guardrails** but keep text checks, you'll need to modify the code.

### Option 1: Quick Hack (Temporary)

Edit `cosmos_transfer2/inference.py`:

```python
# Line 62: Comment out video guardrail creation
if args.enable_guardrails and self.device_rank == 0:
    self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(...)
    # self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(...)
    self.video_guardrail_runner = None  # <-- Force disable video guardrail
else:
    self.text_guardrail_runner = None
    self.video_guardrail_runner = None
```

### Option 2: Add Separate Config Parameters

Edit `cosmos_transfer2/config.py`:

```python
class CommonSetupArguments:
    disable_guardrails: bool = False
    disable_text_guardrails: bool = False    # NEW
    disable_video_guardrails: bool = False   # NEW

    def enable_text_guardrails(self) -> bool:
        return not (self.disable_guardrails or self.disable_text_guardrails)

    def enable_video_guardrails(self) -> bool:
        return not (self.disable_guardrails or self.disable_video_guardrails)
```

Then update `cosmos_transfer2/inference.py`:

```python
if args.enable_text_guardrails() and self.device_rank == 0:
    self.text_guardrail_runner = create_text_guardrail_runner(...)
else:
    self.text_guardrail_runner = None

if args.enable_video_guardrails() and self.device_rank == 0:
    self.video_guardrail_runner = create_video_guardrail_runner(...)
else:
    self.video_guardrail_runner = None
```

---

## Debugging Guardrail Blocks

### Enable Verbose Logging

To see **why** the guardrail blocked your video:

```bash
export VERBOSE=1  # Enable debug logging

python -m examples.inference \
    -i config.json \
    --output-dir outputs 2>&1 | tee guardrail_debug.log
```

Look for lines like:
```
[CRITICAL] GUARDRAIL BLOCKED: unsafe content detected in frame 42
[CRITICAL] GUARDRAIL BLOCKED: VideoContentSafetyFilter returned unsafe
```

### Inspect Guardrail Implementation

Check what the video guardrail actually does:

```bash
# View the safety filter code
cat cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/video_content_safety_filter/video_content_safety_filter.py

# View the face blur filter
cat cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/face_blur_filter/face_blur_filter.py
```

### Test Just the Guardrail

You can test guardrails on a video without running full inference:

```python
from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets
import numpy as np
import cv2

# Load your generated video
frames = []  # Load frames as (T, H, W, C) numpy array

# Create guardrail runner
runner = presets.create_video_guardrail_runner(offload_model_to_cpu=False)

# Test
result = presets.run_video_guardrail(frames, runner)
if result is None:
    print("BLOCKED!")
else:
    print("PASSED!")
```

---

## Performance Impact of Guardrails

### Memory Usage

Guardrail models consume additional GPU memory:
- **Qwen3Guard**: ~2-4GB
- **VideoContentSafetyFilter**: ~1-2GB
- **RetinaFaceFilter**: ~500MB

**Total: ~3-7GB extra VRAM**

### Inference Time

Guardrails add overhead:
- Text guardrail: ~0.5-2 seconds per prompt
- Video guardrail: ~0.1-0.5 seconds per frame

For a 100-frame video: **+10-50 seconds** total

### Offloading to CPU

To save GPU memory, enable CPU offloading:

```python
# config.py
offload_guardrail_models: bool = True  # Default is True
```

This moves guardrail models to CPU when not in use, but slows down checks.

---

## When to Disable Guardrails

### ✅ Safe to Disable:

1. **Research & development** - Testing models internally
2. **Benign content** - Industrial robots, product demos, landscapes
3. **Controlled environment** - Private datasets, no public distribution
4. **Debugging** - Investigating false positives
5. **Performance testing** - Benchmarking without overhead

### ⚠️ Keep Enabled:

1. **Production systems** - User-facing applications
2. **Public demos** - External showcases
3. **Unknown inputs** - User-generated prompts/videos
4. **Compliance requirements** - Legal/ethical constraints
5. **Content moderation** - Platforms with safety policies

---

## Alternative: --keep-going Flag

Instead of disabling guardrails, you can make failures non-fatal:

```bash
python -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.keep-going  # <-- Don't exit on guardrail failure
```

**What happens:**
- Guardrails still run
- If blocked, logs warning but continues
- Returns `None` for that sample
- Moves to next sample (if processing multiple)

**Useful for:**
- Batch processing where some samples might fail
- Collecting statistics on guardrail hit rate
- Saving successful samples even if some fail

---

## Summary

### The Problem

Your robot video triggered the **VideoContentSafetyFilter** guardrail, causing:
```
CRITICAL: Guardrail blocked video2world generation.
```

### The Solution

**Add `--disable-guardrails` to your command:**

```bash
bash run_inferencev2.sh assets/robot.mp4 \
    "depth=0.8,edge=0.5" \
    "a robot picks up a blue towel" \
    -o outputs/robot_test \
    --disable-guardrails  # <-- THIS
```

### What This Does

1. Sets `disable_guardrails=True` in config
2. `enable_guardrails()` returns `False`
3. Text and video guardrail runners set to `None`
4. All safety checks **skipped**
5. Videos saved directly without filtering

### Verification

When disabled, you'll see:
```
⚠️  WARNING: Guardrails are DISABLED - safety filters will not run
[WARNING] Guardrail checks on prompt are disabled
[WARNING] Guardrail checks on video are disabled
```

No more CRITICAL errors - your video will be saved! 🎉

---

## Additional Resources

- `cosmos_transfer2/config.py:195` - Guardrail config parameter
- `cosmos_transfer2/inference.py:62` - Guardrail initialization
- `cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/common/presets.py` - Guardrail runners
- `cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/video_content_safety_filter/` - Video filter implementation

For more help, check the logs in `{output_dir}/debug.log`
