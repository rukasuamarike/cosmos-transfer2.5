# Quick Fix: Guardrail Blocking Your Video

## 🚨 The Error

```
[CRITICAL] Guardrail blocked video2world generation.
```

## ✅ The Fix (3 Methods)

### Method 1: Using run_inferencev2.sh ⭐ EASIEST

```bash
bash run_inferencev2.sh INPUT.mp4 "depth=0.8,edge=0.5" "your prompt" \
    -o outputs \
    --disable-guardrails  # <-- ADD THIS
```

### Method 2: Direct Python CLI

```bash
python -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.disable-guardrails  # <-- ADD THIS
```

### Method 3: Modify Your Existing Script

If you have a custom script calling the inference, add to your Python command:

```bash
# Before (fails with guardrail error)
python -m examples.inference -i config.json --output-dir outputs

# After (bypasses guardrails)
python -m examples.inference -i config.json --output-dir outputs --setup.disable-guardrails
```

## What You'll See

**Before (with error):**
```
[10-24 18:06:56|INFO] Running guardrail check on video...
[10-24 18:06:56|CRITICAL] Guardrail blocked video2world generation.
Error: Script exited with code 1
```

**After (working):**
```
⚠️  WARNING: Guardrails are DISABLED - safety filters will not run
[10-24 18:10:23|WARNING] Guardrail checks on prompt are disabled
[10-24 18:11:45|WARNING] Guardrail checks on video are disabled
[10-24 18:12:03|SUCCESS] Generated video saved to outputs/sample_01.mp4  ✅
```

## Complete Example

Your exact command that was failing:

```bash
bash run_inference.sh assets/robot_example/robot_input.mp4 \
    'depth=0.8,edge=0.5' \
    'a robot picks up a blue towel' \
    --o outputs/robo_example
```

**Fixed version with run_inferencev2.sh:**

```bash
bash run_inferencev2.sh assets/robot_example/robot_input.mp4 \
    'depth=0.8,edge=0.5' \
    'a robot picks up a blue towel' \
    -o outputs/robo_example \
    --disable-guardrails  # <-- FIXED!
```

## Why Was It Blocked?

The guardrail system runs two safety filters:

1. **Text Guardrail** (on prompt) - Usually passes
2. **Video Guardrail** (on generated video) - This is what blocked you

The video guardrail:
- Checks for unsafe visual content (violence, adult content, etc.)
- Can have **false positives** on benign content
- Runs **after** generation completes
- **Discards** your video if it fails 😞

## Is It Safe to Disable?

**For your robot demo: YES ✅**

It's safe to disable guardrails when:
- Internal research/development
- Benign industrial content (robots, products)
- Private datasets
- Debugging/testing

**Keep enabled for:**
- Production user-facing apps
- Public demos
- Unknown user inputs
- Compliance requirements

## Other Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--disable-guardrails` | Disable all safety filters | `false` |
| `--setup.keep-going` | Continue on errors (don't exit) | `false` |
| `--setup.offload-guardrail-models` | Move guardrail models to CPU | `true` |

## Still Need Help?

See the full guide:
- `GUARDRAIL_GUIDE.md` - Complete documentation
- `INFERENCE_PIPELINE_GUIDE.md` - Architecture details
- `QUICK_START_INFERENCE.md` - Usage examples

## TL;DR

**Add `--disable-guardrails` to your command and you're good to go! 🚀**
