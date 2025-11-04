# Cosmos-Transfer2.5 Checkpoint Download Guide

This guide explains how to pre-download all required model checkpoints before running inference.

## Why Pre-download Checkpoints?

By default, Cosmos-Transfer2.5 downloads checkpoints automatically during the first inference run. However, pre-downloading has several advantages:

1. **Faster first inference**: No waiting for downloads during inference
2. **Better control**: Download specific models you need
3. **Network flexibility**: Download when you have better internet
4. **Multi-GPU setups**: Avoid download conflicts when multiple workers try to download simultaneously

## What Gets Downloaded?

The inference pipeline requires several checkpoints:

| Component | Size | Description |
|-----------|------|-------------|
| Control models (depth, edge, seg, blur) | ~8GB total | Video-to-video transfer models |
| T5 text encoder | ~42GB | google-t5/t5-11b for prompt processing |
| VAE tokenizer | ~1GB | Wan2.1 VAE for encoding/decoding |
| Guardrail models | ~15GB | Safety filters (text and video) |
| **Total** | **~66GB** | |

## Quick Start

### Download All Checkpoints (Recommended)

```bash
# Activate your conda environment first
conda activate cosmos-transfer2.5

# Download everything
bash download_checkpoints.sh
```

or

```bash
python download_checkpoints.py
```

### Download Specific Models Only

```bash
# Only control models (no text encoder, no guardrails)
bash download_checkpoints.sh --models depth,edge,seg,blur

# Skip guardrails (saves ~15GB)
bash download_checkpoints.sh --skip-guardrails

# Only depth and edge models
bash download_checkpoints.sh --models depth,edge
```

## Usage Options

### Python Script

```bash
python download_checkpoints.py [OPTIONS]

Options:
  --skip-guardrails       Skip downloading guardrail checkpoints
  --models MODEL_LIST     Comma-separated list: depth,edge,seg,blur,all
  --help                 Show help message
```

### Bash Script

```bash
bash download_checkpoints.sh [OPTIONS]

Same options as Python script
```

## Download Locations

Checkpoints are downloaded to:
- **HuggingFace cache**: `~/.cache/huggingface/`
- **Imaginaire cache**: `~/.cache/imaginaire/`

These directories are automatically managed by the caching system.

## Checkpoint Details

### Control Models (Transfer2.5)

These are the main video-to-video transfer models:

- **depth** (UUID: `0f214f66-ae98-43cf-ab25-d65d09a7e68f`)
  - Depth map-based control

- **edge** (UUID: `ecd0ba00-d598-4f94-aa09-e8627899c431`)
  - Edge/Canny detection control

- **seg** (UUID: `fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab`)
  - Segmentation-based control

- **blur** (UUID: `20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1`)
  - Blur/visibility control

### Text Encoder

- **T5-11B** (UUID: `4dbf13c6-1d30-4b02-99d6-75780dd8b744`)
  - Repository: `google-t5/t5-11b`
  - Used for encoding text prompts

### Tokenizer

- **Wan2.1 VAE** (UUID: `685afcaa-4de2-42fe-b7b9-69f7a2dee4d8`)
  - Used for video encoding/decoding

### Guardrails

- **Cosmos-Guardrail1** (UUID: `9c7b7da4-2d95-45bb-9cb8-2eed954e9736`)
  - Text and video safety filters
  - Includes Qwen3Guard and VideoContentSafetyFilter

## Integration with Inference

The download script triggers the same download mechanism used during inference:

```
Inference Flow:
run_inferencev2.sh
  └─> examples/inference.py
      └─> cosmos_transfer2/inference.py (Control2WorldInference)
          ├─> MODEL_CHECKPOINTS[variant].path  ← Downloads control models
          ├─> guardrail_presets.create_*_runner()  ← Downloads guardrails
          └─> ControlVideo2WorldInference
              ├─> load_model_from_checkpoint()  ← Downloads via get_checkpoint_path()
              └─> get_t5_from_prompt()  ← Downloads T5 model
```

## Troubleshooting

### "PyTorch is not installed" Error

```bash
# Make sure you're in the right environment
conda activate cosmos-transfer2.5

# Verify torch is installed
python -c "import torch; print(torch.__version__)"

# If not installed, install the package
pip install -e .
```

### Download Fails / Network Issues

The download script uses HuggingFace Hub's built-in retry logic. If downloads fail:

1. Check your internet connection
2. Ensure you have enough disk space (~66GB)
3. Check HuggingFace status: https://status.huggingface.co/
4. Try downloading individual models:
   ```bash
   bash download_checkpoints.sh --models depth
   bash download_checkpoints.sh --models edge
   # etc.
   ```

### Already Downloaded - How to Verify?

```bash
# Check HuggingFace cache
ls -lh ~/.cache/huggingface/hub/

# Check Imaginaire cache
ls -lh ~/.cache/imaginaire/

# Run download script - it will skip already downloaded files
bash download_checkpoints.sh
```

### Disk Space Issues

If you're low on disk space:

1. Skip guardrails to save ~15GB:
   ```bash
   bash download_checkpoints.sh --skip-guardrails
   ```

2. Download only models you need:
   ```bash
   bash download_checkpoints.sh --models edge,depth
   ```

## Manual Checkpoint Management

If you need to manually manage checkpoints:

### Clear Cache

```bash
# Clear all HuggingFace downloads
rm -rf ~/.cache/huggingface/

# Clear Imaginaire cache
rm -rf ~/.cache/imaginaire/
```

### Move Checkpoints

```bash
# Set custom cache directory (before running script)
export HF_HOME=/path/to/custom/cache
export IMAGINAIRE_CACHE_DIR=/path/to/custom/cache

bash download_checkpoints.sh
```

## Advanced: Checkpoint Database

All checkpoints are defined in:
```
cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py
```

Each checkpoint has:
- **UUID**: Unique identifier
- **S3 URI**: Internal NVIDIA S3 location (for INTERNAL users)
- **HuggingFace**: Public repository location
- **Metadata**: Model info (resolution, FPS, etc.)

The download mechanism automatically selects:
- S3 if `INTERNAL=True`
- HuggingFace otherwise

## Related Files

- `download_checkpoints.py` - Main Python download script
- `download_checkpoints.sh` - Bash wrapper script
- `cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py` - Checkpoint database
- `cosmos_transfer2/config.py` - Model variant mappings
- `cosmos_transfer2/inference.py` - Inference logic that uses checkpoints

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify your environment setup
3. Check the main README.md for system requirements
4. File an issue on GitHub with:
   - Error message
   - System info (OS, GPU, disk space)
   - Download command used
