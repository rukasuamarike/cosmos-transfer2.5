# Parallel Inference with Context Parallelism

## Overview

The `parallel_inference()` function in `cosmos_inference.py` provides proper multi-GPU inference with Context Parallelism (CP) for Cosmos Transfer1.

## Memory Efficiency Comparison

| Method | Memory per GPU | Total Memory | Status |
|--------|---------------|--------------|---------|
| `run_cosmos_inference()` (Legacy) | 120-140GB | 240-280GB | ❌ DEPRECATED |
| `parallel_inference()` (New) | 60-70GB | 120-140GB | ✅ RECOMMENDED |

**Why the difference?**
- Legacy: Each GPU loads the full model independently (duplication)
- New: Model weights distributed across GPUs via Context Parallelism (sharing)

## Usage

### Basic Usage

```python
from cosmos_inference import parallel_inference

# This function must be called FROM WITHIN torchrun
parallel_inference(
    input_video_path="input.mp4",
    output_dir="./output",
    prompts=["robot picks up red cube", "robot places blue sphere"],
    batch_size=2,
    workspace_dir="/root/app",
)
```

### Launch with torchrun

```bash
torchrun --nproc_per_node=2 python -c "
from cosmos_inference import parallel_inference
parallel_inference('input.mp4', './output', prompts=['...'])
"
```

Or with a script:

```bash
torchrun --nproc_per_node=2 test_parallel_inference.py
```

### Modal Integration

```python
import modal

app = modal.App("cosmos-transfer1")

@app.function(
    gpu=modal.gpu.H100(count=2),
    timeout=3600,
)
def inference_endpoint(video_bytes: bytes, prompts: list[str]):
    """
    Modal automatically handles torchrun when gpu count > 1.
    Just call parallel_inference() directly.
    """
    import tempfile
    import os
    from cosmos_inference import parallel_inference

    rank = int(os.environ.get("RANK", "0"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input
        input_path = os.path.join(tmpdir, "input.mp4")
        with open(input_path, "wb") as f:
            f.write(video_bytes)

        # Run inference with CP
        output_dir = os.path.join(tmpdir, "output")
        parallel_inference(
            input_video_path=input_path,
            output_dir=output_dir,
            prompts=prompts,
            batch_size=min(len(prompts), 4),
            workspace_dir="/root/app",
        )

        # Only rank 0 returns results
        if rank == 0:
            outputs = []
            for i in range(len(prompts)):
                video_path = os.path.join(output_dir, f"video_{i}", "output.mp4")
                if os.path.exists(video_path):
                    with open(video_path, "rb") as f:
                        outputs.append(f.read())
            return outputs

        return None
```

## How Context Parallelism Works

### Architecture

```
GPU 0                          GPU 1
├── Model weights (shard 0)    ├── Model weights (shard 1)
├── Frames 0-60               ├── Frames 61-121
├── Attention layers          ├── Attention layers
└── Compute on shard 0        └── Compute on shard 1
        │                             │
        └─────── all_gather ──────────┘
                     │
              Full video output
```

### Initialization Flow

1. **torchrun** launches N processes (one per GPU)
2. Each process calls `parallel_inference()`
3. `parallel_inference()` calls `demo(args, control_inputs)`
4. `demo()` initializes distributed:
   ```python
   from megatron.core import parallel_state
   distributed.init()
   parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
   ```
5. Model loading with CP enabled:
   ```python
   process_group = parallel_state.get_context_parallel_group()
   pipeline = DiffusionControl2WorldGenerationPipeline(
       process_group=process_group,  # Enables CP
   )
   ```
6. Models automatically shard via `enable_context_parallel()`

### No Offloading Needed

**Why we don't use offloading with CP:**

```python
# BAD: Offloading with Context Parallelism
offload_text_encoder_model=True  # ❌ Defeats parallelism
# - GPU 0 computes → offloads to CPU → waits
# - GPU 1 computes → offloads to CPU → waits
# - Next iteration: reload from CPU → slow!

# GOOD: Keep everything on GPU
offload_text_encoder_model=False  # ✅ Parallel computation
# - GPU 0 & GPU 1 compute simultaneously
# - No CPU transfers
# - Maximum throughput
```

## Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_video_path` | Required | Path to input video |
| `output_dir` | Required | Output directory |
| `prompts` | `None` | List of prompts for batch processing |
| `batch_size` | `1` | Number of videos to process in parallel |
| `num_gpus` | Auto-detect | Number of GPUs (from WORLD_SIZE) |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guidance` | `6.0` | Classifier-free guidance scale |
| `num_steps` | `35` | Number of diffusion steps |
| `fps` | `30` | Output video FPS |
| `seed` | `1` | Random seed |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_dir` | `/root/app/checkpoints` | Model checkpoints |
| `controlnet_specs_path` | `assets/augment.json` | Controlnet config |
| `negative_prompt` | Default | Negative guidance prompt |
| `workspace_dir` | `/root/app` | Cosmos workspace |

## Batch Processing

Process multiple prompts efficiently:

```python
prompts = [
    "robot picks up red cube",
    "robot picks up blue sphere",
    "robot stacks blocks",
    "robot sorts objects",
]

parallel_inference(
    input_video_path="input.mp4",
    output_dir="./outputs",
    prompts=prompts,
    batch_size=2,  # Process 2 at a time
    num_gpus=2,
)

# Execution:
# - Iteration 1: prompts[0:2] with CP across 2 GPUs
# - Iteration 2: prompts[2:4] with CP across 2 GPUs
#
# For each iteration:
# - GPU 0 processes frames 0-60 of both videos
# - GPU 1 processes frames 61-121 of both videos
# - Results gathered and saved
```

## Performance Expectations

### Single Video (121 frames)

- **Time**: ~30-60 seconds with 2x H100 GPUs
- **Memory**: ~60-70GB per GPU (120-140GB total)
- **Throughput**: CP parallelizes temporal dimension

### Batch of 4 Videos (batch_size=2)

- **Time**: ~60-120 seconds total (2 batches)
- **Amortized**: ~15-30 seconds per video
- **Memory**: Same (~60-70GB per GPU)

## Testing

### Local Test

```bash
# Create test script
cat > test.py << 'EOF'
from cosmos_inference import parallel_inference
parallel_inference(
    input_video_path="/root/app/input/test.mp4",
    output_dir="/root/app/output/test",
    prompts=["test prompt"],
)
EOF

# Run with torchrun
torchrun --nproc_per_node=2 test.py
```

### Validate Memory Usage

1. Launch inference with torchrun
2. Monitor GPU memory in separate terminal:
   ```bash
   watch -n 1 nvidia-smi
   ```
3. Expected: ~60-70GB per GPU (not 120-140GB)

### Output Structure

```
output_dir/
├── video_0/
│   ├── output.mp4
│   └── prompt.txt
├── video_1/
│   ├── output.mp4
│   └── prompt.txt
└── ...
```

## Troubleshooting

### "Must be called from within torchrun"

**Problem**: Called `parallel_inference()` directly without torchrun

**Solution**:
```bash
# Wrong
python script.py

# Right
torchrun --nproc_per_node=2 script.py
```

### Memory still doubled

**Problem**: Using `run_cosmos_inference()` instead of `parallel_inference()`

**Solution**: Switch to `parallel_inference()` and launch with torchrun

### Modal Integration Issues

**Problem**: Modal not launching with torchrun

**Solution**: Modal automatically handles torchrun when `gpu.count > 1`. Just call `parallel_inference()` directly in your function.

## Migration from Legacy

### Before (Legacy - uses 2x memory)

```python
from cosmos_inference import run_cosmos_inference

run_cosmos_inference(
    input_video_path="input.mp4",
    output_dir="./output",
    num_gpus=2,
    offload_text_encoder=True,  # Doesn't help
)
```

### After (New - memory efficient)

```bash
# Launch with torchrun
torchrun --nproc_per_node=2 python -c "
from cosmos_inference import parallel_inference
parallel_inference(
    input_video_path='input.mp4',
    output_dir='./output',
    # num_gpus auto-detected
    # No offloading needed
)
"
```

## Technical Details

### Context Parallelism Implementation

- **File**: `cosmos_transfer1/diffusion/module/parallel.py`
- **Split function**: `split_inputs_cp()` - divides sequence across GPUs
- **Gather function**: `cat_outputs_cp()` - concatenates results
- **Broadcast**: `broadcast()` - syncs data across GPUs

### Model Sharding Points

From `world_generation_pipeline.py:363-369`:

```python
if process_group is not None:
    # Enable CP in base model
    self.model.model.net.enable_context_parallel(process_group)
    # Enable CP in base transformer
    self.model.model.base_model.net.enable_context_parallel(process_group)
    # Enable CP in hint encoders (if using multiple controlnets)
    if hasattr(self.model.model, "hint_encoders"):
        self.model.model.hint_encoders.net.enable_context_parallel(process_group)
```

### Distributed Initialization

From `transfer.py:206-215`:

```python
if cfg.num_gpus > 1:
    from megatron.core import parallel_state
    from cosmos_transfer1.utils import distributed

    distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=cfg.num_gpus)
    process_group = parallel_state.get_context_parallel_group()
    device_rank = distributed.get_rank(process_group)
```

## References

- [Cosmos Transfer1 Documentation](https://github.com/NVIDIA/Cosmos)
- [Megatron-Core Parallelism](https://github.com/NVIDIA/Megatron-LM)
- [Multi-GPU Checkpoint Strategies](./longform_cosmos_transfer1/docs/multi-gpu-checkpoint-strategies.md)
