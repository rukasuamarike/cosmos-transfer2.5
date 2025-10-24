# Cosmos-Transfer2.5 Inference Pipeline Architecture

## Overview

This document provides a deep dive into how the Cosmos-Transfer2.5 inference pipeline processes requests from JSON configuration files to final video outputs.

## Pipeline Flow Diagram

```
User Input (Shell Script)
    │
    ├─> Parse arguments (video, controls, prompt)
    │
    ├─> Generate JSON config file(s)
    │   └─> {
    │         "name": "sample_01",
    │         "prompt": "a robot picks up a towel",
    │         "video_path": "/path/to/input.mp4",
    │         "guidance": 3,
    │         "seed": 2025,
    │         "depth": {"control_weight": 0.8},
    │         "edge": {"control_weight": 0.5}
    │       }
    │
    ├─> Call Python CLI: examples/inference.py
    │   ├─> tyro.cli(Args) parses arguments:
    │   │   ├─> -i/--input-files: JSON config paths
    │   │   └─> setup.output_dir (-o/--output-dir): REQUIRED output directory
    │   │
    │   ├─> InferenceArguments.from_files()
    │   │   ├─> Load JSON configs
    │   │   ├─> Validate with Pydantic models
    │   │   └─> Extract batch_hint_keys (controls used)
    │   │
    │   ├─> Control2WorldInference(setup, batch_hint_keys)
    │   │   ├─> Load model checkpoints (per control type)
    │   │   ├─> Initialize ControlVideo2WorldInference pipeline
    │   │   └─> Setup guardrails (if enabled)
    │   │
    │   └─> inference.generate(samples, output_dir)
    │       └─> For each sample:
    │           ├─> Load input video
    │           ├─> Compute/load control inputs (depth, edge, seg, vis)
    │           ├─> Get text embeddings (T5 or Reason1)
    │           ├─> Run guardrail checks (optional)
    │           ├─> Generate video via diffusion model
    │           │   └─> ControlVideo2WorldInference.generate_img2world()
    │           │       ├─> Process input video & controls
    │           │       ├─> Chunk-wise generation for long videos
    │           │       ├─> Diffusion sampling with guidance
    │           │       └─> Decode latents to video
    │           ├─> Save output video & control visualizations
    │           └─> Save metadata (prompt, config)
    │
    └─> Output:
        ├─> {output_dir}/{sample_name}/output.mp4
        ├─> {output_dir}/{sample_name}_control_depth.mp4
        ├─> {output_dir}/{sample_name}_control_edge.mp4
        ├─> {output_dir}/{sample_name}.txt (prompt)
        └─> {output_dir}/{sample_name}.json (full config)
```

## Key Components

### 1. Shell Script Layer (`run_inferencev2.sh`)

**Purpose:** Simplify user interaction by providing a friendly CLI interface

**Responsibilities:**
- Parse user-friendly arguments
- Generate JSON config files
- Invoke Python CLI with proper arguments
- Handle multi-GPU detection and setup

**Key Insight:** This layer is optional - users can directly call `examples/inference.py` with JSON configs

### 2. Python CLI Entry Point (`examples/inference.py`)

**Purpose:** Parse structured arguments and orchestrate the inference workflow

**Key Classes:**
- `Args`: Top-level argument container
  - `input_files`: List of JSON config paths
  - `setup`: `SetupArguments` (model config, output dir, etc.)
  - `overrides`: Runtime parameter overrides

**Flow:**
```python
args = tyro.cli(Args)  # Parse CLI arguments with tyro
samples, hint_keys = InferenceArguments.from_files(args.input_files, args.overrides)
inference = Control2WorldInference(args.setup, hint_keys)
inference.generate(samples, args.setup.output_dir)
```

### 3. Configuration Layer (`cosmos_transfer2/config.py`)

**Purpose:** Define and validate all inference parameters using Pydantic

**Key Models:**

#### `SetupArguments`
```python
class SetupArguments:
    output_dir: Path  # REQUIRED - where to save outputs
    model: Literal["edge", "depth", "seg", "vis"]  # Which model to load
    checkpoint_path: str | None  # Override default checkpoint
    context_parallel_size: int  # Number of GPUs for model parallelism
    enable_guardrails: bool  # Safety filters
```

#### `InferenceArguments`
```python
class InferenceArguments:
    # Required
    name: str  # Sample identifier
    prompt: str  # Text description
    video_path: Path  # Input video

    # Optional controls (at least one required)
    depth: DepthConfig | None
    edge: EdgeConfig | None
    seg: SegConfig | None
    vis: BlurConfig | None

    # Generation parameters
    guidance: int = 3  # CFG scale
    seed: int = 2025
    resolution: str = "720"
    num_conditional_frames: int = 1
    num_video_frames_per_chunk: int = 93
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
```

#### Control Configs
```python
class DepthConfig:
    control_weight: float = 1.0  # 0.0 to 1.0
    control_path: Path | None = None  # Pre-computed depth map (optional)

class EdgeConfig:
    control_weight: float = 1.0
    control_path: Path | None = None  # Pre-computed edge map (optional)
    preset_edge_threshold: Literal["very_low", "low", "medium", "high", "very_high"] = "medium"
```

**Why Pydantic?**
- Automatic validation of all parameters
- Type safety and IDE autocomplete
- Clear error messages for invalid configs
- Easy JSON serialization/deserialization

### 4. Inference Orchestrator (`cosmos_transfer2/inference.py`)

**Purpose:** Manage model loading, sample generation, and guardrails

**Class:** `Control2WorldInference`

**Initialization:**
```python
def __init__(self, args: SetupArguments, batch_hint_keys: list[str]):
    # Load appropriate model checkpoints based on controls
    if len(batch_hint_keys) == 1:
        checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=batch_hint_keys[0])]
    else:
        # Multi-control: load multiple checkpoints
        checkpoint_list = [MODEL_CHECKPOINTS[ModelKey(variant=k)] for k in batch_hint_keys]

    # Initialize the diffusion model pipeline
    self.inference_pipeline = ControlVideo2WorldInference(
        registered_exp_name=EXPERIMENTS[experiment].registered_exp_name,
        checkpoint_paths=checkpoint_list,
        ...
    )

    # Setup guardrails (text & video safety filters)
    if args.enable_guardrails:
        self.text_guardrail_runner = create_text_guardrail_runner()
        self.video_guardrail_runner = create_video_guardrail_runner()
```

**Generation:**
```python
def generate(self, samples: list[InferenceArguments], output_dir: Path):
    for sample in samples:
        # Run text guardrail on prompt
        guardrail_presets.run_text_guardrail(sample.prompt, ...)

        # Generate video
        output_video, control_videos, fps, hw = self.inference_pipeline.generate_img2world(
            video_path=sample.video_path,
            prompt=sample.prompt,
            hint_key=sample.hint_keys,  # e.g., ["depth", "edge"]
            control_weight=sample.control_weight_dict,  # e.g., {"depth": "0.8", "edge": "0.5"}
            ...
        )

        # Run video guardrail on output
        guardrail_presets.run_video_guardrail(output_video, ...)

        # Save outputs
        save_video(output_video, f"{output_dir}/{sample.name}.mp4")
        save_video(control_videos['depth'], f"{output_dir}/{sample.name}_control_depth.mp4")
```

### 5. Diffusion Pipeline (`cosmos_transfer2/_src/transfer2/inference/inference_pipeline.py`)

**Purpose:** Core video generation using diffusion models with control conditioning

**Class:** `ControlVideo2WorldInference`

**Key Methods:**

#### `__init__`: Load Model
```python
def __init__(self, registered_exp_name, checkpoint_paths, ...):
    # Load diffusion model from checkpoint
    model, config = load_model_from_checkpoint(
        experiment_name=registered_exp_name,
        s3_checkpoint_dir=checkpoint_paths,
        config_file="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py",
        load_ema_to_reg=True,  # Use EMA weights for better quality
    )

    # For multi-control: load additional branches
    if isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1:
        model.load_multi_branch_checkpoints(checkpoint_paths)

    self.model = model
```

#### `generate_img2world`: Main Generation Loop
```python
@torch.no_grad()
def generate_img2world(
    self,
    prompt: str,
    video_path: str,
    guidance: int = 7,
    seed: int = 1,
    resolution: str = "720",
    control_weight: str = "1.0",
    hint_key: list[str] = ["edge"],
    num_video_frames_per_chunk: int = 93,
    num_conditional_frames: int = 1,
    ...
):
    # 1. Load and process input video
    input_frames, fps, aspect_ratio, original_hw = read_and_process_video(
        video_path, resolution=resolution
    )

    # 2. Get text embeddings
    text_embeddings = get_t5_from_prompt(prompt, text_encoder_class="T5")

    # 3. Load or compute control inputs
    control_input_dict = read_and_process_control_input(
        video_path=video_path,
        hint_key=hint_key,  # ["depth", "edge"]
        resolution=resolution,
    )
    # If control paths not provided, compute on-the-fly via augmentors

    # 4. Chunk-wise generation for long videos
    num_chunks = calculate_num_chunks(input_frames, num_video_frames_per_chunk)

    all_chunks = []
    prev_output = torch.zeros(...)  # Initialize with zeros

    for chunk_id in range(num_chunks):
        # Prepare current chunk of input frames
        cur_input_frames = input_frames[:, chunk_start:chunk_end]

        # Build data batch
        data_batch = {
            "video": prev_output,  # Previous generated frames
            "t5_text_embeddings": text_embeddings,
            "control_input_depth": control_input_dict["depth"],
            "control_input_edge": control_input_dict["edge"],
            "control_weight": [0.8, 0.5],  # Parsed from control_weight string
            "num_conditional_frames": num_conditional_frames if chunk_id > 0 else 0,
        }

        # Generate latent samples via diffusion
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            seed=seed,
            num_steps=35,  # Diffusion steps
        )

        # Decode latents to RGB video
        video = self.model.decode(sample)  # Shape: (B, C, T, H, W)

        all_chunks.append(video)

        # Use last frames as conditioning for next chunk
        prev_output = video[:, :, -num_conditional_frames:]

    # 5. Concatenate all chunks
    full_video = torch.cat(all_chunks, dim=2)

    return full_video, control_video_dict, fps, original_hw
```

#### Control Processing Pipeline

**On-the-fly computation** (if `control_path` not provided):

```python
# Depth: Use Depth-Anything model
depth_map = depth_anything_model(input_frames)

# Edge: Canny edge detection
edge_map = cv2.Canny(input_frames, threshold1, threshold2)

# Segmentation: Use SAM2 with text prompt
seg_mask = sam2_segment(input_frames, seg_control_prompt)

# Blur/Vis: Gaussian blur for visibility mask
blur_map = cv2.GaussianBlur(input_frames, kernel_size)
```

**Pre-computed paths** (if `control_path` provided):
```python
control_video = read_video(control_path)  # Load pre-computed control signal
```

## Data Flow Example

### Input JSON Config
```json
{
    "name": "robot_demo",
    "prompt": "a humanoid robot picking up a blue towel in a modern factory",
    "video_path": "/workspace/assets/robot_input.mp4",
    "guidance": 3,
    "seed": 2025,
    "resolution": "720",
    "depth": {
        "control_weight": 0.8
    },
    "edge": {
        "control_weight": 0.5,
        "preset_edge_threshold": "medium"
    }
}
```

### CLI Invocation
```bash
python -m examples.inference \
    -i robot_demo_config.json \
    --output-dir outputs/robot_test
```

### Internal Processing

1. **tyro parses CLI args:**
```python
Args(
    input_files=[Path("robot_demo_config.json")],
    setup=SetupArguments(
        output_dir=Path("outputs/robot_test"),
        model="edge",  # Auto-detected from controls
        checkpoint_path="s3://bucket/edge_model.pt",
        ...
    ),
    overrides=InferenceOverrides()
)
```

2. **Load and validate JSON:**
```python
InferenceArguments(
    name="robot_demo",
    prompt="a humanoid robot picking up...",
    video_path=Path("/workspace/assets/robot_input.mp4"),
    hint_keys=["depth", "edge"],  # Computed property
    control_weight_dict={"depth": "0.8", "edge": "0.5"},
    ...
)
```

3. **Model inference:**
```python
# Load input video: (3, 100, 720, 1280)
# Compute depth maps: (1, 100, 720, 1280)
# Compute edge maps: (1, 100, 720, 1280)
# Encode to latents: (B, C_latent, T, H_latent, W_latent)
# Diffusion sampling with guidance=3
# Decode to video: (1, 3, 100, 720, 1280)
```

4. **Save outputs:**
```
outputs/robot_test/
├── robot_demo/
│   └── output.mp4
├── robot_demo_control_depth.mp4
├── robot_demo_control_edge.mp4
├── robot_demo.txt
└── robot_demo.json
```

## Critical CLI Arguments

### The `--output-dir` Requirement

**Why it's required:**
```python
# From config.py line 180
class CommonSetupArguments:
    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory."""
```

This is a **required field** without a default value because:
1. Outputs need a clear destination
2. Prevents accidental overwrites
3. Enables organized experiment tracking
4. Required for logging initialization

**Fix for run_inference.sh:**
The old script passed `-o` but the CLI expects `--output-dir` (or `-o`). The real issue was that `-o` wasn't being passed at all in some cases.

### Model Selection

**Automatic:**
```python
# If using single control, auto-select that model
if len(batch_hint_keys) == 1:
    model = batch_hint_keys[0]  # "edge", "depth", etc.

# If multi-control, use multi-branch model
else:
    model = "multibranch_720p_t24_..."
```

**Manual override:**
```bash
python -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.model depth  # Force depth model
```

## Multi-GPU Support

### Context Parallelism

For very long videos or high resolution:

```bash
# Use all available GPUs
torchrun --nproc_per_node=8 \
    -m examples.inference \
    -i config.json \
    --output-dir outputs \
    --setup.context-parallel-size 8
```

**How it works:**
1. Distributes sequence (time) dimension across GPUs
2. Each GPU processes T/N frames
3. Communication via process groups (NCCL)
4. Enabled in model with `model.net.enable_context_parallel(process_group)`

### Data Parallelism

For batch processing multiple samples:

```bash
# Process 4 samples in parallel on 4 GPUs
torchrun --nproc_per_node=4 \
    -m examples.inference \
    -i config1.json config2.json config3.json config4.json \
    --output-dir outputs
```

## Common Issues and Solutions

### Issue 1: Missing `--output-dir`
```
Error: The following arguments are required: -o/--output-dir
```

**Solution:**
```bash
# Always provide output directory
python -m examples.inference -i config.json --output-dir my_outputs
```

### Issue 2: No controls specified
```
ValidationError: No controls provided, please provide at least one control key
```

**Solution:** Add at least one control to JSON:
```json
{
    "name": "sample",
    "prompt": "...",
    "video_path": "...",
    "edge": {"control_weight": 0.8}  // Add this
}
```

### Issue 3: Video path not found
```
FileNotFoundError: video_path '/path/to/video.mp4' does not exist
```

**Solution:** Use absolute paths in JSON configs or ensure paths are relative to CWD

## Performance Optimization

### Chunk Size
```python
num_video_frames_per_chunk: int = 93  # Default
# Larger = more VRAM, faster
# Smaller = less VRAM, slower
```

### Resolution
```python
resolution: str = "720"  # or "480"
# 480p: ~2x faster, uses ~50% VRAM
# 720p: Better quality
```

### Diffusion Steps
```python
num_steps: int = 35  # Default
# Fewer steps = faster but lower quality
# More steps = slower but better quality
```

### Control Weights
```python
# Higher weight = stronger control signal adherence
"depth": {"control_weight": 0.8}  # Strong
"edge": {"control_weight": 0.3}   # Weak
```

## Extending the Pipeline

### Adding a New Control Type

1. **Define config in `config.py`:**
```python
class MyControlConfig(ControlConfig):
    preset_my_param: Threshold = "medium"
    control_path: ResolvedFilePath | None = None

class InferenceArguments(CommonInferenceArguments):
    my_control: MyControlConfig | None = None

CONTROL_KEYS = ["edge", "vis", "depth", "seg", "my_control"]
```

2. **Add checkpoint in `config.py`:**
```python
MODEL_CHECKPOINTS = {
    ModelKey(variant="my_control"): get_checkpoint_by_uuid("uuid-here"),
}
```

3. **Implement augmentor in `datasets/augmentors/control_input.py`:**
```python
def get_augmentor_for_eval(data_dict, output_keys, ...):
    if "my_control" in output_keys:
        data_dict["control_input_my_control"] = compute_my_control(
            data_dict["input_video"]
        )
```

4. **Use in JSON:**
```json
{
    "my_control": {
        "control_weight": 0.7,
        "preset_my_param": "high"
    }
}
```

## Summary

The Cosmos-Transfer2.5 inference pipeline is a **layered architecture**:

1. **Shell wrapper** (`run_inferencev2.sh`) - User-friendly CLI
2. **Python CLI** (`examples/inference.py`) - Argument parsing with tyro
3. **Config layer** (`config.py`) - Pydantic validation
4. **Orchestrator** (`inference.py`) - Model loading & guardrails
5. **Diffusion engine** (`inference_pipeline.py`) - Video generation

**Key takeaway:** The `--output-dir` argument is **required** and must be explicitly provided either via the shell script (`-o`) or when calling the Python CLI directly (`--output-dir`).
