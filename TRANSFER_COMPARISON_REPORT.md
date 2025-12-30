# Cosmos Transfer1 Comparison Report: longform vs cosmos-transfer1

**Purpose**: Detailed analysis of differences between `longform_cosmos_transfer1` and `cosmos-transfer1` implementations to inform Transfer2.5 development.

---

## Executive Summary

The **longform** version is a streamlined, inference-optimized fork that removes training/research artifacts and adds pipeline parallelism scaffolding. Key improvements center around:

1. **Disabled guardrails** for faster iteration
2. **Removed benchmarking overhead**
3. **Stripped knowledge distillation (KD) artifacts**
4. **Simplified preprocessor** (fewer dependencies)
5. **Leaner inference_utils** (~300 fewer lines)
6. **Pipeline parallelism architecture** (scaffolded, not fully implemented)

---

## 1. Import & Dependency Differences

### 1.1 transfer.py Imports

| Original (cosmos-transfer1) | Longform | Impact |
|----------------------------|----------|--------|
| `import time` | Removed | No timing overhead |
| `import pickle` | Removed | No serialization overhead |
| `import cosmos_transfer1.utils.fix_vllm_registration` | Removed | Cleaner module loading |
| `from cosmos_transfer1.utils.easy_io import easy_io` | Removed | No disk I/O utilities |

**Insight for Transfer2.5**: The vLLM registration fix suggests upstream transformers/vLLM version conflicts. Consider pinning exact versions or resolving at the package level.

### 1.2 world_generation_pipeline.py Imports

| Original | Longform | Impact |
|----------|----------|--------|
| `from collections import defaultdict` (duplicated) | Single import | Cleaner code |
| `get_ctrl_batch` function import | Removed | Unused function eliminated |
| `get_video_batch` function import | Removed | Unused function eliminated |
| Model config name for distilled: `"CTRL_7Bv1pt3_lvg_fsdp_distilled_121frames..."` | `"dev_v2w_ctrl_7bv1pt3_VisControlCanny_video_only_dmd2_fsdp"` | Different checkpoint config |

**Key Observation**: Longform uses a development distilled config (`dev_v2w_ctrl_7bv1pt3_VisControlCanny_video_only_dmd2_fsdp`) vs production config in original.

---

## 2. Guardrail Handling (Critical Difference)

### Original cosmos-transfer1 (world_generation_pipeline.py lines 730-735):
```python
for i, single_prompt in enumerate(prompts):
    is_safe = self._run_guardrail_on_prompt_with_offload(single_prompt)
    if is_safe:
        safe_indices.append(i)
    else:
        log.critical(f"Input text prompt {i+1} is not safe")
```

### Longform version (world_generation_pipeline.py lines 714-722):
```python
for i, single_prompt in enumerate(prompts):
    safe_indices.append(i)
    # oops, im being useless and misclassifying safe prompts as unsafe
    # is_safe = self._run_guardrail_on_prompt_with_offload(single_prompt)
    # if is_safe:
    #     safe_indices.append(i)
    # else:
    #     log.critical(f"Input text prompt {i+1} is not safe")
```

**Same pattern for video guardrails** (lines 771-780):
```python
for i, video in enumerate(videos):
    all_videos.append(video)
    all_final_prompts.append(safe_prompts[i])
    # oops im blocking progress
    # safe_video = self._run_guardrail_on_video_with_offload(video)
```

**Impact for Transfer2.5**:
- Guardrails add significant latency (model loading, inference, offloading)
- Consider: Optional guardrail flag, async guardrails, or guardrail caching
- The developer comment reveals production frustration with false positives

---

## 3. Pipeline Constructor Differences

### 3.1 DiffusionControl2WorldGenerationPipeline.__init__

| Parameter | Original | Longform |
|-----------|----------|----------|
| `disable_guardrail` | Present (bool) | **Absent** |
| `save_input_noise` | Present (bool) | **Absent** |

**Original constructor** (lines 148-149):
```python
disable_guardrail: bool = False,
save_input_noise: bool = False,
```

**Longform completely removes** these parameters, hardcoding behavior.

### 3.2 Base Class Initialization

**Original** passes `disable_guardrail` to parent:
```python
super().__init__(
    ...
    disable_guardrail=disable_guardrail,
)
```

**Longform** doesn't pass it (relies on commented-out guardrail code).

---

## 4. Knowledge Distillation (KD) Artifacts Removal

### 4.1 In transfer.py

**Original** (lines 163-167, 384-388, 419-421):
```python
parser.add_argument(
    "--save_input_noise",
    action="store_true",
    help="Save input noise for ODE pairs generation used for Knowledge Distillation",
)

# Later in demo():
if cfg.save_input_noise:
    videos, final_prompts, input_noise = batch_outputs
else:
    videos, final_prompts = batch_outputs
    input_noise = [None] * len(videos)

# And saving:
if cfg.save_input_noise:
    easy_io.dump(noise, noise_save_path)
```

**Longform**: All removed. No `--save_input_noise` argument, no noise handling, no disk writes.

### 4.2 In world_generation_pipeline.py

**Original** `generate_world_from_control` call includes:
```python
save_input_noise=save_input_noise,
```

**Original** `_run_model` unpacks:
```python
if self.save_input_noise:
    latents, noise = samples
    if i_clip == 0:
        input_noise = noise.to(torch.float32).cpu().numpy()
else:
    latents = samples
```

**Longform**: Clean single return path:
```python
latents = generate_world_from_control(...)
```

**Impact for Transfer2.5**: If KD is needed, implement as optional plugin rather than inline code.

---

## 5. Benchmarking Removal

### Original (transfer.py lines 357-382):
```python
num_repeats = 4 if cfg.benchmark else 1
time_sum = 0
for i in range(num_repeats):
    if cfg.benchmark and i > 0:
        torch.cuda.synchronize()
        start_time = time.time()
    batch_outputs = pipeline.generate(...)
    if cfg.benchmark and i > 0:
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        time_sum += elapsed
        log.info(f"[iter {i} / {num_repeats - 1}] Generation time: {elapsed:.1f} seconds.")

if cfg.benchmark:
    time_avg = time_sum / (num_repeats - 1)
    log.critical(f"The benchmarked generation time for Cosmos-Transfer1 is {time_avg:.1f} seconds.")
```

**Longform**: Single pass, no timing:
```python
batch_outputs = pipeline.generate(...)
```

**Performance Impact**:
- Removes 3x redundant generation passes in benchmark mode
- Eliminates `torch.cuda.synchronize()` overhead
- No timing instrumentation

---

## 6. Preprocessor Differences (Critical for Video Quality)

### 6.1 Missing Preprocessor Models in Longform

**Original** (preprocessors.py) has:
```python
from cosmos_transfer1.auxiliary.edge_control.edge_control import EdgeControlModel
from cosmos_transfer1.auxiliary.vis_control.vis_control import VisControlModel

self.vis_model = None
self.edge_model = None

def vis(self, in_video, out_video, blur_strength="medium"):
    if self.vis_model is None:
        self.vis_model = VisControlModel(blur_strength=blur_strength)
    self.vis_model(in_video, out_video)

def edge(self, in_video, out_video, canny_threshold="medium"):
    if self.edge_model is None:
        self.edge_model = EdgeControlModel(canny_threshold=canny_threshold)
    self.edge_model(in_video, out_video)
```

**Longform**: These are **completely absent**. No `vis` or `edge` preprocessor methods.

### 6.2 Control Modality Support

| Modality | Original | Longform |
|----------|----------|----------|
| depth | Yes | Yes |
| seg | Yes | Yes |
| keypoint | Yes | Yes |
| vis | Yes | **No** |
| edge | Yes | **No** |

**Impact**: Longform cannot auto-generate vis/edge control inputs from raw video - must be pre-computed.

### 6.3 Parameter Passing

**Original** preprocessor call (transfer.py lines 324-332):
```python
preprocessors(
    current_video_path,
    current_prompt,
    current_control_inputs,
    video_save_subfolder,
    cfg.regional_prompts if hasattr(cfg, "regional_prompts") else None,
    blur_strength=cfg.blur_strength,
    canny_threshold=cfg.canny_threshold,
)
```

**Longform** preprocessor call (transfer.py lines 306-312):
```python
preprocessors(
    current_video_path,
    current_prompt,
    current_control_inputs,
    video_save_subfolder,
    cfg.regional_prompts if hasattr(cfg, "regional_prompts") else None,
)
```

**Missing**: `blur_strength` and `canny_threshold` parameters not passed.

### 6.4 Rank-Safe Filenames (Original Only)

**Original** generates rank-specific filenames to prevent file corruption in multi-GPU:
```python
out_tensor = os.path.join(
    output_folder, f"{hint_key}_control_weight_{int(os.environ.get('LOCAL_RANK', 0))}.pt"
)
out_video = os.path.join(
    output_folder, f"{hint_key}_control_weight_{int(os.environ.get('LOCAL_RANK', 0))}.mp4"
)
```

**Longform** uses simple names (potential race condition in multi-GPU):
```python
out_tensor = os.path.join(output_folder, f"{hint_key}_control_weight.pt")
out_video = os.path.join(output_folder, f"{hint_key}_control_weight.mp4")
```

---

## 7. Pipeline Parallelism (Longform Exclusive)

### 7.1 New Function: demo_pipeline_parallel

**Location**: longform transfer.py lines 385-592

**Architecture Vision** (documented in comments):
```
GPU 0: Preprocessors + Guardrails
GPU 1: Text encoder (T5)
GPU 2: Tokenizer (VAE encoder/decoder)
GPU 3: Diffusion transformer (DiT)
```

**Current Status**: Scaffolded but NOT implemented:
```python
# TODO: Implement actual pipeline parallelism with manual device placement
# For now, fall back to context parallelism (original behavior)
log.warning("Pipeline parallelism not fully implemented yet - using context parallelism")
```

### 7.2 Entry Point Routing

**Longform** main block:
```python
if __name__ == "__main__":
    args, control_inputs = parse_arguments()
    # Use pipeline parallel version for multi-GPU
    if args.num_gpus > 1:
        demo_pipeline_parallel(args, control_inputs)
    else:
        demo(args, control_inputs)
```

**Original** always uses single `demo()`:
```python
if __name__ == "__main__":
    args, control_inputs = parse_arguments()
    demo(args, control_inputs)
```

**Impact for Transfer2.5**:
- Pipeline parallelism would enable larger batch sizes by distributing memory
- Current implementation just adds logging overhead without actual parallelism
- Foundation exists for true pipeline parallelism implementation

---

## 8. inference_utils.py Differences (~300 Lines)

### 8.1 Removed from Longform

| Feature | Lines in Original | Purpose |
|---------|-------------------|---------|
| `switch_config_for_inference` | ~35 lines | Context manager for extend model config |
| `visualize_latent_tensor_bcthw` | ~25 lines | Debug visualization |
| `visualize_tensor_bcthw` | ~15 lines | Debug visualization |
| `generate_video_from_batch_with_loop` | ~200 lines | Advanced loop generation for ExtendDiffusionModel |

### 8.2 ExtendDiffusionModel Support

**Original** imports:
```python
from cosmos_transfer1.diffusion.training.models.extend_model import ExtendDiffusionModel
```

**Longform** imports (different model):
```python
from cosmos_transfer1.diffusion.model.model_v2w_multiview import DiffusionV2WMultiviewModel
```

### 8.3 Tokenizer Access Pattern

**Original** (lines 946-965) handles both causal and non-causal tokenizers:
```python
# For causal tokenizer
num_frames_condition = (
    num_of_latent_overlap // model.tokenizer.latent_chunk_duration * model.tokenizer.pixel_chunk_duration
)
```

**Longform** (lines 950-965) uses video_vae sub-attribute:
```python
if getattr(model.tokenizer.video_vae, "is_casual", True):
    # For casual model
    num_frames_condition = (
        num_of_latent_overlap
        // model.tokenizer.video_vae.latent_chunk_duration
        * model.tokenizer.video_vae.pixel_chunk_duration
    )
```

**Note**: Original has typo `is_casual` (should be `is_causal`) - likely carried over.

### 8.4 Distilled Checkpoint Handling

**Original** (line 1251-1253):
```python
raise ValueError(
    f"No default distilled checkpoint for {hint_key}. Users must specify ckpt_path in config."
)
```

**Longform** (line 1261):
```python
log.info(f"No default distilled checkpoint for {hint_key}. Using full checkpoint")
```

**Impact**: Longform is more permissive, falling back gracefully instead of raising.

---

## 9. Distilled Pipeline Differences

### 9.1 DistilledControl2WorldGenerationPipeline._load_network

**Original** (lines 1287-1312):
```python
assert len(self.control_inputs) == 1, "Distilled model only supports single control input"
for _, config in self.control_inputs.items():
    checkpoint_path = config["ckpt_path"]
    break
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

**Longform** (lines 1263-1286):
```python
checkpoint_path = f"{self.checkpoint_dir}/{self.checkpoint_name}"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
# ... also loads base_model differently
self.model.model.base_model.load_state_dict(base_state_dict, strict=False)  # Extra line
```

**Key Difference**: Original loads from `control_inputs[*]["ckpt_path"]`, Longform loads from `checkpoint_name` directly.

### 9.2 Distilled Model - x_sigma_max Handling

**Original** (lines 1369-1375):
```python
if input_video is not None:
    input_frames = input_video[:, :, start_frame:end_frame].cuda()
    x0 = self.model.encode(input_frames).contiguous()
    x_sigma_max = self.model.get_x_from_clean(x0, self.sigma_max, seed=(self.seed + i_clip))
else:
    assert False
    x_sigma_max = None
```

**Longform** doesn't have this section - distilled pipeline is simpler.

### 9.3 generate_world_from_control Call

**Original** distilled (line 1430):
```python
is_negative_prompt=False,  # Unused for distilled models
```

**Longform** non-distilled (line 1415):
```python
is_negative_prompt=True,
```

---

## 10. Long Video Generation Loop

### 10.1 Commented-Out Latent Reuse

**Longform** has commented experimental code (lines 602-610):
```python
# Simple periodic reset approach - clean and effective
# if prev_latents is not None:
#     # Solution #2: Use stored latents to avoid VAE encode-decode roundtrip
#     condition_latent = prev_latents
# else:
    # Fallback to original method
prev_frames = split_video_into_patches(...)
input_frames = prev_frames.bfloat16().cuda() / 255.0 * 2 - 1
condition_latent = self.model.encode(input_frames).contiguous()
```

And (lines 643-644):
```python
# Store latents cleanly without any modification
# prev_latents = latents[:, :, -self.num_input_frames :].contiguous()
```

**Original**: No such experimental code.

**Insight**: Developer explored skipping VAE encode-decode roundtrip between clips (would save significant compute) but reverted.

---

## 11. Output Filename Handling

### Single Video Mode

**Original** (lines 355-356):
```python
video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")
```

**Longform** (same lines, but without noise):
```python
video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")
# No noise_save_path
```

### Batch Mode

**Original** adds noise path:
```python
noise_save_path = os.path.join(video_save_subfolder, "noise.pickle")
```

---

## 12. Recommendations for Transfer2.5

### 12.1 Architecture Improvements

1. **Implement actual pipeline parallelism** - The longform scaffolding provides the entry point. Use `torch.distributed` with custom device placement:
   ```python
   # Suggested implementation
   self.t5_device = f"cuda:{rank % world_size}"
   self.vae_device = f"cuda:{(rank + 1) % world_size}"
   self.dit_device = f"cuda:{(rank + 2) % world_size}"
   ```

2. **Optional guardrails** - Add `--disable-guardrails` flag for development/testing while keeping production safety.

3. **Async guardrails** - Run safety checks in parallel with next batch preparation.

### 12.2 Performance Optimizations

1. **Latent reuse between clips** - Implement the commented-out `prev_latents` approach to skip VAE roundtrip:
   ```python
   if prev_latents is not None:
       condition_latent = prev_latents  # Skip encode-decode
   ```

2. **Remove benchmark mode from production** - Keep in separate profiling script.

3. **Lazy preprocessor initialization** - Only load depth/seg models when needed.

### 12.3 Code Quality

1. **Fix `is_casual` typo** - Should be `is_causal`

2. **Rank-safe filenames** - Use LOCAL_RANK in all multi-GPU file outputs:
   ```python
   suffix = f"_r{os.environ.get('LOCAL_RANK', 0)}"
   ```

3. **Remove KD artifacts from inference** - Implement as optional training plugin.

### 12.4 Modular Improvements

1. **Separate preprocessor package** - vis/edge control models should be optional dependencies.

2. **Config-driven behavior** - Replace hardcoded guardrail disabling with config flag.

3. **Unified checkpoint loading** - Standardize between `ckpt_path` in control_inputs vs `checkpoint_name`.

---

## Appendix: Quick Reference

### File Line Counts

| File | Original | Longform | Delta |
|------|----------|----------|-------|
| transfer.py | 434 | 601 | +167 (pipeline parallel) |
| world_generation_pipeline.py | 1456 | 1442 | -14 |
| preprocessors.py | ~200 | ~130 | ~-70 |
| inference_utils.py | 1582 | 1285 | -297 |

### Key Functional Differences

| Feature | Original | Longform |
|---------|----------|----------|
| Guardrails | Active | Disabled |
| Benchmarking | Supported | Removed |
| KD noise saving | Supported | Removed |
| vis/edge preprocessors | Yes | No |
| Pipeline parallelism | No | Scaffolded |
| Rank-safe filenames | Yes | No |
| ExtendDiffusionModel support | Yes | No |
| Debug visualizations | Yes | No |

---

*Report generated for Transfer2.5 development planning*
