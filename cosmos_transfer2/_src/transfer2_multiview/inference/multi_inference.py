# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Inference script for constructing data_batch from videos, control videos (control), and captions, 
then running transfer2_multiview model.

Expected directory structure:
```
input_root/
├── videos/                                    # Input video folder
│   ├── camera_front_wide_120fov/      # or camera_front_wide_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── camera_cross_right_120fov/     # or camera_cross_right_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── camera_rear_right_70fov/       # or camera_rear_right_70fov/
│   ├── camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
│   ├── camera_rear_left_70fov/        # or camera_rear_left_70fov/
│   ├── camera_cross_left_120fov/      # or camera_cross_left_120fov/
│   └── camera_front_tele_30fov/       # or camera_front_tele_30fov/
│
├── control/                            # Control video folder (world scenario)
│   ├── camera_front_wide_120fov/      # or camera_front_wide_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── camera_cross_right_120fov/     # or camera_cross_right_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── camera_rear_right_70fov/       # or camera_rear_right_70fov/
│   ├── camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
│   ├── camera_rear_left_70fov/        # or camera_rear_left_70fov/
│   ├── camera_cross_left_120fov/      # or camera_cross_left_120fov/
│   └── camera_front_tele_30fov/       # or camera_front_tele_30fov/
│
└── captions/                                  # Caption folder (optional, uses default prompt if not present)
    ├── camera_front_wide_120fov/      # or camera_front_wide_120fov/
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── camera_cross_right_120fov/     # or camera_cross_right_120fov/
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── camera_rear_right_70fov/       # or camera_rear_right_70fov/
    ├── camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
    ├── camera_rear_left_70fov/        # or camera_rear_left_70fov/
    ├── camera_cross_left_120fov/      # or camera_cross_left_120fov/
    └── camera_front_tele_30fov/       # or camera_front_tele_30fov/

Notes:
- The videos/ folder is required (input videos)
- The control/ folder is required (control signal videos)
- The captions/ folder is optional; if not present, a preset default driving scene description is used
- Each camera's subfolder name supports two formats: "{camera_name}" or "{camera_name}"
- video_id must be consistent across all camera folders in all three directories
- All 7 camera views must have corresponding subfolders and files

Camera view to View Index mapping:
- camera_front_wide_120fov: 0
- camera_cross_right_120fov: 1
- camera_rear_right_70fov: 2
- camera_rear_tele_30fov: 3
- camera_rear_left_70fov: 4
- camera_cross_left_120fov: 5
- camera_front_tele_30fov: 6
```

Usage:
```bash
Create an S3 with jobs as JSON --> store for experiments
Inside a job, it just needs to have the prompts and the indexes for all the videos.

EXP=
ckpt_path=



PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.multi_inference \
    --experiment transfer2p5_2b_mv_7train7_res480p_fps10_t24_frombase2p5avfinetune_mads_only_allcaption_uniform_nofps_wm_condition_i2v_and_t2v \
    --ckpt_path s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv_7train7_res480p_fps10_t24_frombase2p5avfinetune_mads_only_allcaption_uniform_nofps_wm_condition_i2v_and_t2v-0/checkpoints/iter_000010000/ \
    --context_parallel_size 8 \
    --num_conditional_frames 1 \
    --guidance 3.0 \
    --input_root /root/app/job \
    --save_root results/transfer2_multiview_480p_i2v_grid_eachview/ \
    --max_samples 5 --target_height 480 --target_width 832 \
    --stack_mode grid_auto  \
    --save_each_view \
    model.config.base_load_from=null

# auto-regressive
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/yiflu/project_official_i4/condition_assets/multiview-inference-assets-1204/normal-200frame-10fps \
    --num_conditional_frames 1 \
    --guidance 5.0 \
    --save_root results/transfer2_multiview_480p_i2v_long_grid_eachview/ \
    --max_samples 50 --target_height 480 --target_width 832 \
    --use_autoregressive --target_frames 277 \
    --stack_mode grid_auto \
    --save_each_view \
    model.config.base_load_from=null
```
"""

import argparse
import os
import json
from pathlib import Path

import torch as th
import torchvision

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2_multiview.scripts.mv_visualize_helper import (
    arrange_video_visualization,
    save_each_view_separately,
)
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference

NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"


def calculate_autoregressive_frames(chunk_size: int, chunk_overlap: int, num_chunks: int) -> int:
    """Calculate total frames needed for autoregressive generation.

    Args:
        chunk_size: Frames per chunk (from model's state_t)
        chunk_overlap: Overlapping frames between chunks
        num_chunks: Number of chunks to generate

    Returns:
        Total frames needed across all chunks

    Example:
        chunk_size=93, overlap=1, num_chunks=20
        → 93 + (93-1)*19 = 1841 frames
    """
    return chunk_size + (chunk_size - chunk_overlap) * (num_chunks - 1)


# # Camera name to view index mapping
# CAMERA_TO_VIEW_INDEX = {
#     "camera_front_wide_120fov": 0,
#     "camera_cross_right_120fov": 1,
#     "camera_rear_right_70fov": 2,
#     "camera_rear_tele_30fov": 3,
#     "camera_rear_left_70fov": 4,
#     "camera_cross_left_120fov": 5,
#     "camera_front_tele_30fov": 6,
# }

# view_keys = list(CAMERA_TO_VIEW_INDEX.keys())

# # Camera-specific caption prefixes describing camera position and orientation
# CAMERA_TO_CAPTION_PREFIX = {
#     "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
#     "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
#     "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
#     "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
#     "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
#     "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
#     "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
# }
DEFAULT_PROMPT = "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution."
# """
# A clear daytime driving scene on an open road. The weather is sunny with bright natural lighting and good visibility. 
# The sky is partly cloudy with scattered white clouds. The road surface is dry and well-maintained. 
# The overall atmosphere is calm and peaceful with moderate traffic conditions. The lighting creates clear 
# shadows and provides excellent contrast for safe navigation."""


def load_video(
    video_path: str,
    target_frames: int = 93,
    target_size: tuple[int, int] = (720, 1280),
    allow_variable_length: bool = False
) -> th.Tensor:
    """
    Load video and process it to target size and frame count.

    Args:
        video_path: Path to video file
        target_frames: Target number of frames (only enforced if allow_variable_length=False)
        target_size: Target resolution (H, W)
        allow_variable_length: If True, load all frames without padding/cropping to target_frames

    Returns:
        Video tensor with shape (C, T, H, W), dtype uint8
    """
    try:
        # Load video using easy_io
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    video_tensor = th.from_numpy(video_frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    C, T, H, W = video_tensor.shape

    # Handle variable length mode
    if not allow_variable_length:
        # Original behavior: adjust to target_frames
        if T > target_frames:
            video_tensor = video_tensor[:, :target_frames, :, :]
        elif T < target_frames:
            # Pad with last frame
            last_frame = video_tensor[:, -1:, :, :]
            padding_frames = target_frames - T
            last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)
            video_tensor = th.cat([video_tensor, last_frame_repeated], dim=1)
    # else: keep all frames as-is for variable length mode

    # Convert to uint8: (C, T, H, W) -> (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = (video_tensor * 255.0).to(th.uint8)

    # Adjust resolution
    target_h, target_w = target_size
    if H != target_h or W != target_w:
        # Use resize and center crop
        video_tensor = resize_and_crop(video_tensor, target_size)

    # Convert back to (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)

    return video_tensor


def resize_and_crop(video: th.Tensor, target_size: tuple[int, int]) -> th.Tensor:
    """
    Resize video and center crop.

    Args:
        video: Input video with shape (T, C, H, W)
        target_size: Target resolution (H, W)

    Returns:
        Resized video with shape (T, C, target_H, target_W)
    """
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = target_size

    # Calculate scaling ratio to match the smaller dimension to target
    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(scaling_ratio * orig_h), int(scaling_ratio * orig_w))

    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, target_size)

    return video_cropped


def load_multiview_videos(
    input_root: Path,
    video_id: str,
    camera_order: list[str],
    target_frames: int = 93,
    target_size: tuple[int, int] = (720, 1280),
    folder_name: str = "videos",
    allow_variable_length: bool = False,
) -> th.Tensor:
    """
    Load multi-view videos from a specified folder.

    Args:
        input_root: Input root directory
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order
        target_frames: Target number of frames per view
        target_size: Target resolution (H, W)
        folder_name: Name of the folder containing videos (e.g., "videos" or "control")
        allow_variable_length: If True, load all frames without padding/cropping

    Returns:
        Multi-view video tensor with shape (C, V*T, H, W)
    """
    videos_dir = input_root / folder_name
    video_tensors = []

    for camera in camera_order:
        if (videos_dir / f"{camera}").exists():
            sub_dir = f"{camera}"
        elif (videos_dir / camera).exists():
            sub_dir = camera
        else:
            raise FileNotFoundError(f"Folder not found: {videos_dir / f'{camera}'} or {videos_dir / camera}")

        video_path = videos_dir / sub_dir / f"{video_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load single view video: (C, T, H, W)
        video_tensor = load_video(str(video_path), target_frames, target_size, allow_variable_length=allow_variable_length)
        video_tensors.append(video_tensor)

    # Concatenate all views: (C, V*T, H, W)
    multiview_video = th.cat(video_tensors, dim=1)

    return multiview_video


def load_multiview_captions(
    input_root: Path, video_id: str, camera_order: list[str]
) -> list[str]:
    """
    Load multi-view captions. Uses default prompt if captions directory does not exist.

    Args:
        input_root: Input root directory
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order

    Returns:
        List of captions, one per view
    """
    captions_dir = input_root / "captions"

    # If captions directory does not exist, use default prompt
    if not captions_dir.exists():
        log.warning(
            f"Captions directory not found: {captions_dir}. Using default driving scene prompt for all cameras."
        )
        return [DEFAULT_PROMPT] * len(camera_order)

    captions = []

    for camera in camera_order:
        if (captions_dir / f"{camera}").exists():
            sub_dir = f"{camera}"
        elif (captions_dir / camera).exists():
            sub_dir = camera
        else:
            raise FileNotFoundError(f"Folder not found: {captions_dir / f'{camera}'} or {captions_dir / camera}")

        caption_filename = f"{sub_dir}/{video_id}.txt"
        caption_path = captions_dir / caption_filename

        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        captions.append(caption)

    return captions


def construct_data_batch(
    multiview_video: th.Tensor,
    control_video: th.Tensor,
    captions: list[str],
    camera_order: list[str],
    num_conditional_frames: int = 1,
    fps: float = 10.0,
    target_frames_per_view: int = 93,
) -> dict:
    """
    Construct data_batch for model inference.

    Args:
        multiview_video: Multi-view input video tensor with shape (C, V*T, H, W)
        control_video: Multi-view control video tensor with shape (C, V*T, H, W)
        captions: List of captions
        camera_order: List of camera names in order
        num_conditional_frames: Number of conditional frames
        fps: Frames per second
        target_frames_per_view: Number of frames per view

    Returns:
        data_batch dictionary
    """
    C, VT, H, W = multiview_video.shape
    n_views = len(camera_order)
    T = VT // n_views

    # Add batch dimension: (C, V*T, H, W) -> (1, C, V*T, H, W)
    multiview_video = multiview_video.unsqueeze(0)
    control_video = control_video.unsqueeze(0)

    # Construct correct view_indices based on camera order
    # Each view's T frames all use that view's corresponding view index
    view_indices_list = []
    for idx,camera in enumerate(camera_order):
        view_idx = idx
        view_indices_list.extend([view_idx] * T)
    view_indices = th.tensor(view_indices_list, dtype=th.int64).unsqueeze(0)  # (1, V*T)

    # Construct view_indices_selection: view indices of cameras in camera_order
    view_indices_selection = th.tensor(
        [idx for idx,camera in enumerate(camera_order)], dtype=th.int64
    ).unsqueeze(0)  # (1, n_views)

    # Find position of front_wide_120fov in camera_order as ref_cam_view_idx_sample_position
    ref_cam_position = (
        camera_order.index("camera_front_wide_120fov") if "camera_front_wide_120fov" in camera_order else 0
    )

    # Construct data_batch
    data_batch = {
        "video": multiview_video,
        "control_input_hdmap_bbox": control_video,
        "ai_caption": [captions],
        "view_indices": view_indices,  # (1, V*T), using correct view index
        "fps": th.tensor([fps], dtype=th.float64),
        "chunk_index": th.tensor([0], dtype=th.int64),
        "frame_indices": th.arange(target_frames_per_view).unsqueeze(0),  # (1, T)
        "num_video_frames_per_view": th.tensor([target_frames_per_view], dtype=th.int64),
        "view_indices_selection": view_indices_selection,  # (1, n_views), using correct view index
        "camera_keys_selection": [camera_order],
        "sample_n_views": th.tensor([n_views], dtype=th.int64),
        "padding_mask": th.zeros(1, 1, H, W, dtype=th.float32),
        "ref_cam_view_idx_sample_position": th.tensor([ref_cam_position], dtype=th.int64),
        "front_cam_view_idx_sample_position": [None],
        "original_hw": th.tensor([[[H, W]] * n_views], dtype=th.int64),  # (1, n_views, 2)
        NUM_CONDITIONAL_FRAMES_KEY: num_conditional_frames,
    }

    return data_batch


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transfer2 Multiview inference from videos, control videos, and captions"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # Generation parameters
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance value")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Number of conditional frames")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion steps")
    parser.add_argument(
        "--use_negative_prompt",
        action="store_true",
        default=True,
        help="Use default negative prompt for additional guidance.",
    )
    parser.add_argument("--distillation", type=str, default="", help="Distillation type.", choices=["", "dmd2"])
    parser.add_argument("--control_weight", type=float, default=1.0, help="Control weight")
    # Input/output
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Input root directory containing videos/, control/, and captions/ subdirectories",
    )
    parser.add_argument(
        "--view_meta",
        type=str,
        required=True,
        help="json file with camera dicts and default prompts",
    )
    parser.add_argument("--save_root", type=str, default="results/transfer2_multiview_cli/", help="Save root")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to generate")
    parser.add_argument(
        "--stack_mode",
        type=str,
        default="width",
        choices=["height", "width", "time", "grid", "grid_auto"],
        help="Video stacking mode for visualization. 'width': horizontal row, 'height': vertical column, "
             "'grid': predefined 3x3 grid (requires specific camera keys), 'grid_auto': auto-balanced grid for any views (recommended), "
             "'time': no spatial rearrangement.",
    )
    # Video parameters
    parser.add_argument("--target_frames", type=int, default=93, help="Target number of frames per view")
    parser.add_argument("--target_height", type=int, default=720, help="Target video height")
    parser.add_argument("--target_width", type=int, default=1280, help="Target video width")
    # Caption parameters
    parser.add_argument(
        "--add_camera_prefix",
        action="store_true",
        default=True,
        help="Add camera-specific position/orientation prefix to captions",
    )
    parser.add_argument(
        "--no_camera_prefix",
        action="store_false",
        dest="add_camera_prefix",
        help="Do not add camera-specific prefix to captions",
    )
    # Save options
    parser.add_argument(
        "--save_each_view",
        action="store_true",
        help="Save each camera view as a separate video file",
    )
    # Autoregressive generation
    parser.add_argument(
        "--use_autoregressive",
        action="store_true",
        help="Use autoregressive generation for long videos",
    )
    parser.add_argument(
        "--use_cache_offload",
        action="store_true",
        help="Use generation caches for long videos",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=1,
        help="Number of overlapping frames between chunks in autoregressive generation",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the inference.")
    parser.add_argument("--hierarchical_cp", action="store_true", help="Use hierarchical CP algorithm (a2a + p2p)")
    # Experiment options
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def main():
    os.environ["NVTE_FUSED_ATTN"] = "0"
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.enable_grad(False)

    args = parse_arguments()

    # Initialize directories and camera metadata
    input_root = Path(args.input_root)
    with open(args.view_meta) as f:
        camera_dict = json.load(f)
    view_keys = list(camera_dict.keys())
    videos_dir = input_root / "videos"

    # Prepare experiment options
    experiment_opts = list(args.opts) if args.opts else []
    if args.use_cuda_graphs:
        experiment_opts.append("model.config.net.use_cuda_graphs=True")
    if args.hierarchical_cp:
        experiment_opts.append("model.config.net.atten_backend='transformer_engine'")

    # Initialize inference handler
    vid2world_cli = ControlVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
        hierarchical_cp=args.hierarchical_cp,
        experiment_opts=experiment_opts,
    )
    mem_bytes = th.cuda.memory_allocated(device=th.device("cuda" if th.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Calculate chunk size from model's state_t (frames per chunk)
    # Formula: chunk_size = 1 + (state_t - 1) * 4
    # Example: state_t=24 → chunk_size=93 frames
    chunk_size = vid2world_cli.model.tokenizer.get_pixel_num_frames(
        vid2world_cli.model.config.state_t
    )

    if rank0:
        log.info(f"Model state_t={vid2world_cli.model.config.state_t}, chunk_size={chunk_size} frames per view")

    # Create output directory
    os.makedirs(args.save_root, exist_ok=True)

    # Verify required directories exist
    control_dir = input_root / "control"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    if not control_dir.exists():
        raise FileNotFoundError(f"World scenario directory not found: {control_dir}")

    # Find first camera directory and get video files
    if (videos_dir / f"{view_keys[0]}").exists():
        first_camera_dir = videos_dir / f"{view_keys[0]}"
    else:
        first_camera_dir = videos_dir / view_keys[0]

    video_files = sorted(first_camera_dir.glob("*.mp4"))
    if len(video_files) == 0:
        raise FileNotFoundError(f"No video files found in {first_camera_dir}")

    # Get all video IDs (from first camera directory)
    video_ids = [f.stem for f in video_files[: args.max_samples]]
    log.info(f"Found {len(video_ids)} video IDs, processing {min(len(video_ids), args.max_samples)} samples")

    for i, video_id in enumerate(video_ids):
        if rank0:
            log.info(f"Processing sample {i + 1}/{len(video_ids)}: {video_id}")

        try:
            # NEW: Load first video to detect actual frame count
            first_video_path = videos_dir / f"{view_keys[0]}" / f"{video_id}.mp4"
            video_frames, _ = easy_io.load(str(first_video_path))
            detected_frames = video_frames.shape[0]

            if args.use_autoregressive:
                # Calculate optimal num_chunks for detected frame count
                effective_chunk_size = chunk_size - args.chunk_overlap
                num_chunks = max(1, (detected_frames - args.chunk_overlap + effective_chunk_size - 1) // effective_chunk_size)
                expected_frames = calculate_autoregressive_frames(chunk_size, args.chunk_overlap, num_chunks)

                if rank0:
                    log.info(f"Detected {detected_frames} frames, using {num_chunks} chunks")
                    log.info(f"Autoregressive mode will process {expected_frames} frames")

                # Use detected frame count as-is
                target_frames = detected_frames
            else:
                # Non-autoregressive: must match chunk_size exactly
                if detected_frames != chunk_size:
                    raise ValueError(
                        f"Non-autoregressive mode requires exactly {chunk_size} frames "
                        f"(based on state_t={vid2world_cli.model.config.state_t}), "
                        f"but video has {detected_frames} frames. "
                        f"Use --use_autoregressive for longer videos or resize videos to {chunk_size} frames."
                    )
                target_frames = chunk_size

            # Load multi-view input videos
            multiview_video = load_multiview_videos(
                input_root,
                video_id,
                view_keys,
                target_frames=target_frames,
                target_size=(args.target_height, args.target_width),
                folder_name="videos",
                allow_variable_length=args.use_autoregressive,
            )
            if rank0:
                log.info(f"Loaded input multiview video: {multiview_video.shape}")

            # Load multi-view control videos (control)
            control_video = load_multiview_videos(
                input_root,
                video_id,
                view_keys,
                target_frames=target_frames,
                target_size=(args.target_height, args.target_width),
                folder_name="control",
                allow_variable_length=args.use_autoregressive,
            )
            if rank0:
                log.info(f"Loaded control video (control): {control_video.shape}")

            # Load multi-view captions
            captions = load_multiview_captions(
                input_root, video_id, view_keys
            )
            # Add camera-specific prefix if enabled
            if args.add_camera_prefix:
                for idx,camera in enumerate(view_keys):
                    captions[idx] = f"{camera_dict[camera]} {captions[idx]}"  

            if rank0:
                log.info(f"Loaded {len(captions)} captions")
                log.info(f"First caption preview: {captions[0][:100]}...")

            # Construct data_batch
            data_batch = construct_data_batch(
                multiview_video,
                control_video,
                captions,
                view_keys,
                num_conditional_frames=args.num_conditional_frames,
                fps=args.fps,
                target_frames_per_view=args.target_frames,
            )

            # Add control weight
            data_batch["control_weight"] = args.control_weight

            # Run inference
            if args.use_autoregressive:
                # Remove num_conditional_frames from batch for autoregressive mode
                # (it should be passed as function argument instead)
                if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
                    del data_batch[NUM_CONDITIONAL_FRAMES_KEY]
                # Use autoregressive generation
                import time

                th.cuda.synchronize()
                start_time = time.time()
                os.makedirs(args.save_root, exist_ok=True)
                video, control = vid2world_cli.generate_autoregressive_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + i,
                    num_conditional_frames=args.num_conditional_frames,
                    num_steps=args.num_steps,
                    n_views=len(view_keys),
                    chunk_size=vid2world_cli.model.tokenizer.get_pixel_num_frames(vid2world_cli.model.config.state_t),
                    chunk_overlap=args.chunk_overlap,
                    use_negative_prompt=args.use_negative_prompt,
                    distillation=args.distillation,
                    dynamic_cache_chunks=args.use_cache_offload,
                    dynamic_cache_fps=args.fps,
                    dynamic_cache_dir=args.save_root
                )

                th.cuda.synchronize()
                end_time = time.time()
                if rank0:
                    log.info(f"Time taken for autoregressive generation: {end_time - start_time:.2f} seconds")

                if rank0:
                    log.info(f"Video shape before unsqueeze: {video.shape}, Control shape: {control.shape}")

                # Add batch dimension for saving
                video = video.unsqueeze(0)
                control = control.unsqueeze(0)

                if rank0:
                    log.info(f"Video shape after unsqueeze: {video.shape}, Control shape: {control.shape}")
                    log.info(f"Starting arrange_video_visualization with stack_mode={args.stack_mode}")

                # Check for CUDA errors before proceeding
                th.cuda.synchronize()
                if rank0:
                    log.info("CUDA synchronized successfully after generation")

                # Move to CPU before arranging to avoid GPU OOM
                # Grid layout with large videos creates ~108GB intermediate tensors
                if rank0:
                    log.info("Moving tensors to CPU to avoid GPU memory issues during arrangement...")
                video_cpu = video.cpu()
                control_cpu = control.cpu()

                if rank0:
                    log.info(f"Moved to CPU. Freeing GPU memory...")
                # Free GPU memory
                del video, control
                th.cuda.empty_cache()

                # Apply visualization layout on CPU
                if rank0:
                    log.info(f"Arranging video on CPU (this may take a minute)...")
                try:
                    video_arranged = arrange_video_visualization(video_cpu, data_batch, method=args.stack_mode)
                    if rank0:
                        log.info(f"Video arranged successfully, shape: {video_arranged.shape}")
                except Exception as e:
                    if rank0:
                        log.error(f"Error arranging video: {e}")
                    raise

                if rank0:
                    log.info(f"Arranging control on CPU...")
                try:
                    control_arranged = arrange_video_visualization(control_cpu, data_batch, method=args.stack_mode)
                    if rank0:
                        log.info(f"Control arranged successfully, shape: {control_arranged.shape}")
                except Exception as e:
                    if rank0:
                        log.error(f"Error arranging control: {e}")
                    raise

                # Free CPU memory from original tensors
                del video_cpu, control_cpu

                # Create save directory
                if rank0:
                    os.makedirs(args.save_root, exist_ok=True)
                    log.info(f"Save directory created/verified: {args.save_root}")
                if rank0:
                    video_path = f"{args.save_root}/inference_{video_id}_video"
                    save_img_or_video(video_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved video to {video_path}")

                    video_path = f"{args.save_root}/inference_{video_id}_control"
                    save_img_or_video(control_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved control to {video_path}")

                    # Save each view separately if requested (only generated video, not control)
                    if args.save_each_view:
                        save_dir = f"{args.save_root}/inference_{video_id}"
                        save_each_view_separately(
                            mv_video=video[0],
                            data_batch=data_batch,
                            save_dir=save_dir,
                            fps=args.fps,
                        )

            else:
                # Extract control video from data_batch for saving
                control_video = (
                    data_batch["control_input_hdmap_bbox"].float() / 255.0
                ).cpu()  # (1, 3, V*T, H, W), convert to [0,1]

                # Use single-shot generation
                video = vid2world_cli.generate_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + i,
                    num_steps=args.num_steps,
                    use_negative_prompt=args.use_negative_prompt,
                    distillation=args.distillation,
                ).cpu()

                # Apply visualization layout
                video_arranged = arrange_video_visualization(video, data_batch, method=args.stack_mode)
                control_arranged = arrange_video_visualization(control_video, data_batch, method=args.stack_mode)

                # Save results
                if rank0:
                    video_path = f"{args.save_root}/inference_{video_id}_video"
                    save_img_or_video(video_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved video to {video_path}")

                    video_path = f"{args.save_root}/inference_{video_id}_control"
                    save_img_or_video(control_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved control to {video_path}")

                    # Save each view separately if requested (only generated video, not control)
                    if args.save_each_view:
                        save_dir = f"{args.save_root}/inference_{video_id}"
                        save_each_view_separately(
                            mv_video=video[0],
                            data_batch=data_batch,
                            save_dir=save_dir,
                            fps=args.fps,
                        )

        except Exception as e:
            log.error(f"Error processing {video_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Synchronize all processes
    if args.context_parallel_size > 1:
        th.distributed.barrier()

    # Cleanup distributed resources
    vid2world_cli.cleanup()


if __name__ == "__main__":
    main()
