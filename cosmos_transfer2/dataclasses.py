# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for Transfer2.5 with JSONL batch generation support."""

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from pydantic import BaseModel, Field

class ControlConfig(BaseModel):
    control_path: str | None = None
    mask_path: str | None = None
    control_weight: float = 1.0
    mask_prompt: str | None = None

class EdgeConfig(ControlConfig):
    preset_edge_threshold: str = "medium"

class BlurConfig(ControlConfig):
    preset_blur_strength: str = "medium"

class SegConfig(ControlConfig):
    control_prompt: str | None = None

class DepthConfig(ControlConfig):
    pass

class ControlSpec(BaseModel):
    depth: DepthConfig | None = None
    edge: EdgeConfig | None = None
    seg: SegConfig | None = None
    vis: BlurConfig | None = None
    negative_prompt: str | None = None


class InferenceSample(BaseModel):
    """Schema for a single line in the JSONL batch file."""
    name: str = Field(..., description="Unique identifier (used for output filename)")
    prompt: str = Field(..., description="Text prompt for generation")
    video_path: str = Field(..., description="Path to input video")
    negative_prompt: str | None = Field(None, description="Negative guidance prompt")
    seed: int = Field(1, description="Random seed")
    guidance: float = Field(7.0, description="CFG scale")
    num_steps: int = Field(35, description="Number of diffusion steps")
    sigma_max: float | None = Field(None, description="Max noise level")
    num_conditional_frames: int = Field(1, description="Frames to condition on")
    num_video_frames_per_chunk: int = Field(93, description="Frames per chunk")
    resolution: str = Field("1024x576", description="Output resolution")
    image_context_path: str | None = Field(None, description="Image for context")
    edge: EdgeConfig | None = Field(None, description="Edge control")
    depth: DepthConfig | None = Field(None, description="Depth control")
    vis: BlurConfig | None = Field(None, description="Blur control")
    seg: SegConfig | None = Field(None, description="Segmentation control")

    def to_jsonl_dict(self) -> dict:
        data = self.model_dump(exclude_none=True)
        defaults = {
            "seed": 1, "guidance": 7.0, "num_steps": 35, "num_conditional_frames": 1,
            "num_video_frames_per_chunk": 93, "resolution": "1024x576",
        }
        for key, default_val in defaults.items():
            if key in data and data[key] == default_val:
                del data[key]
        return data

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_jsonl_dict())


class SetupConfig(BaseModel):
    model: str = "edge"
    checkpoint_path: str | None = None
    experiment: str | None = None
    config_file: str = "cosmos_transfer2/_src/predict2/configs/video2world/config.py"
    context_parallel_size: int | None = None
    disable_guardrails: bool = False
    offload_guardrail_models: bool = True
    keep_going: bool = True
    profile: bool = False
    benchmark: bool = False
    compile_tokenizer: str = "none"
    enable_parallel_tokenizer: bool = False
    parallel_tokenizer_grid: tuple[int, int] = (-1, -1)


class FileData(BaseModel):
    original_video_id: str
    new_file_ids: List[str]


class BatchVideoRequest(BaseModel):
    user_id: str
    dataset_id: str
    prompt_list: List[str]
    video_list: List[FileData]
    job_id: str | None = None
    spec_opt: ControlSpec | None = None
    setup_opt: SetupConfig | None = None
    config_path: str | None = None
    save_ctrl: bool | None = None
    guidance: int | None = 3
    num_steps: int | None = 35
    resolution: str | None = "720"
    seed: int | None = 2025
    num_conditional_frames: int | None = 1


class BatchSubmitBody(BaseModel):
    request: BatchVideoRequest
    batch_size: int = 1


@dataclass(frozen=True)
class Transfer2Item:
    video_id: str
    new_ids: List[str]
    local_load_path: str
    local_save_dir: str
    remote_load_path: str
    remote_save_dir: str
    save_ctrl: bool | None = None


@dataclass(frozen=True)
class Transfer2Job:
    id: str
    spec_path: str
    items: List[Transfer2Item]
    setup_config: SetupConfig | None = None

    @classmethod
    def from_req(
        cls, body: "BatchSubmitBody", job_id: str | None = None,
        mount_path: str = "/mnt", job_dir: str = "/jobs",
    ) -> "Transfer2Job":
        req = body.request
        request_dir = Path(req.user_id) / req.dataset_id
        job_id = job_id or str(uuid.uuid4())
        items = []
        for filedata in req.video_list:
            video = Transfer2Item(
                video_id=filedata.original_video_id,
                new_ids=filedata.new_file_ids,
                remote_load_path=str(Path(mount_path) / request_dir / f"raw/{filedata.original_video_id}.mp4"),
                local_load_path=str(Path(job_dir) / request_dir / job_id / f"{filedata.original_video_id}.mp4"),
                local_save_dir=str(Path(job_dir) / request_dir / job_id),
                remote_save_dir=str(Path(mount_path) / request_dir / "generated"),
                save_ctrl=req.save_ctrl,
            )
            items.append(video)
        video = items[0] if items else None
        prompt = req.prompt_list[0] if req.prompt_list else "A realistic video"
        spec_path = make_spec_json(
            prompt=prompt, video_path=video.local_load_path if video else "",
            spec=req.spec_opt, output_dir=str(Path(job_dir) / request_dir / job_id),
            guidance=req.guidance or 3, num_steps=req.num_steps or 35,
            resolution=req.resolution or "720", seed=req.seed or 2025,
            num_conditional_frames=req.num_conditional_frames or 1, filename="output",
        )
        return cls(id=job_id, spec_path=spec_path, items=items, setup_config=req.setup_opt)


@dataclass
class BatchJSONLGenerator:
    """Generator for creating JSONL batch files for Transfer2.5 inference."""
    samples: List[InferenceSample] = field(default_factory=list)

    def add_sample(
        self, name: str, prompt: str, video_path: str, *,
        negative_prompt: str | None = None, seed: int = 1, guidance: float = 7.0,
        num_steps: int = 35, sigma_max: float | None = None, num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93, resolution: str = "1024x576",
        image_context_path: str | None = None, edge: EdgeConfig | dict | None = None,
        depth: DepthConfig | dict | None = None, vis: BlurConfig | dict | None = None,
        seg: SegConfig | dict | None = None,
    ) -> "BatchJSONLGenerator":
        if isinstance(edge, dict):
            edge = EdgeConfig(**edge)
        if isinstance(depth, dict):
            depth = DepthConfig(**depth)
        if isinstance(vis, dict):
            vis = BlurConfig(**vis)
        if isinstance(seg, dict):
            seg = SegConfig(**seg)
        sample = InferenceSample(
            name=name, prompt=prompt, video_path=video_path, negative_prompt=negative_prompt,
            seed=seed, guidance=guidance, num_steps=num_steps, sigma_max=sigma_max,
            num_conditional_frames=num_conditional_frames,
            num_video_frames_per_chunk=num_video_frames_per_chunk, resolution=resolution,
            image_context_path=image_context_path, edge=edge, depth=depth, vis=vis, seg=seg,
        )
        self.samples.append(sample)
        return self

    def add_from_request(
        self, request: BatchVideoRequest, video_path: str, name_prefix: str = "sample",
    ) -> "BatchJSONLGenerator":
        edge = depth = vis = seg = None
        if request.spec_opt:
            edge, depth = request.spec_opt.edge, request.spec_opt.depth
            vis, seg = request.spec_opt.vis, request.spec_opt.seg
        negative_prompt = request.spec_opt.negative_prompt if request.spec_opt else None
        for i, prompt in enumerate(request.prompt_list):
            self.add_sample(
                name=f"{name_prefix}_{i}", prompt=prompt, video_path=video_path,
                negative_prompt=negative_prompt, seed=(request.seed or 2025) + i,
                guidance=float(request.guidance or 7), num_steps=request.num_steps or 35,
                resolution=request.resolution or "720",
                num_conditional_frames=request.num_conditional_frames or 1,
                edge=edge, depth=depth, vis=vis, seg=seg,
            )
        return self

    def add_from_items(
        self, items: List[Transfer2Item], prompts: List[str], base_seed: int = 2025, **kwargs,
    ) -> "BatchJSONLGenerator":
        sample_idx = 0
        for item in items:
            for new_id, prompt in zip(item.new_ids, prompts):
                self.add_sample(
                    name=new_id, prompt=prompt, video_path=item.local_load_path,
                    seed=base_seed + sample_idx, **kwargs,
                )
                sample_idx += 1
        return self

    def __iter__(self) -> Iterator[InferenceSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def to_jsonl_lines(self) -> List[str]:
        return [sample.to_jsonl_line() for sample in self.samples]

    def to_jsonl_string(self) -> str:
        return "\n".join(self.to_jsonl_lines())

    def write(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sample in self.samples:
                f.write(sample.to_jsonl_line() + "\n")
        return str(path)

    def clear(self) -> "BatchJSONLGenerator":
        self.samples.clear()
        return self


def make_spec_json(
    prompt: str, video_path: str, spec: ControlSpec | None, output_dir: str,
    guidance: int = 3, num_steps: int = 35, resolution: str = "720",
    seed: int = 2025, num_conditional_frames: int = 1, filename: str = "output",
) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(output_dir, "spec.json")
    spec_dict = {
        "name": filename, "prompt": prompt, "video_path": video_path,
        "guidance": guidance, "num_steps": num_steps, "resolution": resolution,
        "seed": seed, "num_conditional_frames": num_conditional_frames,
    }
    if spec:
        spec_data = spec.model_dump(exclude_none=True)
        if "negative_prompt" in spec_data:
            spec_dict["negative_prompt"] = spec_data.pop("negative_prompt")
        for key in ["depth", "edge", "seg", "vis"]:
            if key in spec_data and spec_data[key]:
                spec_dict[key] = spec_data[key]
    with open(json_path, "w") as f:
        json.dump(spec_dict, f, indent=2)
    return json_path


def make_batch_jsonl(
    prompts: List[str], video_path: str, output_dir: str, *,
    spec: ControlSpec | None = None, guidance: float = 7.0, num_steps: int = 35,
    resolution: str = "720", base_seed: int = 2025, num_conditional_frames: int = 1,
    name_prefix: str = "sample", filename: str = "batch.jsonl",
) -> str:
    generator = BatchJSONLGenerator()
    edge = depth = vis = seg = negative_prompt = None
    if spec:
        edge, depth, vis, seg = spec.edge, spec.depth, spec.vis, spec.seg
        negative_prompt = spec.negative_prompt
    for i, prompt in enumerate(prompts):
        generator.add_sample(
            name=f"{name_prefix}_{i}", prompt=prompt, video_path=video_path,
            negative_prompt=negative_prompt, seed=base_seed + i, guidance=guidance,
            num_steps=num_steps, resolution=resolution,
            num_conditional_frames=num_conditional_frames,
            edge=edge, depth=depth, vis=vis, seg=seg,
        )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return generator.write(out_path / filename)


def make_setup_json(setup: SetupConfig, output_dir: str, filename: str = "setup") -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, "w") as f:
        json.dump(setup.model_dump(exclude_none=True), f, indent=2)
    return json_path


__all__ = [
    "ControlConfig", "EdgeConfig", "BlurConfig", "SegConfig", "DepthConfig", "ControlSpec",
    "InferenceSample", "SetupConfig", "FileData", "BatchVideoRequest", "BatchSubmitBody",
    "Transfer2Item", "Transfer2Job", "BatchJSONLGenerator",
    "make_spec_json", "make_batch_jsonl", "make_setup_json",
]
