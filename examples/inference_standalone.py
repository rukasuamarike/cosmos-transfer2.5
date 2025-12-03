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
Standalone inference script for Cosmos-Transfer2.5 that doesn't depend on cosmos_oss.

Usage:
    # Single GPU:
    python -m examples.inference_standalone -i config.json -o outputs/

    # Multi-GPU with context parallelism:
    torchrun --nproc_per_node=2 -m examples.inference_standalone -i config.json -o outputs/ --context-parallel-size 2
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Union

import pydantic
import torch
import tyro


def init_environment():
    """Initialize the environment for distributed training."""
    # Set up CUDA
    if torch.cuda.is_available():
        # Set device based on local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    # Disable gradients globally for inference
    torch.set_grad_enabled(False)


def cleanup_environment():
    """Clean up distributed environment."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def init_output_dir(output_dir: Path, profile: bool = False):
    """Initialize output directory."""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)


# Import after environment setup to avoid issues
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    InferenceArguments,
    InferenceOverrides,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
    ],
    tyro.conf.ConsolidateSubcommandArgs,
]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path(s) to the inference parameter file(s).
    If multiple files are provided, run "batch" inference. The model will be loaded once and all samples run sequentially.
    If there are different hint keys across the batch, the multicontrol model will be used regardless of the each sample's hint keys.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""

    control: ControlUnion = EdgeConfig()
    """Control help. Run control:edge --help for more information about edge etc."""


def main(args: Args):
    inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)

    if args.setup.benchmark:
        if len(inference_samples) == 1:
            inference_samples = inference_samples * 4
            log.info(f"Repeating inference sample 4 times for benchmarking.")

    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference import Control2WorldInference

    inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)

    main(args)
    cleanup_environment()
