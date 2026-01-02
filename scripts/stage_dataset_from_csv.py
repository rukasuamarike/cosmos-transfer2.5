#!/usr/bin/env python3
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
CSV-Driven Dataset Staging Script

NOTE: Designed for container usage where:
- S3 is mounted at specified path (e.g., /mnt/s3-bucket/)
- Staging directory is writable (e.g., /root/app/dataset/)
- User has permissions to create symlinks

This script reads a CSV manifest and stages a multiview dataset by symlinking
files from mounted S3 into the directory structure expected by MultiviewTransferDataset.

Expected CSV columns:
- sample_id: Unique identifier for the sample
- canonical_camera: One of the 7 CAMERA_FOLDERS values
- video_path: Relative path from s3_mount to video file
- caption_path: Relative path from s3_mount to caption JSON
- control_path: Relative path from s3_mount to control video
- source_dataset (optional): Original dataset name
- source_camera (optional): Original camera name
- fps (optional): Original FPS

Required directory structure (created by this script):
dataset_dir/
├── videos/
│   ├── camera_front_wide_120fov/
│   ├── camera_cross_left_120fov/
│   ├── camera_cross_right_120fov/
│   ├── camera_rear_left_70fov/
│   ├── camera_rear_right_70fov/
│   ├── camera_rear_tele_30fov/
│   └── camera_front_tele_30fov/
├── captions/
│   ├── camera_front_wide_120fov/
│   └── ... (same camera folders)
└── control/
    ├── camera_front_wide_120fov/
    └── ... (same camera folders)
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


# Pydantic schema for CSV validation
class DatasetRow(BaseModel):
    """Schema for dataset CSV rows with automatic validation."""

    sample_id: str
    canonical_camera: Literal[
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
        "camera_front_tele_30fov",
    ]
    video_path: str
    caption_path: str
    control_path: str
    source_dataset: str | None = None
    source_camera: str | None = None
    fps: float | None = None

    @field_validator("sample_id", "video_path", "caption_path", "control_path")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# Constants
CAMERA_FOLDERS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_front_tele_30fov",
]


def load_manifest(csv_path: Path) -> list[DatasetRow]:
    """Load and validate CSV manifest using Pydantic."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row_dict in enumerate(reader, start=1):
            try:
                rows.append(DatasetRow(**row_dict))
            except Exception as e:
                print(f"Error: Invalid data in row {idx}")
                print(f"  {e}")
                sys.exit(1)
    return rows


def group_by_sample(rows: list[DatasetRow]) -> dict[str, list[DatasetRow]]:
    """Group rows by sample_id."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.sample_id].append(row)
    return dict(groups)


def create_directories(staging_dir: Path) -> None:
    """Create the required directory structure."""
    for subdir in ["videos", "captions", "control"]:
        for camera in CAMERA_FOLDERS:
            (staging_dir / subdir / camera).mkdir(parents=True, exist_ok=True)


def stage_sample(sample_id: str, rows: list[DatasetRow], s3_mount: Path, staging_dir: Path) -> None:
    """Stage a single sample by creating symlinks and caption files."""
    for row in rows:
        camera = row.canonical_camera
        # Construct file paths
        video_src = s3_mount / row.video_path
        caption_src = s3_mount / row.caption_path
        control_src = s3_mount / row.control_path

        video_dst = staging_dir / "videos" / camera / f"{sample_id}.mp4"
        caption_dst = staging_dir / "captions" / camera / f"{sample_id}.json"
        control_dst = staging_dir / "control" / camera / f"{sample_id}.mp4"

        try:
            # Validate source files exist
            if not video_src.exists():
                raise FileNotFoundError(f"Video file not found: {video_src}")
            if not caption_src.exists():
                raise FileNotFoundError(f"Caption file not found: {caption_src}")
            if not control_src.exists():
                raise FileNotFoundError(f"Control file not found: {control_src}")

            # Create symlinks for video and control
            if video_dst.exists():
                video_dst.unlink()
            video_dst.symlink_to(video_src)

            if control_dst.exists():
                control_dst.unlink()
            control_dst.symlink_to(control_src)

            # Copy and format caption JSON
            with open(caption_src, "r", encoding="utf-8") as f:
                caption_data = json.load(f)

            # Ensure caption is in the correct format
            if isinstance(caption_data, dict) and "caption" in caption_data:
                caption_text = caption_data["caption"]
            elif isinstance(caption_data, str):
                caption_text = caption_data
            else:
                caption_text = str(caption_data)

            # Write caption in standardized format
            with open(caption_dst, "w", encoding="utf-8") as f:
                json.dump({"caption": caption_text}, f, ensure_ascii=False, indent=2)

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            raise
        except Exception as e:
            print(f"  Error staging {sample_id}/{camera}: {e}")
            raise


def validate_sample_groups(groups: dict[str, list[DatasetRow]]) -> None:
    """Validate that all samples have the correct number of cameras."""
    for sample_id, rows in groups.items():
        if len(rows) != len(CAMERA_FOLDERS):
            print(f"Warning: Sample {sample_id} has {len(rows)} cameras, expected {len(CAMERA_FOLDERS)}")

        # Check for duplicate cameras
        cameras = [row.canonical_camera for row in rows]
        if len(cameras) != len(set(cameras)):
            duplicates = [cam for cam in cameras if cameras.count(cam) > 1]
            print(f"Error: Sample {sample_id} has duplicate cameras: {duplicates}")
            sys.exit(1)


def validate_source_files(groups: dict[str, list[DatasetRow]], s3_mount: Path) -> None:
    """Validate that all source files exist."""
    missing_files = []

    for sample_id, rows in groups.items():
        for row in rows:
            video_path = s3_mount / row.video_path
            caption_path = s3_mount / row.caption_path
            control_path = s3_mount / row.control_path

            if not video_path.exists():
                missing_files.append(f"Video: {video_path}")
            if not caption_path.exists():
                missing_files.append(f"Caption: {caption_path}")
            if not control_path.exists():
                missing_files.append(f"Control: {control_path}")

    if missing_files:
        print("Error: Missing source files:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        sys.exit(1)


def verify_staged_dataset(staging_dir: Path) -> None:
    """Verify the staged dataset structure."""
    videos_dir = staging_dir / "videos"
    captions_dir = staging_dir / "captions"
    control_dir = staging_dir / "control"

    video_count = sum(1 for _ in videos_dir.rglob("*.mp4"))
    caption_count = sum(1 for _ in captions_dir.rglob("*.json"))
    control_count = sum(1 for _ in control_dir.rglob("*.mp4"))

    print(f"Videos: {video_count}")
    print(f"Captions: {caption_count}")
    print(f"Control: {control_count}")

    if video_count != caption_count or video_count != control_count:
        print("Warning: File counts don't match!")
    else:
        print("Dataset staging complete")


def clean_staging_dir(staging_dir: Path) -> None:
    """Clean the staging directory."""
    if staging_dir.exists():
        print(f"Cleaning staging directory: {staging_dir}")
        shutil.rmtree(staging_dir)
        print("Staging directory cleaned")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage multiview dataset from CSV manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage dataset from CSV
  python stage_dataset_from_csv.py \\
      --csv manifest.csv \\
      --s3-mount /mnt/s3-bucket \\
      --staging-dir /root/app/dataset

  # Clean and restage
  python stage_dataset_from_csv.py \\
      --csv manifest.csv \\
      --s3-mount /mnt/s3-bucket \\
      --staging-dir /root/app/dataset \\
      --clean
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV manifest file",
    )
    parser.add_argument(
        "--s3-mount",
        type=str,
        required=True,
        help="Path to mounted S3 bucket (e.g., /mnt/s3-bucket)",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        required=True,
        help="Directory where dataset will be staged (e.g., /root/app/dataset)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean staging directory before starting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).resolve()
    s3_mount = Path(args.s3_mount).resolve()
    staging_dir = Path(args.staging_dir).resolve()

    # Validate inputs
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    if not s3_mount.exists():
        print(f"Error: S3 mount point not found: {s3_mount}")
        sys.exit(1)

    # Clean if requested
    if args.clean:
        clean_staging_dir(staging_dir)

    # Load CSV manifest (Pydantic validates automatically)
    print("Loading CSV manifest...")
    rows = load_manifest(csv_path)

    if not rows:
        print("Error: CSV file is empty")
        sys.exit(1)

    # Group by sample
    groups = group_by_sample(rows)
    print(f"Loaded {len(rows)} entries for {len(groups)} samples")

    # Validate sample groups
    print("\nValidating sample groups...")
    validate_sample_groups(groups)
    all_have_7 = all(len(rows) == len(CAMERA_FOLDERS) for rows in groups.values())
    if all_have_7:
        print("All samples have 7 cameras")

    # Validate source files
    print("\nValidating source files...")
    validate_source_files(groups, s3_mount)
    print("All source files found")

    # Create directory structure
    print("\nCreating directory structure...")
    create_directories(staging_dir)
    print("Directories created")

    # Stage samples
    print(f"\nStaging {len(groups)} samples...")
    for idx, (sample_id, sample_rows) in enumerate(groups.items(), 1):
        try:
            stage_sample(sample_id, sample_rows, s3_mount, staging_dir)
            if idx % 10 == 0 or idx == len(groups):
                print(f"  Progress: {idx}/{len(groups)} samples", end="\r", flush=True)
        except Exception as e:
            print(f"\nError staging sample {sample_id}: {e}")
            sys.exit(1)

    print()  # New line after progress

    # Verify staged dataset
    print("\nVerifying staged dataset...")
    verify_staged_dataset(staging_dir)

    print(f"\nStaged dataset ready at: {staging_dir}")


if __name__ == "__main__":
    main()
