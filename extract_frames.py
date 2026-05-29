#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame extraction script for IITR-VSOD dataset.

Extracts frames from video files (both infrared source videos and annotation
videos) and saves them as PNG images. GT frames are binarized to strict 0/255.

Expected input structure:
    IITR-VSOD-DATA/
    ├── videos/{action}/{video}.mp4          # Source infrared videos
    └── annotations/{action}/{video}.mp4     # GT annotation videos

Output structure:
    output_dir/
    └── {action}/
        └── {video_name}/
            ├── GT_object_level/             # Binarized GT frames
            │   ├── frame_000000.png
            │   └── ...
            └── Imgs/                        # Source infrared frames
                ├── frame_000000.png
                └── ...

Usage:
    python extract_frames.py \
        --video_root /path/to/videos \
        --anno_root /path/to/annotations \
        --output_dir /path/to/output \
        [--threshold 128]
"""

import os
import sys
import cv2
import argparse

import numpy as np
from tqdm import tqdm


def binarize_frame(frame: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Binarize a grayscale frame: pixels > threshold -> 255, else -> 0.
    
    Args:
        frame: Input frame (grayscale or BGR).
        threshold: Binarization threshold (default: 128).
    
    Returns:
        Binarized grayscale image with values {0, 255}.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def extract_frames_from_video(video_path: str, save_dir: str,
                               binarize: bool = False, threshold: int = 128):
    """
    Extract all frames from a video file and save as PNG images.
    
    Args:
        video_path: Path to input MP4 video.
        save_dir: Directory to save extracted frames.
        binarize: Whether to apply binarization (for GT videos).
        threshold: Binarization threshold.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path)
    
    frame_count = 0
    for _ in tqdm(range(total_frames), desc=f"  {video_name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        
        if binarize:
            frame = binarize_frame(frame, threshold)
        
        frame_name = f"frame_{frame_count:06d}.png"
        cv2.imwrite(os.path.join(save_dir, frame_name), frame)
        frame_count += 1
    
    cap.release()
    print(f"    Extracted {frame_count} frames -> {save_dir}")


def process_all_videos(video_root: str, anno_root: str, output_dir: str,
                       threshold: int = 128):
    """
    Process all videos in the dataset: extract frames from both source
    videos and annotation videos.
    
    Args:
        video_root: Root directory containing source videos (videos/).
        anno_root: Root directory containing annotation videos (annotations/).
        output_dir: Root directory for saving extracted frames.
        threshold: Binarization threshold for GT frames.
    """
    if not os.path.exists(video_root):
        raise FileNotFoundError(f"Video root not found: {video_root}")
    if not os.path.exists(anno_root):
        raise FileNotFoundError(f"Annotation root not found: {anno_root}")
    
    # Process each action category
    for action_name in sorted(os.listdir(video_root)):
        action_video_dir = os.path.join(video_root, action_name)
        action_anno_dir = os.path.join(anno_root, action_name)
        
        if not os.path.isdir(action_video_dir):
            continue
        if not os.path.exists(action_anno_dir):
            print(f"[WARNING] No annotation folder for action: {action_name}")
            continue
        
        print(f"\n[{action_name}]")
        
        # Process each video in the action category
        for video_file in sorted(os.listdir(action_video_dir)):
            if not video_file.endswith(".mp4"):
                continue
            
            video_name = os.path.splitext(video_file)[0]
            
            # Extract GT frames (with binarization)
            gt_video_path = os.path.join(action_anno_dir, video_file)
            gt_save_dir = os.path.join(output_dir, action_name, video_name, "GT_object_level")
            if os.path.exists(gt_video_path):
                extract_frames_from_video(gt_video_path, gt_save_dir,
                                          binarize=True, threshold=threshold)
            else:
                print(f"  [SKIP] GT video not found: {video_file}")
            
            # Extract source infrared frames (no binarization)
            src_video_path = os.path.join(action_video_dir, video_file)
            src_save_dir = os.path.join(output_dir, action_name, video_name, "Imgs")
            if os.path.exists(src_video_path):
                extract_frames_from_video(src_video_path, src_save_dir,
                                          binarize=False)
            else:
                print(f"  [SKIP] Source video not found: {video_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from IITR-VSOD video files."
    )
    parser.add_argument("--video_root", type=str, required=True,
                        help="Root directory of source videos (videos/).")
    parser.add_argument("--anno_root", type=str, required=True,
                        help="Root directory of annotation videos (annotations/).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted frames.")
    parser.add_argument("--threshold", type=int, default=128,
                        help="Binarization threshold for GT frames (default: 128).")
    
    args = parser.parse_args()
    
    process_all_videos(
        video_root=args.video_root,
        anno_root=args.anno_root,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
    
    print(f"\n[SUCCESS] All frames extracted to: {args.output_dir}")


if __name__ == "__main__":
    main()
