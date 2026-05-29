#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for IITR-VSOD dataset.

Computes per-video and dataset-level metrics (Smeasure, maxFm, meanFm, maxEm, meanEm,
MAE, wFmeasure, adpFm) and saves results to CSV and log files.

Usage:
    python evaluate.py --pred_root /path/to/predictions --gt_root /path/to/gt \\
                       --dataset test --output_dir ./results

Requirements:
    - OpenCV (cv2)
    - NumPy
    - metrics.py (in the same directory)
"""

import os
import sys
import time
import csv
import argparse

import cv2
import numpy as np

import metrics as M


def cal_metrics(pred_root: str, gt_root: str, dataset: str,
                output_dir: str = ".", tag: str = ""):
    """
    Evaluate predictions against ground truth for a given dataset split.

    Args:
        pred_root: Root directory containing prediction frames organized as
                   {pred_root}/{dataset}/{video_name}/{frame_name}.png
        gt_root:   Root directory containing GT frames organized as
                   {gt_root}/{dataset}/{video_name}/GT_object_level/{frame_name}.png
        dataset:   Dataset split name, e.g., 'test'.
        output_dir: Directory to save output CSV and log files.
        tag:       Optional tag prefix for output filenames.

    Outputs:
        {output_dir}/{tag}per_sample_results.csv  - per-video metrics
        {output_dir}/{tag}eval_log.txt            - dataset-level summary
    """
    gt_dataset = os.path.join(gt_root, dataset)
    pred_dataset = os.path.join(pred_root, dataset)

    per_sample_path = os.path.join(output_dir, f"{tag}per_sample_results.csv")
    log_path = os.path.join(output_dir, f"{tag}eval_log.txt")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset}")
    print(f"  GT:     {gt_dataset}")
    print(f"  Pred:   {pred_dataset}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    if not os.path.exists(gt_dataset):
        raise FileNotFoundError(f"GT path not found: {gt_dataset}")
    if not os.path.exists(pred_dataset):
        raise FileNotFoundError(f"Prediction path not found: {pred_dataset}")

    video_list = sorted(os.listdir(gt_dataset))
    total_videos = len(video_list)
    print(f"Found {total_videos} videos\n")

    # ---- Initialize per-sample CSV ----
    csv_file = open(per_sample_path, 'w', newline='', encoding='utf-8')
    fieldnames = ['video_id', 'Smeasure', 'MAE', 'maxFm', 'meanFm',
                  'maxEm', 'meanEm', 'wFmeasure', 'adpFm']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()

    # ---- Accumulators for dataset-level average ----
    sm_sum = wfm_sum = mae_sum = meanEm_sum = adpFm_sum = meanFm_sum = 0.0
    maxFm_curves, maxEm_curves = [], []
    processed_videos = 0
    start_time = time.perf_counter()

    for video_idx, video_name in enumerate(video_list):
        video_start = time.perf_counter()
        print(f"[{video_idx+1}/{total_videos}] {video_name}  ", end='', flush=True)

        fm = M.Fmeasure()
        wfm = M.WeightedFmeasure()
        sm = M.Smeasure()
        em = M.Emeasure()
        mae = M.MAE()

        gt_video_path = os.path.join(gt_dataset, video_name, "GT_object_level")
        if not os.path.exists(gt_video_path):
            print("SKIP (no GT_object_level)")
            continue

        gt_frames = sorted(os.listdir(gt_video_path))
        if len(gt_frames) > 2:
            gt_frames = gt_frames[1:-1]  # exclude first and last frame

        for gt_name in gt_frames:
            gt_path = os.path.join(gt_video_path, gt_name)
            pred_path = os.path.join(pred_dataset, video_name, gt_name)

            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is None or pred_img is None:
                continue

            fm.step(pred=pred_img, gt=gt_img)
            wfm.step(pred=pred_img, gt=gt_img)
            sm.step(pred=pred_img, gt=gt_img)
            em.step(pred=pred_img, gt=gt_img)
            mae.step(pred=pred_img, gt=gt_img)

        # ---- Compute per-video metrics ----
        fm_res = fm.get_results()['fm']
        wfm_res = wfm.get_results()['wfm']
        sm_res = sm.get_results()['sm']
        em_res = em.get_results()['em']
        mae_res = mae.get_results()['mae']

        meanFm = fm_res['curve'].mean()
        maxFm = fm_res['curve'].max()
        meanEm = 0 if em_res['curve'] is None else em_res['curve'].mean()
        maxEm = 0 if em_res['curve'] is None else em_res['curve'].max()
        adpFm = fm_res['adp']

        # Accumulate
        sm_sum += sm_res;  wfm_sum += wfm_res;  mae_sum += mae_res
        meanEm_sum += meanEm;  adpFm_sum += adpFm;  meanFm_sum += meanFm
        maxFm_curves.append(fm_res['curve'])
        maxEm_curves.append(em_res['curve'])
        processed_videos += 1

        # Write per-sample result immediately
        csv_writer.writerow({
            'video_id': video_name,
            'Smeasure': round(sm_res, 6),
            'MAE': round(mae_res, 6),
            'maxFm': round(maxFm, 6),
            'meanFm': round(meanFm, 6),
            'maxEm': round(maxEm, 6),
            'meanEm': round(meanEm, 6),
            'wFmeasure': round(wfm_res, 6),
            'adpFm': round(adpFm, 6),
        })
        csv_file.flush()

        print(f"SM:{sm_res:.4f} MAE:{mae_res:.4f} maxFm:{maxFm:.4f}  ({time.perf_counter()-video_start:.1f}s)")

    csv_file.close()

    # ---- Dataset-level summary ----
    if processed_videos == 0:
        raise RuntimeError("No videos were processed.")

    avg_sm = sm_sum / processed_videos
    avg_mae = mae_sum / processed_videos
    avg_meanFm = meanFm_sum / processed_videos
    avg_meanEm = meanEm_sum / processed_videos
    avg_wfm = wfm_sum / processed_videos
    avg_adpFm = adpFm_sum / processed_videos

    maxF = max(np.array(maxFm_curves).mean(0)) if maxFm_curves else 0.0
    maxE = max(np.array(maxEm_curves).mean(0)) if maxEm_curves else 0.0

    elapsed = time.perf_counter() - start_time

    print(f"\n{'='*60}")
    print(f"Done!  {processed_videos}/{total_videos} videos  ({elapsed:.1f}s)")
    print(f"  Sm: {avg_sm:.4f}  |  MAE: {avg_mae:.4f}  |  maxFm: {maxF:.4f}")
    print(f"  maxEm: {maxE:.4f}  |  meanFm: {avg_meanFm:.4f}")
    print(f"  Saved: {per_sample_path}")
    print(f"  Log:   {log_path}")
    print(f"{'='*60}\n")

    # Write summary log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"VSOD Evaluation Log | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pred: {pred_dataset}  |  GT: {gt_dataset}\n")
        f.write(f"Videos: {processed_videos}/{total_videos}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Smeasure: {round(avg_sm, 4)};  ")
        f.write(f"MAE: {round(avg_mae, 4)};  ")
        f.write(f"maxFm: {round(maxF, 4)};  ")
        f.write(f"maxEm: {round(maxE, 4)};  ")
        f.write(f"meanFm: {round(avg_meanFm, 4)};  ")
        f.write(f"meanEm: {round(avg_meanEm, 4)};  ")
        f.write(f"wFmeasure: {round(avg_wfm, 4)};  ")
        f.write(f"adpFm: {round(avg_adpFm, 4)};\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saliency predictions on IITR-VSOD dataset."
    )
    parser.add_argument("--pred_root", type=str, required=True,
                        help="Root directory of prediction frames.")
    parser.add_argument("--gt_root", type=str, required=True,
                        help="Root directory of ground-truth frames.")
    parser.add_argument("--dataset", type=str, default="test",
                        help="Dataset split name (default: test).")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save output files (default: ./results).")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional prefix for output filenames.")

    args = parser.parse_args()

    cal_metrics(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        dataset=args.dataset,
        output_dir=args.output_dir,
        tag=args.tag
    )


if __name__ == '__main__':
    main()
