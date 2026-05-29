# IITR-VSOD Evaluation Tools

This repository provides companion scripts for the IITR-VSOD Infrared Video Salient Object Detection Dataset.

---

## Dataset Download

The IITR-VSOD dataset is publicly available at:

- **Zenodo**: [https://doi.org/10.5281/zenodo.20319967]

The download package includes:
- 1,470 infrared video sequences (`videos/`)
- Corresponding pixel-level saliency annotation videos (`annotations/`)
- Data split files: `train.txt`, `val.txt`, `test.txt`
- Video difficulty labels: `difficulty_labels.txt`

---

## Dataset Structure

IITR-VSOD-DATA/
├── videos/
│   └── ...
├── annotations/
│   └── ...
├── train.txt
├── val.txt
├── test.txt
└── difficulty_labels.txt

---

## Experimental Environment

This project has been tested with the following environment:

```
Python 3.12
CUDA 12.8
PyTorch 2.10.0
torchvision 0.25.0
```

Full dependencies are listed in `requirements.txt`:

```
numpy==2.2.6
opencv-python==4.13.0.90
pandas==2.3.2
scikit-image==0.25.2
scipy==1.15.3
torch==2.10.0
torchaudio==2.10.0
torchvision==0.25.0
torchmetrics==1.8.2
tensorboardX==2.6.5
tqdm==4.67.3
matplotlib==3.10.8
pillow==12.1.0
imageio==2.37.2
pytorch_ssim==0.1
```

Quick install:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Extract Frames from Videos

Use `extract_frames.py` to extract PNG frames from video files:

```bash
python extract_frames.py \
  --video_root /path/to/videos \
  --anno_root /path/to/annotations \
  --output_dir /path/to/output_frames \
  --threshold 128
```

Output structure:

```
output_frames/
├── carry_point_gun/
│   ├── carry_point_gun_1/
│   │   ├── GT_object_level/   # GT binary mask frames
│   │   └── Imgs/              # Source infrared frames
│   └── ...
└── ...
```

---

### 2. Evaluate Model Predictions

Use `evaluate.py` to compute evaluation metrics:

```bash
python evaluate.py \
  --pred_root /path/to/predictions \
  --gt_root /path/to/ground_truth \
  --dataset test \
  --output_dir ./results \
  --tag my_method_
```

Output files:
- `my_method_per_sample_results.csv`: per-video metrics
- `my_method_eval_log.txt`: dataset-level summary

---

### 3. Metrics Computation Module

`metrics.py` implements S-measure, max F-measure, max E-measure, MAE, and weighted F-measure.

---

## Code Files

Scripts included in this repository:

| File | Description |
|------|-------------|
| `extract_frames.py` | Extract frames from videos (infrared + GT) |
| `evaluate.py` | Model evaluation script, outputs CSV and log |
| `metrics.py` | Core evaluation metrics module |
| `requirements.txt` | Python dependencies |

Official implementations of benchmarked methods:

- **FSNet**: https://github.com/GewelsJI/FSNet
- **MMNet**: https://github.com/zhaoxing2022/MMN-VSOD
- **STDNet**: https://github.com/hellozhuo/stdnet/tree/main
