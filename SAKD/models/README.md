# Model Checkpoints

This directory contains the lightweight student model weights generated during Phase 1 and Phase 2 of the semi-supervised ROI extraction pipeline. 

To comply with GitHub's 100 MB file size limit, the heavy Teacher model (156.8 MB) is hosted externally. Only the lightweight Student weights (14.25 MB) are required to run inference and deployment demonstrations.

## Model Registry

| Checkpoint File | Architecture | Training Phase | File Size | Validation IoU | Storage Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `student_best_earlystop.pth` | SegFormer MiT-B0 | Phase 1 (10% Labeled) | 14.25 MB | **0.8517** | Included in repo |
| `semi_10pct_student.pth` | SegFormer MiT-B0 Backbone | Phase 2 (10% Labeled) | ~14 MB | — | Included in repo |
| `semi_10pct_fusion.pth` | MultiScaleFusionHead | Phase 2 (10% Labeled) | ~2 MB | — | Included in repo |
| `best_segformer_mit_b1_...` | SegFormer MiT-B1 (Teacher) | Fully Supervised | 156.8 MB | **0.8283** | [Download from Kaggle](#) |

> **Note:** Replace the `[Download from Kaggle](#)` link above with the actual URL to your Kaggle Model or Google Drive where the Teacher model is hosted.

## Loading the Inference Model

To run a live demonstration of the model extracting building footprints from satellite imagery, you only need the Phase 1 Student weights.

The provided `src/infer.py` script automatically loads these weights:

```python
import torch
from transformers import SegformerForSemanticSegmentation

# 1. Initialize the lightweight MiT-B0 architecture
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True
)

# 2. Load the trained Phase 1 weights from this directory
checkpoint = torch.load("student_best_earlystop.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()
```

## Why is the Student Model Better for Deployment?
Despite being trained on only **10% labeled data**, the MiT-B0 Student model achieves a higher Validation IoU (0.8517) than the fully supervised MiT-B1 Teacher model (0.8283) while being **11x smaller** (14.25 MB vs 156.8 MB). This drastic reduction in storage and memory makes it highly suitable for real-time edge computing and UAV deployments.