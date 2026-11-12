import os
import random
import math
import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import albumentations as A
from tqdm import tqdm
import gc
import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = "/root/.cache/kagglehub/datasets/sengulgs/whu-building-dataset/versions/1/WHU"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS_PER_RATIO = 25     # 2 ratios × 20 epochs fits budget
LR = 6e-5
NUM_CLASSES = 2
NUM_WORKERS = 4

# Phase-2 loss weights
LAMBDA_KD = 0.5
LAMBDA_SCALE = 0.3
LAMBDA_BOUNDARY = 0.3
LAMBDA_UNSUP = 0.5
TEMPERATURE = 4.0
TAU = 0.6

# Multiple label ratios to test
LABEL_RATIOS = [0.1, 0.2]   # 10%, 20%

TEACHER_CKPT_PATHS = [
    "/root/best_segformer_mit_b1_epoch50_iou0.8283.pt",
    "/kaggle/input/models/parthivmandala/best-segformer-mit-b1-epoch50-iou0-8283-pt/pytorch/default/1/best_segformer_mit_b1_epoch50_iou0.8283.pt",
    "/kaggle/input/segformer-mit-b1-whu-building/pytorch/default/1/best_segformer_mit_b1_epoch50_iou0.8283.pt",
]
TEACHER_CKPT_PATH = None
for path in TEACHER_CKPT_PATHS:
    if os.path.exists(path):
        TEACHER_CKPT_PATH = path
        print(f"Found teacher checkpoint at: {path}")
        break
if TEACHER_CKPT_PATH is None:
    print("Teacher checkpoint not found")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def evaluate_full_metrics(model, fusion, loader, device=DEVICE):
    model.eval(); fusion.eval()
    TP = FP = FN = TN = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(pixel_values=x, output_hidden_states=True)
            features = [outputs.hidden_states[i] for i in range(4)]
            logits = fusion(features, target_size=y.shape[-2:])
            preds = torch.argmax(logits, dim=1)
            TP += ((preds == 1) & (y == 1)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
    eps = 1e-6
    return {
        'IoU': TP / (TP + FP + FN + eps),
        'Dice': (2*TP) / (2*TP + FP + FN + eps),
        'Precision': TP / (TP + FP + eps),
        'Recall': TP / (TP + FN + eps),
        'F1': 2*TP/(2*TP+FP+FN+eps),  # same as Dice for binary
        'Accuracy': (TP+TN) / (TP+TN+FP+FN+eps)
    }

def evaluate_iou(model, fusion, loader, device=DEVICE):
    return evaluate_full_metrics(model, fusion, loader, device)['IoU']

def evaluate_with_tta(model, fusion, loader, device=DEVICE):
    model.eval(); fusion.eval()
    TP = FP = FN = TN = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x, output_hidden_states=True)
            feats = [out.hidden_states[i] for i in range(4)]
            logits_orig = fusion(feats, target_size=y.shape[-2:])
            x_h = torch.flip(x, dims=[3])
            out_h = model(pixel_values=x_h, output_hidden_states=True)
            logits_h = torch.flip(fusion([out_h.hidden_states[i] for i in range(4)], target_size=y.shape[-2:]), dims=[3])
            x_v = torch.flip(x, dims=[2])
            out_v = model(pixel_values=x_v, output_hidden_states=True)
            logits_v = torch.flip(fusion([out_v.hidden_states[i] for i in range(4)], target_size=y.shape[-2:]), dims=[2])
            preds = torch.argmax((logits_orig + logits_h + logits_v) / 3.0, dim=1)
            TP += ((preds == 1) & (y == 1)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
    eps = 1e-6
    return {
        'IoU': TP / (TP + FP + FN + eps),
        'Dice': (2*TP) / (2*TP + FP + FN + eps),
        'Precision': TP / (TP + FP + eps),
        'Recall': TP / (TP + FN + eps),
        'F1': 2*TP/(2*TP+FP+FN+eps),
        'Accuracy': (TP+TN) / (TP+TN+FP+FN+eps)
    }

all_ratio_results = []  # collect results across all ratios

for LABEL_RATIO in LABEL_RATIOS:
    print(f"\n{'='*70}")
    print(f"  TRAINING WITH {LABEL_RATIO*100:.0f}% LABELED DATA")
    print(f"{'='*70}")

    # Split data for this ratio
    num_labeled = int(LABEL_RATIO * len(all_train_pairs))
    labeled_pairs = all_train_pairs[:num_labeled]
    unlabeled_pairs = all_train_pairs[num_labeled:]
    print(f"Labeled: {len(labeled_pairs)} | Unlabeled: {len(unlabeled_pairs)}")

    # Create loaders
    labeled_ds = WHULabeledDataset(
        [p[0] for p in labeled_pairs], [p[1] for p in labeled_pairs], image_processor, train_tf)
    unlabeled_ds = WHUUnlabeledDataset(
        [p[0] for p in unlabeled_pairs], image_processor, unlabeled_tf)
    labeled_loader = DataLoader(labeled_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Fresh student + fusion + adapter for each ratio
    student = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2, output_hidden_states=True,
        id2label={0: "bg", 1: "building"}, label2id={"bg": 0, "building": 1},
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    fusion_head = MultiScaleFusionHead(in_channels=STUDENT_CHANNELS, out_channels=128).to(DEVICE)
    feature_adapter = FeatureAdapter(student_channels=STUDENT_CHANNELS, teacher_channels=TEACHER_CHANNELS).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': student.parameters(), 'lr': LR},
        {'params': fusion_head.parameters(), 'lr': LR},
        {'params': feature_adapter.parameters(), 'lr': LR * 2},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_RATIO, eta_min=1e-6)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()

    # Training
    best_iou = 0.0
    patience = 10
    counter = 0
    history = []
    save_prefix = f"semi_{int(LABEL_RATIO*100)}pct"

    for epoch in range(1, EPOCHS_PER_RATIO + 1):
        student.train(); fusion_head.train(); feature_adapter.train()
        total_loss = total_sup = total_kd = total_scale = total_bnd = total_unsup = 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        num_steps = max(len(labeled_loader), len(unlabeled_loader))

        for step in tqdm(range(num_steps), desc=f"[{LABEL_RATIO*100:.0f}%] Epoch {epoch}"):
            optimizer.zero_grad()

            # ---- LABELED BATCH ----
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)
            x_l, y_l = x_l.to(DEVICE), y_l.to(DEVICE)

            with torch.cuda.amp.autocast():
                out_l = student(pixel_values=x_l, output_hidden_states=True)
                feats_l = [out_l.hidden_states[i] for i in range(4)]
                s_logits_l = fusion_head(feats_l, target_size=y_l.shape[-2:])
                loss_sup = criterion_sup(s_logits_l, y_l)
                loss_bnd = boundary_loss(s_logits_l, y_l)

                with torch.no_grad():
                    t_out_l = teacher(pixel_values=x_l, output_hidden_states=True)
                    t_logits_l = F.interpolate(t_out_l.logits, y_l.shape[-2:], mode="bilinear", align_corners=False)
                    t_probs_l = F.softmax(t_logits_l, dim=1)
                    conf_l, _ = torch.max(t_probs_l, dim=1, keepdim=True)
                    mask_l = (conf_l > TAU).float()

                loss_kd_l = kd_loss(s_logits_l, t_logits_l, TEMPERATURE, mask_l)
                loss_scale_l = feature_adapter(feats_l, t_out_l.hidden_states[:4])
                labeled_loss = loss_sup + LAMBDA_BOUNDARY * loss_bnd + LAMBDA_KD * loss_kd_l + LAMBDA_SCALE * loss_scale_l

            # ---- UNLABELED BATCH ----
            try:
                x_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_u = next(unlabeled_iter)
            x_u = x_u.to(DEVICE)

            with torch.cuda.amp.autocast():
                out_u = student(pixel_values=x_u, output_hidden_states=True)
                feats_u = [out_u.hidden_states[i] for i in range(4)]
                s_logits_u = fusion_head(feats_u, target_size=(IMG_SIZE, IMG_SIZE))

                with torch.no_grad():
                    t_out_u = teacher(pixel_values=x_u, output_hidden_states=True)
                    t_logits_u = F.interpolate(t_out_u.logits, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
                    t_probs_u = F.softmax(t_logits_u, dim=1)
                    conf_u, _ = torch.max(t_probs_u, dim=1, keepdim=True)
                    mask_u = (conf_u > TAU).float()

                loss_kd_u = kd_loss(s_logits_u, t_logits_u, TEMPERATURE, mask_u)
                loss_scale_u = feature_adapter(feats_u, t_out_u.hidden_states[:4])
                unlabeled_loss = LAMBDA_KD * loss_kd_u + LAMBDA_SCALE * loss_scale_u

            # ---- COMBINED ----
            with torch.cuda.amp.autocast():
                loss = labeled_loss + LAMBDA_UNSUP * unlabeled_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_kd += (loss_kd_l.item() + loss_kd_u.item())
            total_scale += (loss_scale_l.item() + loss_scale_u.item())
            total_bnd += loss_bnd.item()
            total_unsup += unlabeled_loss.item()

        scheduler.step()
        val_iou = evaluate_iou(student, fusion_head, val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch, 'loss': total_loss, 'loss_sup': total_sup,
            'loss_kd': total_kd, 'loss_scale': total_scale, 'loss_bnd': total_bnd,
            'loss_unsup': total_unsup, 'val_iou': val_iou, 'lr': current_lr
        })

        print(f"[{LABEL_RATIO*100:.0f}%] Epoch {epoch} | Loss {total_loss:.4f} | Val IoU {val_iou:.4f} | LR {current_lr:.2e}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(student.state_dict(), f"{save_prefix}_student.pth")
            torch.save(fusion_head.state_dict(), f"{save_prefix}_fusion.pth")
            print(f"  -> Saved best (IoU={best_iou:.4f})")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluate best checkpoint
    student.load_state_dict(torch.load(f"{save_prefix}_student.pth", map_location=DEVICE))
    fusion_head.load_state_dict(torch.load(f"{save_prefix}_fusion.pth", map_location=DEVICE))

    val_m = evaluate_full_metrics(student, fusion_head, val_loader)
    val_tta_m = evaluate_with_tta(student, fusion_head, val_loader)
    test_m = evaluate_full_metrics(student, fusion_head, test_loader) if test_loader else None
    test_tta_m = evaluate_with_tta(student, fusion_head, test_loader) if test_loader else None

    result = {
        'ratio': LABEL_RATIO, 'num_labeled': len(labeled_pairs), 'num_unlabeled': len(unlabeled_pairs),
        'best_val_iou': best_iou, 'val_iou': val_m['IoU'], 'val_tta_iou': val_tta_m['IoU'],
        'val_dice': val_m['Dice'], 'val_precision': val_m['Precision'], 'val_recall': val_m['Recall'],
    }
    if test_m:
        result.update({'test_iou': test_m['IoU'], 'test_tta_iou': test_tta_m['IoU']})
    all_ratio_results.append(result)

    print(f"\n[{LABEL_RATIO*100:.0f}%] Val IoU: {val_m['IoU']:.4f} | Val TTA: {val_tta_m['IoU']:.4f}")
    if test_m:
        print(f"[{LABEL_RATIO*100:.0f}%] Test IoU: {test_m['IoU']:.4f} | Test TTA: {test_tta_m['IoU']:.4f}")

    # Save per-ratio history
    pd.DataFrame(history).to_csv(f"{save_prefix}_history.csv", index=False)

    # Free memory before next ratio
    del student, fusion_head, feature_adapter, optimizer, scheduler, scaler
    del labeled_ds, unlabeled_ds, labeled_loader, unlabeled_loader
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory freed. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")
