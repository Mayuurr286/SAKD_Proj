class MultiScaleFusionHead(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], out_channels=128):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(64, 2, 1)
        )
    def forward(self, features, target_size):
        resized = []
        for i, feat in enumerate(features):
            feat = self.proj[i](feat)
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized.append(feat)
        return self.fuse(torch.cat(resized, dim=1))

class FeatureAdapter(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Conv2d(s, t, 1) for s, t in zip(student_channels, teacher_channels)])
        self.stage_weights = nn.Parameter(torch.tensor([0.5, 0.75, 1.0, 1.25]))
    def forward(self, student_feats, teacher_feats):
        loss, w_sum = 0.0, 0.0
        for i, (adapter, s, t) in enumerate(zip(self.adapters, student_feats, teacher_feats)):
            s_aligned = adapter(s)
            t_resized = F.adaptive_avg_pool2d(t, s.shape[-2:])
            s_norm = F.normalize(s_aligned, dim=1)
            t_norm = F.normalize(t_resized.detach(), dim=1)
            cos_loss = (1 - (s_norm * t_norm).sum(dim=1).mean())
            w = F.softplus(self.stage_weights[i])
            loss += w * cos_loss
            w_sum += w
        return loss / (w_sum + 1e-6)

def compute_boundary_mask(mask, kernel_size=3):
    mask_float = mask.float().unsqueeze(1)
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask_float, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask_float, kernel_size, stride=1, padding=pad)
    return ((dilated - eroded).squeeze(1) > 0).float()

def boundary_loss(logits, targets, kernel_size=3):
    bm = compute_boundary_mask(targets, kernel_size)
    weight_map = 1.0 + 4.0 * bm
    return (F.cross_entropy(logits, targets, reduction='none') * weight_map).mean()

criterion_sup = nn.CrossEntropyLoss()

def kd_loss(student_logits, teacher_logits, T=TEMPERATURE, mask=None):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(s, t, reduction='none') * (T * T)
    if mask is not None:
        kl = kl * mask
    return kl.mean()

STUDENT_CHANNELS = [32, 64, 160, 256]
TEACHER_CHANNELS = [64, 128, 320, 512]

def find_image_mask_pairs(base_path, split_name='train'):
    images, masks = [], []
    for root, dirs, files in os.walk(base_path):
        if split_name not in root.lower():
            continue
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                full_path = os.path.join(root, file)
                if 'mask' in root.lower() or 'mask' in file.lower():
                    masks.append(full_path)
                else:
                    images.append(full_path)
    return sorted(images), sorted(masks)

train_imgs, train_masks = find_image_mask_pairs(DATA_ROOT, 'train')
val_imgs, val_masks = find_image_mask_pairs(DATA_ROOT, 'val')
test_imgs, test_masks = find_image_mask_pairs(DATA_ROOT, 'test')
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

all_train_pairs = list(zip(train_imgs, train_masks))
random.shuffle(all_train_pairs)
val_pairs = list(zip(val_imgs, val_masks))

from transformers import SegformerImageProcessor
image_processor = SegformerImageProcessor(
    do_resize=False,
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)

train_tf = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
])
unlabeled_tf = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
])
val_tf = A.Compose([A.CenterCrop(IMG_SIZE, IMG_SIZE)])

val_ds = WHULabeledDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs], image_processor, val_tf)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

test_loader = None
if len(test_imgs) > 0 and len(test_masks) > 0:
    test_ds = WHULabeledDataset(test_imgs, test_masks, image_processor, val_tf)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Load teacher ONCE (shared across all ratios)
teacher = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1", num_labels=2, output_hidden_states=True,
    id2label={0: "bg", 1: "building"}, label2id={"bg": 0, "building": 1},
    ignore_mismatched_sizes=True
).to(DEVICE)
if TEACHER_CKPT_PATH and os.path.exists(TEACHER_CKPT_PATH):
    ckpt = torch.load(TEACHER_CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        teacher.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        teacher.load_state_dict(ckpt, strict=False)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print("Teacher loaded and frozen")

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
