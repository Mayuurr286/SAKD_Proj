class WHULabeledDataset(Dataset):
    def __init__(self, img_paths, mask_paths, image_processor, augment=None):
        self.imgs = img_paths
        self.masks = mask_paths
        self.processor = image_processor
        self.augment = augment
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        if self.augment:
            aug = self.augment(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc.pixel_values.squeeze(0)
        mask = torch.from_numpy(mask).long()
        return pixel_values, mask

class WHUUnlabeledDataset(Dataset):
    def __init__(self, img_paths, image_processor, augment=None):
        self.imgs = img_paths
        self.processor = image_processor
        self.augment = augment
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        if self.augment:
            aug = self.augment(image=img)
            img = aug["image"]
        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc.pixel_values.squeeze(0)
        return pixel_values

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