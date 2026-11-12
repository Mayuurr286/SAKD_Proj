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