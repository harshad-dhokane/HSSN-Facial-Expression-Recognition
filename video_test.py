"""
HSSN Video Emotion Detection — Student Expression Analysis
============================================================
Loads a video file and runs the trained Hierarchical Spectral-Spatial
Network (HSSN) model on each frame to detect and classify facial
expressions of students.

Usage:
    python video_test.py                  # Opens file dialog
    python video_test.py path/to/video.mp4  # Direct file path

Controls:
    q / ESC       — Quit
    SPACE         — Play / Pause
    s             — Save screenshot of current frame
    t             — Toggle TTA on/off
    f             — Toggle probability bars
    r             — Toggle recording (save annotated video)
    d             — Toggle emotion dashboard
    RIGHT / .     — Next frame (when paused)
    LEFT  / ,     — Previous frame (when paused)
    UP            — Increase speed (0.25x steps, max 4x)
    DOWN          — Decrease speed (0.25x steps, min 0.25x)
    HOME          — Jump to start
    END           — Jump to end (last 5 seconds)
    1-9           — Jump to 10%-90% of video
    Click on progress bar — Seek to that position
"""

import sys
import os

# Fix Qt backend on Wayland/Gnome — force X11 (xcb) so windows work properly
os.environ["QT_QPA_PLATFORM"] = "xcb"

import json
import time
import argparse
import traceback
import threading
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / 'outputs' / 'exports' / 'hssn_full_model.pth'
CHECKPOINT_PATH = BASE_DIR / 'outputs' / 'checkpoints' / 'best_model.pth'
CLASS_MAPPING_PATH = BASE_DIR / 'outputs' / 'exports' / 'class_mapping.json'
CONFIG_PATH = BASE_DIR / 'outputs' / 'logs' / 'experiment_config.json'
YUNET_MODEL_PATH = str(BASE_DIR / 'face_detection_yunet_2023mar.onnx')
SCREENSHOT_DIR = BASE_DIR / 'outputs' / 'screenshots'
VIDEO_OUTPUT_DIR = BASE_DIR / 'outputs' / 'annotated_videos'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load experiment config
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Load class mapping
with open(CLASS_MAPPING_PATH, 'r') as f:
    CLASS_MAPPING = json.load(f)
    CLASS_NAMES = [CLASS_MAPPING[str(i)] for i in range(len(CLASS_MAPPING))]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = CONFIG.get('image_size', 224)
EMBEDDING_DIM = CONFIG.get('embedding_dim', 256)

# Normalization (from training)
NORM_MEAN = (0.5327, 0.4264, 0.3771)
NORM_STD = (0.2934, 0.2655, 0.2613)

# Emotion colors (BGR for OpenCV)
EMOTION_COLORS = {
    'Neutral':  (200, 200, 200),
    'Happy':    (0, 230, 118),
    'Sad':      (255, 152, 0),
    'Surprise': (0, 191, 255),
    'Fear':     (147, 112, 219),
    'Disgust':  (0, 100, 0),
    'Anger':    (0, 0, 255),
    'Contempt': (180, 180, 0),
}

# Dedicated sidebar width so dashboard is rendered outside the video feed
SIDEBAR_WIDTH = 250

# Emoji/icons for emotions (text fallback)
EMOTION_ICONS = {
    'Neutral':  '😐',
    'Happy':    '😊',
    'Sad':      '😢',
    'Surprise': '😲',
    'Fear':     '😨',
    'Disgust':  '🤢',
    'Anger':    '😡',
    'Contempt': '😏',
}

# YuNet outputs 5 facial landmarks per face:
#  0: right eye, 1: left eye, 2: nose tip, 3: right mouth, 4: left mouth
YUNET_CONF_THRESHOLD = 0.5
YUNET_NMS_THRESHOLD = 0.3


# ──────────────────────────────────────────────
#  MODEL ARCHITECTURE (must match training)
# ──────────────────────────────────────────────

class SpatialPath(nn.Module):
    def __init__(self, channels, dilation_init=1):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation_init,
                                  dilation=dilation_init, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.dw_conv(x)))
        out = self.act(self.bn2(self.pw_conv(out)))
        return out


class SpectralPath(nn.Module):
    def __init__(self, channels, spatial_size):
        super().__init__()
        mask_h = min(spatial_size, 16)
        mask_w = min(spatial_size // 2 + 1, 9)
        self.freq_mask = nn.Parameter(torch.ones(1, channels, mask_h, mask_w) * 0.5)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        x_freq = torch.fft.rfft2(x_f32, norm='ortho')
        mask = torch.sigmoid(self.freq_mask.float())
        if mask.shape[2:] != x_freq.shape[2:]:
            mask = F.interpolate(mask, size=x_freq.shape[2:], mode='bilinear', align_corners=False)
        x_filtered = x_freq * mask
        x_spatial = torch.fft.irfft2(x_filtered, s=x_f32.shape[2:], norm='ortho')
        return self.bn(x_spatial.to(orig_dtype))


class ChannelAttentionGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, spectral_feat):
        combined = torch.cat([
            F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1),
            F.adaptive_avg_pool2d(spectral_feat, 1).flatten(1)
        ], dim=1)
        alpha = self.gate(combined).unsqueeze(-1).unsqueeze(-1)
        return alpha * spatial_feat + (1 - alpha) * spectral_feat


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class DualPathBlock(nn.Module):
    def __init__(self, channels, spatial_size, dilation=1, drop_path_rate=0.0):
        super().__init__()
        self.spatial_path = SpatialPath(channels, dilation)
        self.spectral_path = SpectralPath(channels, spatial_size)
        self.gate = ChannelAttentionGate(channels)
        self.se = SEBlock(channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        spatial_out = self.spatial_path(x)
        spectral_out = self.spectral_path(x)
        fused = self.gate(spatial_out, spectral_out)
        fused = self.se(fused)
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1).item() < self.drop_path_rate:
                return x
        return x + self.residual_scale * fused


class AntiAliasDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        self.register_buffer('blur_kernel', blur_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.conv2d(x, self.blur_kernel, padding=1, groups=self.in_channels)
        x = x[:, :, ::2, ::2]
        x = self.bn(self.conv(x))
        return F.gelu(x)


class CrossStageRefinement(nn.Module):
    def __init__(self, channels, prev_channels, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.q_proj = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.k_proj = nn.Conv2d(prev_channels, channels // 4, 1, bias=False)
        self.v_proj = nn.Conv2d(prev_channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = (channels // 4) ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, current, previous):
        b, c, h, w = current.shape
        prev_up = F.interpolate(previous, size=(h, w), mode='bilinear', align_corners=False)
        ps = min(self.pool_size, h, w)
        q_down = F.adaptive_avg_pool2d(self.q_proj(current), ps)
        k_down = F.adaptive_avg_pool2d(self.k_proj(prev_up), ps)
        v_down = F.adaptive_avg_pool2d(self.v_proj(prev_up), ps)
        q = q_down.flatten(2)
        k = k_down.flatten(2)
        v = v_down.flatten(2)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        refined = torch.bmm(v, attn.transpose(1, 2)).view(b, c, ps, ps)
        refined = F.interpolate(refined, size=(h, w), mode='bilinear', align_corners=False)
        refined = self.out_proj(refined)
        return current + self.gamma * refined


class MultiScalePoolingHead(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 4)):
        super().__init__()
        self.pool_sizes = pool_sizes
        total_tokens = sum(p * p for p in pool_sizes)
        self.proj = nn.Sequential(
            nn.Linear(in_channels * total_tokens, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        pooled = []
        for ps in self.pool_sizes:
            p = F.adaptive_avg_pool2d(x, ps)
            pooled.append(p.flatten(1))
        cat = torch.cat(pooled, dim=1)
        return self.proj(cat)


class SoftplusGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.ones(1, dim) * 0.5)

    def forward(self, x):
        return F.softplus(x) * torch.sigmoid(self.gate_weight.to(x.device) * x)


class EmbeddingHead(nn.Module):
    def __init__(self, in_dim=512, embed_dim=256):
        super().__init__()
        mid_dim = (in_dim + embed_dim) // 2
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.act = SoftplusGate(mid_dim)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(mid_dim, embed_dim)

    def forward(self, x):
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class StochasticNoiseInjection(nn.Module):
    def __init__(self, gaussian_sigma_range=(0.01, 0.08), sp_density_range=(0.001, 0.02)):
        super().__init__()

    def forward(self, x):
        return x


class IlluminationPerturbation(nn.Module):
    def __init__(self, gamma_range=(0.7, 1.4), channel_scale_range=(0.85, 1.15)):
        super().__init__()

    def forward(self, x):
        return x


class HierarchicalSpectralSpatialNet(nn.Module):
    def __init__(self, num_classes=8, embed_dim=256, input_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.use_checkpointing = False

        stage_channels = CONFIG.get('stage_channels', [64, 128, 256, 512])
        blocks_per_stage = CONFIG.get('blocks_per_stage', 3)
        stochastic_depth_rate = CONFIG.get('stochastic_depth_rate', 0.1)
        n_stages = len(stage_channels)

        stem_out_size = input_size // 2
        spatial_sizes = [stem_out_size // (2 ** i) for i in range(n_stages)]
        total_blocks = n_stages * blocks_per_stage

        self.stem = nn.Sequential(
            nn.Conv2d(3, stage_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.GELU(),
        )

        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.cross_refinements = nn.ModuleList()
        self.noise_aug = StochasticNoiseInjection()
        self.illum_aug = IlluminationPerturbation()

        block_counter = 0
        for stage_idx in range(n_stages):
            ch = stage_channels[stage_idx]
            sp_size = spatial_sizes[stage_idx]
            blocks = nn.ModuleList()
            for block_idx in range(blocks_per_stage):
                dilation = 1 + block_idx
                drop_rate = stochastic_depth_rate * block_counter / max(total_blocks - 1, 1)
                blocks.append(DualPathBlock(ch, sp_size, dilation, drop_path_rate=drop_rate))
                block_counter += 1
            self.stages.append(blocks)

            if stage_idx < n_stages - 1:
                self.downsamplers.append(
                    AntiAliasDownsample(stage_channels[stage_idx], stage_channels[stage_idx + 1])
                )
            if stage_idx > 0:
                self.cross_refinements.append(
                    CrossStageRefinement(stage_channels[stage_idx], stage_channels[stage_idx - 1])
                )

        self.multi_pool = MultiScalePoolingHead(stage_channels[-1], pool_sizes=(1, 2, 4))
        self.embedding_head = EmbeddingHead(stage_channels[-1], embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        stage_features = []
        cross_ref_idx = 0

        for stage_idx, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)
            if stage_idx > 0 and cross_ref_idx < len(self.cross_refinements):
                x = self.cross_refinements[cross_ref_idx](x, stage_features[-1])
                cross_ref_idx += 1
            stage_features.append(x)
            if stage_idx < len(self.downsamplers):
                x = self.downsamplers[stage_idx](x)

        pooled = self.multi_pool(x)
        embedding = self.embedding_head(pooled)
        logits = self.classifier(embedding)

        if return_features:
            return logits, embedding, stage_features
        return logits, embedding


# ──────────────────────────────────────────────
#  FACE DETECTION (YuNet) & ALIGNMENT
# ──────────────────────────────────────────────

def create_face_detector(frame_w, frame_h, det_scale=0.5):
    """Create OpenCV YuNet face detector on a downscaled resolution for speed."""
    det_w = int(frame_w * det_scale)
    det_h = int(frame_h * det_scale)
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, '', (det_w, det_h),
        YUNET_CONF_THRESHOLD, YUNET_NMS_THRESHOLD, 5000
    )
    return detector, det_scale


def detect_faces_yunet(detector, frame, det_scale):
    """
    Detect faces using YuNet on a downscaled frame, then scale coords back.
    Returns list of dicts with 'bbox' and 'landmarks'.
    """
    if det_scale < 1.0:
        small = cv2.resize(frame, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_LINEAR)
    else:
        small = frame
    _, raw_faces = detector.detect(small)
    if raw_faces is None:
        return []

    inv = 1.0 / det_scale
    faces = []
    for f in raw_faces:
        x1, y1, bw, bh = int(f[0] * inv), int(f[1] * inv), int(f[2] * inv), int(f[3] * inv)
        x2, y2 = x1 + bw, y1 + bh
        right_eye = (f[4] * inv, f[5] * inv)
        left_eye = (f[6] * inv, f[7] * inv)
        nose = (f[8] * inv, f[9] * inv)
        score = f[14]
        faces.append({
            'bbox': (x1, y1, x2, y2),
            'right_eye': right_eye,
            'left_eye': left_eye,
            'nose': nose,
            'score': score,
        })
    return faces


def align_face_yunet(img_np, face_info, margin=0.3, target_size=224):
    """Align face using YuNet eye landmarks — matches training/webcam pipeline.
    
    Same algorithm as webcam_test.py align_face():
      1. Rotate around eye center (INTER_CUBIC)
      2. Transform bbox coords into rotated space
      3. Crop face with margin, make square
      4. Resize with LANCZOS4
    
    Uses a generous ROI to avoid warpAffine on the entire frame for speed.
    """
    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = face_info['bbox']
    bw, bh = x2 - x1, y2 - y1

    # Eye centers from YuNet landmarks
    left_eye = np.array(face_info['left_eye'], dtype=np.float64)
    right_eye = np.array(face_info['right_eye'], dtype=np.float64)

    # Rotation angle from eye line
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Use a generous ROI (3x face size) to avoid warping the entire frame
    face_size = max(bw, bh)
    roi_expand = face_size * 2.0  # 2x padding on each side
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    roi_x1 = max(0, int(cx - roi_expand))
    roi_y1 = max(0, int(cy - roi_expand))
    roi_x2 = min(w, int(cx + roi_expand))
    roi_y2 = min(h, int(cy + roi_expand))
    roi = img_np[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None
    rh, rw = roi.shape[:2]

    # Adjust coordinates to ROI space
    roi_left_eye = (left_eye[0] - roi_x1, left_eye[1] - roi_y1)
    roi_right_eye = (right_eye[0] - roi_x1, right_eye[1] - roi_y1)
    roi_eye_center = ((roi_left_eye[0] + roi_right_eye[0]) / 2,
                      (roi_left_eye[1] + roi_right_eye[1]) / 2)

    # Rotate the ROI around eye center (like webcam_test.py rotates full frame)
    M = cv2.getRotationMatrix2D(roi_eye_center, angle, scale=1.0)
    rotated = cv2.warpAffine(roi, M, (rw, rh),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # Transform the bbox corners into rotated ROI space
    roi_x1f, roi_y1f = float(x1 - roi_x1), float(y1 - roi_y1)
    roi_x2f, roi_y2f = float(x2 - roi_x1), float(y2 - roi_y1)
    corners = np.array([
        [roi_x1f, roi_y1f],
        [roi_x2f, roi_y1f],
        [roi_x2f, roi_y2f],
        [roi_x1f, roi_y2f],
    ], dtype=np.float64)
    ones = np.ones((4, 1))
    corners_h = np.hstack([corners, ones])
    rotated_corners = (M @ corners_h.T).T

    # Bounding box of rotated corners
    rx_min = rotated_corners[:, 0].min()
    ry_min = rotated_corners[:, 1].min()
    rx_max = rotated_corners[:, 0].max()
    ry_max = rotated_corners[:, 1].max()
    rbw = rx_max - rx_min
    rbh = ry_max - ry_min

    # Add margin (same as webcam align_face)
    cx1 = max(0, int(rx_min - margin * rbw))
    cy1 = max(0, int(ry_min - margin * rbh))
    cx2 = min(rw, int(rx_max + margin * rbw))
    cy2 = min(rh, int(ry_max + margin * rbh))

    # Make square crop centered on face
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    side = max(crop_w, crop_h)
    ccx = (cx1 + cx2) // 2
    ccy = (cy1 + cy2) // 2
    sx1 = max(0, ccx - side // 2)
    sy1 = max(0, ccy - side // 2)
    sx2 = min(rw, sx1 + side)
    sy2 = min(rh, sy1 + side)

    face_crop = rotated[sy1:sy2, sx1:sx2]
    if face_crop.size == 0:
        face_crop = roi
    return cv2.resize(face_crop, (target_size, target_size),
                      interpolation=cv2.INTER_LANCZOS4)


def histogram_equalize_rgb(img_np):
    result = np.zeros_like(img_np)
    for c in range(3):
        channel = img_np[:, :, c]
        hist, bins = np.histogram(channel.ravel(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-8)
        result[:, :, c] = cdf_normalized[channel]
    return result.astype(np.uint8)


# ──────────────────────────────────────────────
#  INFERENCE FUNCTIONS
# ──────────────────────────────────────────────

def preprocess_face(face_np, target_size=224):
    """Convert aligned face numpy array to model input tensor.
    Must match training pipeline exactly:
      1. Per-channel RGB histogram equalization (same as training preprocessing)
      2. PIL Image → TF.to_tensor  (scales [0,255] → [0,1])
      3. TF.normalize with training NORM_MEAN/NORM_STD
    """
    face_eq = histogram_equalize_rgb(face_np)
    face_pil = Image.fromarray(face_eq)
    tensor = TF.to_tensor(face_pil)
    tensor = TF.normalize(tensor, NORM_MEAN, NORM_STD)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict_single(model, tensor, device):
    with torch.no_grad():
        logits, embedding = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0], embedding.cpu().numpy()[0]


def predict_batch(model, tensors_list, device):
    """Process all faces in a single batched GPU forward pass for speed."""
    if not tensors_list:
        return []
    batch = torch.cat(tensors_list, dim=0).to(device)
    with torch.no_grad():
        logits, _ = model(batch)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


def predict_tta(model, face_np, device, target_size=224):
    tta_views = []
    tta_views.append(preprocess_face(face_np, target_size))
    flipped = np.fliplr(face_np).copy()
    tta_views.append(preprocess_face(flipped, target_size))
    for angle in [-5, 5, -10, 10]:
        h, w = face_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(face_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        tta_views.append(preprocess_face(rotated, target_size))
    all_probs = []
    for view in tta_views:
        with torch.no_grad():
            logits, _ = model(view.to(device))
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    return np.mean(all_probs, axis=0)


# ──────────────────────────────────────────────
#  DISPLAY HELPERS
# ──────────────────────────────────────────────

def format_time(seconds):
    """Format seconds to MM:SS or HH:MM:SS."""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    else:
        m = seconds // 60
        s = seconds % 60
        return f"{m}:{s:02d}"


def _clip_region(frame, x1, y1, x2, y2):
    """Clip a rectangle to frame bounds and return (x1, y1, x2, y2) or None."""
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def blur_region(frame, x1, y1, x2, y2, blur_ksize=11, blur_mix=0.72):
    """Apply a light gaussian blur only within a small ROI."""
    rect = _clip_region(frame, x1, y1, x2, y2)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return rect

    k = max(3, int(blur_ksize))
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1:y2, x1:x2] = cv2.addWeighted(blurred, blur_mix, roi, 1.0 - blur_mix, 0)
    return rect


def tint_region(frame, x1, y1, x2, y2, color=(20, 20, 20), alpha=0.25):
    """Blend a color tint onto a small ROI."""
    rect = _clip_region(frame, x1, y1, x2, y2)
    if rect is None:
        return
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    tint = np.full_like(roi, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0)


def draw_prediction(frame, bbox, emotion, confidence, all_probs, face_id=0, show_bars=True):
    """Draw bounding box, emotion label, and probability bars for each face."""
    x1, y1, x2, y2 = bbox
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y2 = max(0, min(int(y2), frame_h - 1))
    if x2 <= x1 or y2 <= y1:
        return frame

    # Face bounding box (slightly thinner)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Face ID badge (smaller + non-opaque)
    badge_r = 9
    badge_cx = min(frame_w - badge_r - 1, x1 + badge_r + 2)
    badge_cy = min(frame_h - badge_r - 1, y1 + badge_r + 2)
    blur_region(frame,
                badge_cx - badge_r - 1, badge_cy - badge_r - 1,
                badge_cx + badge_r + 2, badge_cy + badge_r + 2,
                blur_ksize=9, blur_mix=0.6)
    tint_region(frame,
                badge_cx - badge_r - 1, badge_cy - badge_r - 1,
                badge_cx + badge_r + 2, badge_cy + badge_r + 2,
                color=(30, 30, 30), alpha=0.2)
    cv2.circle(frame, (badge_cx, badge_cy), badge_r, color, 1)
    cv2.putText(frame, str(face_id + 1), (badge_cx - 4, badge_cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)

    # Emotion label above box (smaller + blurred background)
    label = f"{emotion} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad_x, pad_y = 4, 3
    label_w = tw + (2 * pad_x)
    label_h = th + baseline + (2 * pad_y)

    label_x1 = max(0, min(x1, frame_w - label_w - 1))
    preferred_y2 = y1 - 2
    if preferred_y2 - label_h >= 0:
        label_y1 = preferred_y2 - label_h
        label_y2 = preferred_y2
    else:
        label_y1 = min(max(0, y1 + 2), max(0, frame_h - label_h - 1))
        label_y2 = label_y1 + label_h

    blur_region(frame, label_x1, label_y1, label_x1 + label_w, label_y2, blur_ksize=11, blur_mix=0.72)
    tint_region(frame, label_x1, label_y1, label_x1 + label_w, label_y2, color=(20, 20, 20), alpha=0.25)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + label_w, label_y2), color, 1)
    cv2.putText(frame, label, (label_x1 + pad_x, label_y2 - baseline - pad_y),
                font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)

    # Probability bars below bounding box
    if show_bars and all_probs is not None:
        bar_height = 10
        bar_gap = 2
        top_k = 3

        bar_max_width = min(110, max(65, frame_w - 8))
        bar_x = max(0, min(x1, frame_w - bar_max_width - 4))
        bar_block_h = top_k * (bar_height + bar_gap) - bar_gap
        bar_y_start = min(y2 + 4, frame_h - bar_block_h - 56)
        bar_y_start = max(0, bar_y_start)

        sorted_indices = np.argsort(all_probs)[::-1][:top_k]
        for i, idx in enumerate(sorted_indices):
            y = bar_y_start + i * (bar_height + bar_gap)
            if y + bar_height > frame_h - 55:
                break
            name = CLASS_NAMES[idx]
            prob = all_probs[idx]
            bar_color = EMOTION_COLORS.get(name, (200, 200, 200))

            blur_region(frame, bar_x, y, bar_x + bar_max_width, y + bar_height, blur_ksize=9, blur_mix=0.62)
            tint_region(frame, bar_x, y, bar_x + bar_max_width, y + bar_height, color=(24, 24, 24), alpha=0.26)
            bar_width = int(prob * bar_max_width)
            if bar_width > 0:
                tint_region(frame, bar_x, y, bar_x + bar_width, y + bar_height, color=bar_color, alpha=0.45)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height), (85, 85, 85), 1)
            cv2.putText(frame, f"{name[:7]} {prob:.0%}", (bar_x + 2, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (235, 235, 235), 1, cv2.LINE_AA)

    return frame


def draw_progress_bar(frame, current_frame, total_frames, current_time, total_time,
                      is_paused, playback_speed, is_recording):
    """Draw video progress bar and playback controls at the bottom."""
    h, w = frame.shape[:2]
    bar_area_height = 50
    bar_y = h - bar_area_height

    # Dark background for the control area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Progress bar dimensions
    bar_margin = 15
    bar_left = bar_margin + 55   # space for time text
    bar_right = w - bar_margin - 55
    bar_top = bar_y + 8
    bar_bottom = bar_top + 10
    bar_width = bar_right - bar_left

    # Progress fraction
    progress = current_frame / max(total_frames - 1, 1)

    # Bar background (dark gray track)
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (60, 60, 60), -1)

    # Buffered/played portion (accent color)
    played_x = bar_left + int(progress * bar_width)
    cv2.rectangle(frame, (bar_left, bar_top), (played_x, bar_bottom), (0, 160, 255), -1)

    # Playhead circle
    cv2.circle(frame, (played_x, (bar_top + bar_bottom) // 2), 7, (255, 255, 255), -1)
    cv2.circle(frame, (played_x, (bar_top + bar_bottom) // 2), 7, (0, 160, 255), 2)

    # Current time (left of bar)
    time_text = format_time(current_time)
    cv2.putText(frame, time_text, (bar_margin, bar_bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # Total time (right of bar)
    total_text = format_time(total_time)
    cv2.putText(frame, total_text, (bar_right + 5, bar_bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # Bottom row: status info
    info_y = bar_y + 38

    # Play/Pause indicator
    if is_paused:
        # Pause icon (two bars)
        cv2.rectangle(frame, (bar_margin, info_y - 10), (bar_margin + 4, info_y + 2), (255, 255, 255), -1)
        cv2.rectangle(frame, (bar_margin + 8, info_y - 10), (bar_margin + 12, info_y + 2), (255, 255, 255), -1)
        status = "PAUSED"
    else:
        # Play icon (triangle)
        pts = np.array([
            [bar_margin, info_y - 10],
            [bar_margin, info_y + 2],
            [bar_margin + 12, info_y - 4]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (0, 230, 118))
        status = "PLAYING"

    # Status + speed + frame info
    speed_text = f"{playback_speed:.2f}x" if playback_speed != 1.0 else "1x"
    frame_text = f"Frame {current_frame}/{total_frames}"
    info = f"  {status}  |  Speed: {speed_text}  |  {frame_text}"

    if is_recording:
        # Recording indicator (red dot)
        cv2.circle(frame, (w - bar_margin - 8, info_y - 4), 6, (0, 0, 255), -1)
        info += "  |  REC"

    cv2.putText(frame, info, (bar_margin + 18, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame, bar_left, bar_right, bar_top, bar_bottom


def draw_dashboard(frame, emotion_counts, total_analyzed, num_faces):
    """Draw an emotion distribution dashboard inside a sidebar frame."""
    h, w = frame.shape[:2]

    panel_margin = 10
    panel_x = panel_margin
    panel_y = 14
    panel_w = max(130, w - (2 * panel_margin))
    panel_h_target = 30 + NUM_CLASSES * 24 + 38
    panel_h = min(panel_h_target, max(60, h - panel_y - 14))
    panel_bottom = panel_y + panel_h

    blur_region(frame, panel_x, panel_y, panel_x + panel_w, panel_bottom, blur_ksize=21, blur_mix=0.35)
    tint_region(frame, panel_x, panel_y, panel_x + panel_w, panel_bottom, color=(18, 18, 18), alpha=0.76)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_bottom), (80, 80, 80), 1)

    cv2.putText(frame, "Emotion Summary", (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.line(frame, (panel_x + 6, panel_y + 28), (panel_x + panel_w - 6, panel_y + 28), (80, 80, 80), 1)

    bar_max_w = max(36, panel_w - 94)
    max_count = max(emotion_counts.values()) if emotion_counts and max(emotion_counts.values()) > 0 else 1

    for i, name in enumerate(CLASS_NAMES):
        y = panel_y + 44 + i * 24
        if y + 8 > panel_bottom - 24:
            break
        count = emotion_counts.get(name, 0)
        pct = count / max(total_analyzed, 1) * 100

        color = EMOTION_COLORS.get(name, (200, 200, 200))
        cv2.putText(frame, name[:7], (panel_x + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)

        bar_x = panel_x + 62
        bar_y1, bar_y2 = y - 7, y + 5
        tint_region(frame, bar_x, bar_y1, bar_x + bar_max_w, bar_y2, color=(36, 36, 36), alpha=0.35)
        bar_w = int(count / max_count * bar_max_w) if max_count > 0 else 0
        if bar_w > 0:
            tint_region(frame, bar_x, bar_y1, bar_x + bar_w, bar_y2, color=color, alpha=0.60)
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + bar_max_w, bar_y2), (72, 72, 72), 1)
        cv2.putText(frame, f"{pct:.0f}%", (bar_x + bar_max_w + 5, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

    summary_y = max(panel_y + 42, panel_bottom - 24)
    cv2.putText(frame, f"Faces now: {num_faces}", (panel_x + 8, summary_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (165, 165, 165), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total detections: {total_analyzed}", (panel_x + 8, min(h - 8, summary_y + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (145, 145, 145), 1, cv2.LINE_AA)

    return frame


def compose_output_frame(video_frame, emotion_counts, total_analyzed, num_faces, show_dashboard):
    """Attach a fixed right sidebar so dashboard never overlays the video feed."""
    h = video_frame.shape[0]
    sidebar = np.full((h, SIDEBAR_WIDTH, 3), 18, dtype=np.uint8)
    cv2.line(sidebar, (0, 0), (0, h - 1), (70, 70, 70), 1)

    if show_dashboard:
        draw_dashboard(sidebar, emotion_counts, total_analyzed, num_faces)
    else:
        cv2.putText(sidebar, "Dashboard hidden", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "Press 'd' to show", (14, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1, cv2.LINE_AA)

    return np.hstack((video_frame, sidebar))


def draw_controls_help(frame):
    """Draw controls hint at the top."""
    h, w = frame.shape[:2]
    help_text = "SPACE:Play/Pause  Arrows:Seek/Speed  s:Screenshot  t:TTA  f:Bars  r:Record  d:Dashboard  q:Quit"
    cv2.putText(frame, help_text, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    return frame


# ──────────────────────────────────────────────
#  VIDEO FILE SELECTION
# ──────────────────────────────────────────────

def select_video_file():
    """Open a file dialog to select a video file, with fallback to terminal input."""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg']

    # Try tkinter file dialog first
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        filetypes = [
            ("Video files", " ".join(f"*{ext}" for ext in video_extensions)),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MKV files", "*.mkv"),
            ("MOV files", "*.mov"),
            ("All files", "*.*"),
        ]

        filepath = filedialog.askopenfilename(
            title="Select Video File for Expression Analysis",
            filetypes=filetypes,
            initialdir=str(Path.home())
        )
        root.destroy()

        if filepath:
            return filepath
        else:
            print("  No file selected.")
            return None

    except (ImportError, Exception) as e:
        print(f"\n  File dialog unavailable ({e})")
        print("  Please enter the video file path manually:")
        filepath = input("  > ").strip().strip('"').strip("'")
        if filepath and Path(filepath).exists():
            return filepath
        elif filepath:
            print(f"  File not found: {filepath}")
            return None
        else:
            print("  No path entered.")
            return None


# ──────────────────────────────────────────────
#  MOUSE CALLBACK FOR SEEKING
# ──────────────────────────────────────────────

class VideoState:
    """Shared state for mouse callback and main loop."""
    def __init__(self):
        self.seek_requested = False
        self.seek_fraction = 0.0
        self.bar_left = 0
        self.bar_right = 0
        self.bar_top = 0
        self.bar_bottom = 0
        self.frame_h = 0


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks on the progress bar to seek."""
    state = param
    bar_area_y = state.frame_h - 50

    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
        # Check if click is in the progress bar area
        if (state.bar_left <= x <= state.bar_right and
                bar_area_y <= y <= bar_area_y + 25):
            fraction = (x - state.bar_left) / max(state.bar_right - state.bar_left, 1)
            fraction = max(0.0, min(1.0, fraction))
            state.seek_fraction = fraction
            state.seek_requested = True


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HSSN Video Emotion Detection")
    parser.add_argument('video', nargs='?', default=None, help="Path to video file (opens dialog if omitted)")
    parser.add_argument('--no-tta', action='store_true', help="Disable TTA by default")
    parser.add_argument('--speed', type=float, default=1.0, help="Initial playback speed (default: 1.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("  HSSN Video Emotion Detection — Student Analysis")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Max faces  : 30")
    print(f"  Classes    : {', '.join(CLASS_NAMES)}")
    print(f"  Image size : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Model      : HSSN v3")
    print("=" * 60)

    # ── Select video file ──
    video_path = args.video
    if video_path is None:
        print("\n  Select a video file...")
        video_path = select_video_file()

    if video_path is None:
        print("  No video file selected. Exiting.")
        sys.exit(0)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  Error: File not found: {video_path}")
        sys.exit(1)

    print(f"\n  Video: {video_path.name}")

    # ── Load model ──
    print("\n  Loading model...")
    model = HierarchicalSpectralSpatialNet(
        num_classes=NUM_CLASSES,
        embed_dim=EMBEDDING_DIM,
        input_size=IMAGE_SIZE
    ).to(DEVICE)

    if CHECKPOINT_PATH.exists():
        print(f"  Loading checkpoint: {CHECKPOINT_PATH.name}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', '?')
        best_acc = checkpoint.get('val_acc', '?')
        print(f"  Loaded (epoch {best_epoch}, val_acc: {best_acc})")
    elif MODEL_PATH.exists():
        print(f"  Loading full model: {MODEL_PATH.name}")
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        print(f"  Loaded")
    else:
        print(f"  No model found!")
        sys.exit(1)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Open video ──
    print(f"\n  Opening video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_time = total_frames / video_fps

    print(f"  Resolution : {video_w}x{video_h}")
    print(f"  FPS        : {video_fps:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {format_time(total_time)}")

    # ── Create YuNet face detector ──
    print("  Initializing YuNet face detector...")
    face_detector, det_scale = create_face_detector(video_w, video_h, det_scale=0.5)
    print(f"  Face detector ready (detection at {int(video_w*0.5)}x{int(video_h*0.5)})")

    # ── Settings ──
    use_tta = not args.no_tta
    show_bars = True
    show_dashboard = True
    is_paused = False
    is_recording = False
    playback_speed = args.speed
    screenshot_count = 0
    MAX_TRACKED_FACES = 30
    SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    # === THREADED DETECTION: never blocks video playback ===
    # Shared state between main thread (display) and detection thread
    det_lock = threading.Lock()           # protects shared annotations
    latest_annotations = []               # latest detection results (thread-safe via det_lock)
    latest_num_faces = 0
    det_frame = None                      # frame for detection thread to process
    det_frame_lock = threading.Lock()     # protects det_frame
    det_running = True                    # signal to stop detection thread

    # Emotion tracking for dashboard
    emotion_counts = {name: 0 for name in CLASS_NAMES}
    total_detections = 0

    # FPS tracking
    fps_buffer = deque(maxlen=30)

    # Video writer for recording
    video_writer = None
    output_video_path = None

    # Shared state for mouse callback
    vstate = VideoState()
    vstate.frame_h = video_h

    # Create window
    window_name = f'HSSN Video Analysis - {video_path.name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(video_w + SIDEBAR_WIDTH, 1580), min(video_h + 50, 840))

    # ── Detection worker thread ──
    def detection_worker():
        """Continuously detect faces and classify emotions on the latest frame.
        Runs in a background thread so video playback never blocks."""
        nonlocal latest_annotations, latest_num_faces, det_running
        while det_running:
            # Grab the latest frame to process
            with det_frame_lock:
                work_frame = det_frame
            if work_frame is None:
                time.sleep(0.005)  # nothing to do yet
                continue

            try:
                # Detect faces
                detected_faces = detect_faces_yunet(face_detector, work_frame, det_scale)
                num_faces = len(detected_faces)

                new_annotations = []
                if num_faces > 0:
                    frame_rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
                    face_tensors = []
                    face_indices = []
                    face_bboxes = []

                    for face_idx, face_info in enumerate(detected_faces[:MAX_TRACKED_FACES]):
                        try:
                            aligned = align_face_yunet(frame_rgb, face_info,
                                                       margin=0.3, target_size=IMAGE_SIZE)
                            if aligned is None:
                                continue
                            tensor = preprocess_face(aligned, IMAGE_SIZE)
                            face_tensors.append(tensor)
                            face_indices.append(face_idx)
                            face_bboxes.append(face_info['bbox'])
                        except Exception:
                            pass

                    # Batch inference — single GPU forward pass
                    if face_tensors:
                        batch_probs = predict_batch(model, face_tensors, DEVICE)
                        for i, (face_idx, bbox) in enumerate(zip(face_indices, face_bboxes)):
                            probs = batch_probs[i]
                            pred_class = np.argmax(probs)
                            emotion = CLASS_NAMES[pred_class]
                            confidence = probs[pred_class]
                            new_annotations.append((bbox, emotion, confidence, probs, face_idx))

                # Atomically update shared state
                with det_lock:
                    latest_annotations = new_annotations
                    latest_num_faces = num_faces

            except Exception as e:
                traceback.print_exc()
                time.sleep(0.01)

    print(f"\n  Starting video analysis... Press SPACE to play/pause, 'q' to quit.\n")

    # Read the first frame and pause
    ret, frame = cap.read()
    if not ret:
        print("  Cannot read first frame!")
        sys.exit(1)
    is_paused = True
    current_frame_idx = 0

    # Start detection thread
    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()

    # Show first frame and attach mouse callback (must be after imshow for Qt)
    first_output = compose_output_frame(frame, emotion_counts, total_detections, 0, show_dashboard)
    cv2.imshow(window_name, first_output)
    cv2.waitKey(1)
    try:
        cv2.setMouseCallback(window_name, mouse_callback, vstate)
    except cv2.error:
        print("  Warning: Mouse seeking disabled (window backend issue)")

    try:
        while True:
            t_start = time.time()

            # Handle seeking from mouse click
            if vstate.seek_requested:
                target_frame = int(vstate.seek_fraction * (total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if ret:
                    current_frame_idx = target_frame
                vstate.seek_requested = False

            # Read next frame if playing
            if not is_paused:
                # Handle speed by skipping frames
                frames_to_skip = max(1, int(playback_speed)) - 1
                for _ in range(frames_to_skip):
                    cap.grab()
                    current_frame_idx += 1

                ret, frame = cap.read()
                if not ret:
                    # End of video — pause at last frame
                    is_paused = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    current_frame_idx = total_frames - 1
                    print("\n  End of video reached. Paused.")
                else:
                    current_frame_idx += 1

            if frame is None:
                break

            display_frame = frame.copy()
            h, w = frame.shape[:2]
            vstate.frame_h = h

            # ── Feed current frame to detection thread (non-blocking) ──
            with det_frame_lock:
                det_frame = frame.copy()

            # ── Draw latest annotations from detection thread (non-blocking) ──
            with det_lock:
                annotations_snapshot = list(latest_annotations)
                num_faces = latest_num_faces

            for bbox, emotion, confidence, probs, face_idx in annotations_snapshot:
                display_frame = draw_prediction(display_frame, bbox, emotion, confidence,
                                                probs, face_id=face_idx, show_bars=show_bars)

            # Update dashboard counts (once per displayed frame, not per loop when paused)
            if not is_paused and annotations_snapshot:
                for bbox, emotion, confidence, probs, face_idx in annotations_snapshot:
                    emotion_counts[emotion] += 1
                    total_detections += 1

            # FPS calculation
            elapsed = time.time() - t_start
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            avg_fps = np.mean(fps_buffer)

            # Current time in video
            current_time = current_frame_idx / video_fps

            # Draw overlays
            display_frame = draw_controls_help(display_frame)

            # Draw TTA and FPS info (top-right area, below help text)
            tta_text = f"TTA: {'ON' if use_tta else 'OFF'} | FPS: {avg_fps:.1f}"
            cv2.putText(display_frame, tta_text, (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            # Draw progress bar
            display_frame, bar_l, bar_r, bar_t, bar_b = draw_progress_bar(
                display_frame, current_frame_idx, total_frames,
                current_time, total_time, is_paused, playback_speed, is_recording
            )
            vstate.bar_left = bar_l
            vstate.bar_right = bar_r
            vstate.bar_top = bar_t
            vstate.bar_bottom = bar_b

            # Compose final output with dedicated right sidebar
            output_frame = compose_output_frame(
                display_frame, emotion_counts, total_detections, num_faces, show_dashboard
            )

            # Write frame if recording
            if is_recording and video_writer is not None:
                video_writer.write(output_frame)

            # Show
            cv2.imshow(window_name, output_frame)

            # Calculate wait time — play at real video speed
            if is_paused:
                wait_ms = 30  # Just poll for input when paused
            else:
                target_frame_time = 1.0 / (video_fps * playback_speed)
                processing_time = time.time() - t_start
                remaining = target_frame_time - processing_time
                wait_ms = max(1, int(remaining * 1000))

            key = cv2.waitKey(wait_ms) & 0xFF

            # ── Key handling ──
            if key == ord('q') or key == 27:  # q or ESC
                print("\n  Quitting...")
                break

            elif key == ord(' '):  # SPACE — play/pause
                is_paused = not is_paused
                print(f"  {'Paused' if is_paused else 'Playing'}")

            elif key == ord('s'):  # Screenshot
                screenshot_count += 1
                filename = SCREENSHOT_DIR / f"video_screenshot_{screenshot_count:04d}.png"
                cv2.imwrite(str(filename), output_frame)
                print(f"  Screenshot saved: {filename.name}")

            elif key == ord('t'):  # Toggle TTA
                use_tta = not use_tta
                print(f"  TTA {'enabled' if use_tta else 'disabled'}")

            elif key == ord('f'):  # Toggle probability bars
                show_bars = not show_bars
                print(f"  Probability bars {'shown' if show_bars else 'hidden'}")

            elif key == ord('d'):  # Toggle dashboard
                show_dashboard = not show_dashboard
                print(f"  Dashboard {'shown' if show_dashboard else 'hidden'}")

            elif key == ord('r'):  # Toggle recording
                if not is_recording:
                    # Start recording
                    output_name = f"{video_path.stem}_annotated_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    output_video_path = VIDEO_OUTPUT_DIR / output_name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_h, out_w = output_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(output_video_path), fourcc, video_fps, (out_w, out_h)
                    )
                    is_recording = True
                    print(f"  Recording started: {output_name}")
                else:
                    # Stop recording
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print(f"  Recording saved: {output_video_path.name}")

            elif key == 83 or key == ord('.'):  # RIGHT arrow or '.' — next frame
                if is_paused:
                    ret, frame = cap.read()
                    if ret:
                        current_frame_idx += 1
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                        ret, frame = cap.read()
                        current_frame_idx = total_frames - 1

            elif key == 81 or key == ord(','):  # LEFT arrow or ',' — previous frame
                if is_paused:
                    target = max(0, current_frame_idx - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ret, frame = cap.read()
                    if ret:
                        current_frame_idx = target

            elif key == 82:  # UP arrow — increase speed
                current_idx = 0
                for i, s in enumerate(SPEED_OPTIONS):
                    if abs(s - playback_speed) < 0.01:
                        current_idx = i
                        break
                new_idx = min(current_idx + 1, len(SPEED_OPTIONS) - 1)
                playback_speed = SPEED_OPTIONS[new_idx]
                print(f"  Speed: {playback_speed}x")

            elif key == 84:  # DOWN arrow — decrease speed
                current_idx = len(SPEED_OPTIONS) - 1
                for i, s in enumerate(SPEED_OPTIONS):
                    if abs(s - playback_speed) < 0.01:
                        current_idx = i
                        break
                new_idx = max(current_idx - 1, 0)
                playback_speed = SPEED_OPTIONS[new_idx]
                print(f"  Speed: {playback_speed}x")

            elif key == 80:  # HOME — jump to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                current_frame_idx = 0
                # Reset dashboard
                emotion_counts = {name: 0 for name in CLASS_NAMES}
                total_detections = 0
                print("  Jumped to start")

            elif key == 87:  # END — jump to near end
                target = max(0, total_frames - int(5 * video_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if ret:
                    current_frame_idx = target
                print("  Jumped to end")

            elif ord('1') <= key <= ord('9'):
                # Number keys: jump to percentage
                pct = (key - ord('0')) / 10.0
                target = int(pct * total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if ret:
                    current_frame_idx = target
                print(f"  Jumped to {int(pct*100)}%")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        # Stop detection thread
        det_running = False
        det_thread.join(timeout=2.0)
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"  Recording saved: {output_video_path.name}")
        cv2.destroyAllWindows()

        # Print summary
        if total_detections > 0:
            print("\n" + "=" * 60)
            print("  Emotion Detection Summary")
            print("=" * 60)
            print(f"  Video      : {video_path.name}")
            print(f"  Detections : {total_detections}")
            print(f"  Distribution:")
            for name in CLASS_NAMES:
                count = emotion_counts.get(name, 0)
                pct = count / total_detections * 100
                bar = '#' * int(pct / 2)
                print(f"    {name:>10s}: {count:5d} ({pct:5.1f}%) {bar}")
            print("=" * 60)

        print("\n  Done. Goodbye!")


if __name__ == '__main__':
    main()
