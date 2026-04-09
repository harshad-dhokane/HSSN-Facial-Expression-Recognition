"""
HSSN Webcam Emotion Detection — Live Test
==========================================
Tests the trained Hierarchical Spectral-Spatial Network (HSSN) model
on live webcam feed. Detects face using MediaPipe, aligns it, and
classifies the expression in real-time.

Usage:
    python webcam_test.py

Controls:
    q / ESC  — Quit
    s        — Save screenshot
    t        — Toggle TTA on/off
    f        — Toggle FPS display
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import mediapipe as mp

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / 'outputs' / 'exports' / 'hssn_full_model.pth'
CHECKPOINT_PATH = BASE_DIR / 'outputs' / 'checkpoints' / 'best_model.pth'
CLASS_MAPPING_PATH = BASE_DIR / 'outputs' / 'exports' / 'class_mapping.json'
CONFIG_PATH = BASE_DIR / 'outputs' / 'logs' / 'experiment_config.json'
FACE_MODEL_PATH = str(BASE_DIR / 'face_landmarker_v2_with_blendshapes.task')
SCREENSHOT_DIR = BASE_DIR / 'outputs' / 'screenshots'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

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

# Normalization (from training — computed on training set)
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

# Eye landmark indices for alignment
LEFT_EYE_INDICES = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 387, 373, 385, 380]


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
        self.g_range = gaussian_sigma_range
        self.sp_range = sp_density_range

    def forward(self, x):
        return x  # No-op at inference


class IlluminationPerturbation(nn.Module):
    def __init__(self, gamma_range=(0.7, 1.4), channel_scale_range=(0.85, 1.15)):
        super().__init__()
        self.gamma_range = gamma_range
        self.cs_range = channel_scale_range

    def forward(self, x):
        return x  # No-op at inference


class HierarchicalSpectralSpatialNet(nn.Module):
    def __init__(self, num_classes=8, embed_dim=256, input_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.use_checkpointing = False  # No checkpointing at inference

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
#  FACE ALIGNMENT
# ──────────────────────────────────────────────

def create_landmarker():
    """Create MediaPipe FaceLandmarker."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        num_faces=10,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return FaceLandmarker.create_from_options(options)


def get_eye_centers(landmarks, w, h):
    left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_INDICES])
    return left_eye.mean(axis=0), right_eye.mean(axis=0)


def align_face(img_np, landmarks, w, h, margin=0.3, target_size=224):
    """Align face using eye landmarks: rotate → crop → resize."""
    left_eye, right_eye = get_eye_centers(landmarks, w, h)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    all_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in range(len(landmarks))])
    ones = np.ones((all_pts.shape[0], 1))
    pts_h = np.hstack([all_pts, ones])
    rotated_pts = (M @ pts_h.T).T

    x_min, y_min = rotated_pts.min(axis=0)
    x_max, y_max = rotated_pts.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(0, int(x_min - margin * bw))
    y_min = max(0, int(y_min - margin * bh))
    x_max = min(w, int(x_max + margin * bw))
    y_max = min(h, int(y_max + margin * bh))

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    side = max(crop_w, crop_h)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = rotated
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def histogram_equalize_rgb(img_np):
    """Apply histogram equalization per channel (numpy array input/output)."""
    result = np.zeros_like(img_np)
    for c in range(3):
        channel = img_np[:, :, c]
        hist, bins = np.histogram(channel.ravel(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-8)
        result[:, :, c] = cdf_normalized[channel]
    return result.astype(np.uint8)


def get_face_bbox_from_landmarks(landmarks, w, h, margin=0.15):
    """Get bounding box from landmarks for drawing on frame."""
    xs = [landmarks[i].x * w for i in range(len(landmarks))]
    ys = [landmarks[i].y * h for i in range(len(landmarks))]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(0, int(x_min - margin * bw))
    y_min = max(0, int(y_min - margin * bh))
    x_max = min(w, int(x_max + margin * bw))
    y_max = min(h, int(y_max + margin * bh))
    return x_min, y_min, x_max, y_max


# ──────────────────────────────────────────────
#  INFERENCE FUNCTIONS
# ──────────────────────────────────────────────

def preprocess_face(face_np, target_size=224):
    """Convert aligned face numpy array to model input tensor."""
    face_eq = histogram_equalize_rgb(face_np)
    face_pil = Image.fromarray(face_eq)
    tensor = TF.to_tensor(face_pil)
    tensor = TF.normalize(tensor, NORM_MEAN, NORM_STD)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict_single(model, tensor, device):
    """Single forward pass prediction."""
    with torch.no_grad():
        logits, embedding = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0], embedding.cpu().numpy()[0]


def predict_tta(model, face_np, device, target_size=224):
    """Test-Time Augmentation: average over 6 views."""
    tta_views = []

    # 1. Original
    tta_views.append(preprocess_face(face_np, target_size))

    # 2. Horizontal flip
    flipped = np.fliplr(face_np).copy()
    tta_views.append(preprocess_face(flipped, target_size))

    # 3-6. Rotations
    for angle in [-5, 5, -10, 10]:
        h, w = face_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(face_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        tta_views.append(preprocess_face(rotated, target_size))

    # Average probabilities
    all_probs = []
    for view in tta_views:
        with torch.no_grad():
            logits, _ = model(view.to(device))
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


# ──────────────────────────────────────────────
#  DISPLAY HELPERS
# ──────────────────────────────────────────────

def draw_prediction(frame, bbox, emotion, confidence, all_probs, face_id=0, show_bars=True):
    """Draw bounding box, emotion label, and compact probability bars for each face."""
    x1, y1, x2, y2 = bbox
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    frame_h, frame_w = frame.shape[:2]

    # Draw face bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Face ID circle in top-left of bbox
    cv2.circle(frame, (x1 + 12, y1 + 12), 12, color, -1)
    cv2.putText(frame, str(face_id + 1), (x1 + 7, y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw emotion label above box
    label = f"{emotion} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Label background
    label_y1 = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    # Draw compact probability bars below the bounding box
    if show_bars and all_probs is not None:
        bar_x = x1
        bar_y_start = min(y2 + 5, frame_h - (NUM_CLASSES * 18 + 5))
        bar_height = 14
        bar_max_width = min(150, x2 - x1, frame_w - bar_x - 5)
        if bar_max_width < 60:
            bar_max_width = 150
            bar_x = max(0, min(x1, frame_w - bar_max_width - 5))

        # Top 4 emotions only (keep it compact for multi-face)
        sorted_indices = np.argsort(all_probs)[::-1][:4]
        for i, idx in enumerate(sorted_indices):
            y = bar_y_start + i * (bar_height + 3)
            if y + bar_height > frame_h - 40:
                break
            name = CLASS_NAMES[idx]
            prob = all_probs[idx]
            bar_color = EMOTION_COLORS.get(name, (200, 200, 200))

            # Bar background
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                          (40, 40, 40), -1)
            # Bar fill
            bar_width = int(prob * bar_max_width)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                          bar_color, -1)
            # Bar border
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                          (80, 80, 80), 1)
            # Label
            cv2.putText(frame, f"{name}: {prob:.0%}", (bar_x + 2, y + bar_height - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

    return frame


def draw_info(frame, fps, use_tta, num_faces):
    """Draw FPS, face count, and status info on frame."""
    h, w = frame.shape[:2]

    # Status bar at bottom
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)

    face_text = f"Faces: {num_faces}" if num_faces > 0 else "No face detected"
    info_text = f"FPS: {fps:.1f} | {face_text} | TTA: {'ON' if use_tta else 'OFF'} | Device: {DEVICE}"
    cv2.putText(frame, info_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Controls hint
    cv2.putText(frame, "q:Quit | s:Screenshot | t:Toggle TTA | f:Toggle bars", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HSSN Webcam Emotion Detection (Multi-Face)")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Max faces  : 10")
    print(f"  Classes    : {', '.join(CLASS_NAMES)}")
    print(f"  Image size : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Model      : HSSN v3")
    print("=" * 60)

    # ── Load model ──
    print("\n  Loading model...")
    model = HierarchicalSpectralSpatialNet(
        num_classes=NUM_CLASSES,
        embed_dim=EMBEDDING_DIM,
        input_size=IMAGE_SIZE
    ).to(DEVICE)

    # Try loading from checkpoint
    if CHECKPOINT_PATH.exists():
        print(f"  Loading checkpoint: {CHECKPOINT_PATH.name}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', '?')
        best_acc = checkpoint.get('val_acc', '?')
        print(f"  ✓ Loaded (epoch {best_epoch}, val_acc: {best_acc})")
    elif MODEL_PATH.exists():
        print(f"  Loading full model: {MODEL_PATH.name}")
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        print(f"  ✓ Loaded")
    else:
        print(f"  ✗ No model found at:")
        print(f"    {CHECKPOINT_PATH}")
        print(f"    {MODEL_PATH}")
        sys.exit(1)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Create MediaPipe landmarker ──
    print("  Initializing MediaPipe face detector...")
    landmarker = create_landmarker()
    print("  ✓ Face detector ready")

    # ── Open webcam ──
    print("\n  Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ Cannot open webcam! Check your camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✓ Webcam opened: {actual_w}×{actual_h}")

    # ── Settings ──
    use_tta = False
    show_bars = True
    fps_buffer = deque(maxlen=30)
    screenshot_count = 0
    MAX_TRACKED_FACES = 10

    # Per-face prediction smoothing buffers (keyed by face index)
    # Each face gets its own deque to smooth predictions over frames
    face_pred_buffers = {i: deque(maxlen=5) for i in range(MAX_TRACKED_FACES)}

    print("\n  ✓ Starting live multi-face detection... Press 'q' or ESC to quit.\n")

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("  ✗ Failed to read frame")
                break

            # Convert BGR → RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Detect all faces
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            num_faces = len(result.face_landmarks) if result.face_landmarks else 0

            if num_faces > 0:
                for face_idx, landmarks in enumerate(result.face_landmarks):
                    if face_idx >= MAX_TRACKED_FACES:
                        break

                    # Get bounding box for display
                    bbox = get_face_bbox_from_landmarks(landmarks, w, h)

                    # Align face for model input
                    try:
                        aligned_face = align_face(frame_rgb, landmarks, w, h,
                                                   margin=0.3, target_size=IMAGE_SIZE)

                        # Predict
                        if use_tta:
                            probs = predict_tta(model, aligned_face, DEVICE, IMAGE_SIZE)
                        else:
                            tensor = preprocess_face(aligned_face, IMAGE_SIZE)
                            probs, _ = predict_single(model, tensor, DEVICE)

                        # Smooth predictions per face
                        face_pred_buffers[face_idx].append(probs)
                        avg_probs = np.mean(list(face_pred_buffers[face_idx]), axis=0)

                        pred_class = np.argmax(avg_probs)
                        emotion = CLASS_NAMES[pred_class]
                        confidence = avg_probs[pred_class]

                        # Draw results for this face
                        frame = draw_prediction(frame, bbox, emotion, confidence,
                                                avg_probs, face_id=face_idx, show_bars=show_bars)

                    except Exception:
                        # Face alignment failed — draw box only
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, f"Face {face_idx+1}: align failed",
                                    (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Clear buffers for faces no longer visible
                for i in range(num_faces, MAX_TRACKED_FACES):
                    face_pred_buffers[i].clear()
            else:
                # No faces — clear all buffers
                for buf in face_pred_buffers.values():
                    buf.clear()

            # FPS calculation
            elapsed = time.time() - t_start
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            avg_fps = np.mean(fps_buffer)

            # Draw info
            frame = draw_info(frame, avg_fps, use_tta, num_faces)

            # Show frame
            cv2.imshow('HSSN Emotion Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("\n  Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = SCREENSHOT_DIR / f"screenshot_{screenshot_count:04d}.png"
                cv2.imwrite(str(filename), frame)
                print(f"  📸 Screenshot saved: {filename.name}")
            elif key == ord('t'):
                use_tta = not use_tta
                for buf in face_pred_buffers.values():
                    buf.clear()
                print(f"  TTA {'enabled' if use_tta else 'disabled'}")
            elif key == ord('f'):
                show_bars = not show_bars
                print(f"  Probability bars {'shown' if show_bars else 'hidden'}")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\n  ✓ Webcam released. Goodbye!")


if __name__ == '__main__':
    main()
