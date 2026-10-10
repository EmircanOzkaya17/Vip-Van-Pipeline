"""
Bounding box yardımcıları: görüntü sınırlarına kırpma, oransal genişletme,
merkez hesabı ve görüntüden güvenli kesit alma.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Piksel koordinatlarında (x1, y1, x2, y2) kutu
Box = Tuple[int, int, int, int]


def clamp_box(box: Sequence[float], width: int, height: int) -> Box:
    """Bounding box koordinatlarını görüntü sınırları içine hapseder."""
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width, x2)))
    y2 = int(max(0, min(height, y2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def expand_box(
    box: Sequence[float], padding_ratio: float, width: int, height: int
) -> Box:
    """Bounding box'ı verilen oranda genişletir ve sınırlara kırpar."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * padding_ratio, bh * padding_ratio
    return clamp_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def crop_image(
    frame: np.ndarray, box: Sequence[float], padding_ratio: float = 0.0
) -> Optional[np.ndarray]:
    """
    Verilen kutuya göre görüntüden güvenli bir kesit alır.
    Geçersiz / çok küçük kesitlerde None döner.
    """
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(box, padding_ratio, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    """Bounding box merkez noktasını (cx, cy) döner."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
