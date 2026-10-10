"""
OpenCV görselleştirme ve etkileşim: sanal çizgi, track kutuları/etiketleri,
bilgi paneli (HUD) çizimi ve fare ile sanal çizgi seçimi.

Tüm çizim fonksiyonları verilen frame'i yerinde (in-place) değiştirir.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, List, Tuple

import cv2
import numpy as np

from .config import (
    COLOR_LINE,
    COLOR_NON_TARGET,
    COLOR_PENDING,
    COLOR_TARGET,
    COLOR_TEXT,
    DRAW_NON_TARGET_BOXES,
)
from .track import TrackInfo
from .virtual_line import VirtualLine

LOGGER = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Arka planı dolu, okunabilir bir etiket çizer."""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(y, th + 4)
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - baseline + 1), FONT, scale,
                (20, 20, 20), thickness, cv2.LINE_AA)


def draw_virtual_line(frame: np.ndarray, line: VirtualLine) -> None:
    """Sanal çizgiyi, uç noktalarını ve 'TRIGGER LINE' etiketini çizer."""
    p1 = (int(line.start[0]), int(line.start[1]))
    p2 = (int(line.end[0]), int(line.end[1]))
    cv2.line(frame, p1, p2, COLOR_LINE, 3, cv2.LINE_AA)
    cv2.circle(frame, p1, 5, COLOR_LINE, -1)
    cv2.circle(frame, p2, 5, COLOR_LINE, -1)
    draw_label(frame, "TRIGGER LINE", (p2[0] + 8, p2[1]), COLOR_LINE, 0.45)


def draw_tracks(frame: np.ndarray, tracks: List[TrackInfo]) -> None:
    """Aktif track'lerin kutularını ve etiketlerini çizer."""
    for track in tracks:
        if track.is_target is True:
            color = COLOR_TARGET
        elif track.is_target is False:
            if not DRAW_NON_TARGET_BOXES:
                continue
            color = COLOR_NON_TARGET
        else:
            color = COLOR_PENDING

        x1, y1, x2, y2 = track.bbox
        thickness = 2 if track.is_target else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Üst etiket: ID + sınıf + tip
        header = f"ID:{track.track_id}"
        if track.vehicle_class:
            header += f" {track.vehicle_class}"
        if track.is_target is True:
            header += " | VIP VAN"
        elif track.is_target is False:
            header += " | non-target"
        draw_label(frame, header, (x1, y1 - 4), color, 0.45)

        # Alt etiket: skorlar ve plaka
        details: List[str] = []
        if track.final_positive is not None:
            details.append(
                f"p={track.final_positive:.2f} m={track.final_margin:.2f}"
            )
        if track.plate_text:
            details.append(f"[{track.plate_text}]")
        elif track.is_target and track.plate_attempts:
            details.append("plate: ...")
        if details:
            draw_label(frame, " ".join(details), (x1, y2 + 18), color, 0.45)


def draw_hud(
    frame: np.ndarray, frame_index: int, fps: float, stats: Counter, active: int
) -> None:
    """Ekranın üst kısmına özet bilgi paneli çizer."""
    lines = [
        f"Frame: {frame_index} | FPS: {fps:5.1f} | Active: {active}",
        f"Crossings: {stats['line_crossings']} | "
        f"Targets: {stats['target_vehicles']} | "
        f"Plates: {stats['plates_read']}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (430, 22 * len(lines) + 12), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 22 + i * 22), FONT, 0.55, COLOR_TEXT, 1,
                    cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    line: VirtualLine,
    tracks: List[TrackInfo],
    frame_index: int,
    fps: float,
    stats: Counter,
) -> None:
    """Bir frame'in tüm anotasyonlarını çizer: çizgi -> track'ler -> HUD."""
    draw_virtual_line(frame, line)
    draw_tracks(frame, tracks)
    draw_hud(frame, frame_index, fps, stats, len(tracks))


def pick_line_interactively(
    frame: np.ndarray, default_line: VirtualLine
) -> VirtualLine:
    """
    İlk frame üzerinde fare ile iki nokta seçtirerek sanal çizgiyi tanımlar.
    ENTER ile onay, ESC ile varsayılan çizgiye dönüş.
    """
    points: List[Tuple[int, int]] = []
    window = "Sanal cizgiyi sec (2 nokta) - ENTER: onayla, ESC: varsayilan"

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    LOGGER.info("Cizgi secimi: sol tikla 2 nokta belirleyin, ENTER ile onaylayin.")

    while True:
        preview = frame.copy()
        for point in points:
            cv2.circle(preview, point, 6, COLOR_LINE, -1)
        if len(points) == 2:
            cv2.line(preview, points[0], points[1], COLOR_LINE, 3, cv2.LINE_AA)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) == 2:       # ENTER
            cv2.destroyWindow(window)
            return VirtualLine(points[0], points[1], reference_resolution=None)
        if key == 27:                            # ESC
            cv2.destroyWindow(window)
            return default_line
        if key == ord("r"):                      # Yeniden seç
            points.clear()
