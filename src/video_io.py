"""
Video girdi/çıktı: kaynak açma (dosya / RTSP / kamera) ve codec toleranslı
VideoWriter oluşturma.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .config import (
    OUTPUT_FOURCC_CANDIDATES,
    OUTPUT_FPS_FALLBACK,
    OUTPUT_FPS_MAX,
)
from .file_utils import ensure_writable_directory
from .output_paths import stream_output_directory

LOGGER = logging.getLogger(__name__)


def open_video_source(source: str) -> cv2.VideoCapture:
    """Dosya, RTSP veya kamera indeksinden video akışı açar."""
    capture_source: Any = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Video kaynagi acilamadi: {source}")
    return capture


def create_video_writer(
    path: str, fps: float, size: Tuple[int, int]
) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    Çıktı dizinini hazırlar ve çalışan bir codec bularak VideoWriter döndürür.

    Hedef dizin yazılamıyorsa (salt okunur disk, ağ paylaşımı vb.) çalışma
    dizini altındaki yedek klasöre düşülür. Bu yüzden fonksiyon, fiilen
    kullanılan yolu da döndürür.

    Dönüş: (writer, gerçek_yol) - başarısızlıkta (None, None).
    """
    directory = os.path.dirname(path) or os.getcwd()
    if not ensure_writable_directory(directory):
        fallback = stream_output_directory()
        if not ensure_writable_directory(fallback):
            LOGGER.error("Cikti dizini olusturulamadi: %s", directory)
            return None, None
        LOGGER.warning(
            "Cikti dizinine yazilamiyor (%s). Su dizine gecildi: %s",
            directory, fallback,
        )
        path = os.path.join(fallback, os.path.basename(path))

    if not fps or not np.isfinite(fps) or fps < 1.0:
        LOGGER.debug("Kaynak FPS gecersiz (%s) -> %.1f", fps, OUTPUT_FPS_FALLBACK)
        fps = OUTPUT_FPS_FALLBACK
    fps = float(min(fps, OUTPUT_FPS_MAX))

    for codec in OUTPUT_FOURCC_CANDIDATES:
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        except Exception as exc:
            LOGGER.debug("Codec olusturulamadi (%s): %s", codec, exc)
            continue
        if writer.isOpened():
            LOGGER.info(
                "Cikti videosu: %s  (codec=%s, %dx%d @ %.1f FPS)",
                path, codec, size[0], size[1], fps,
            )
            return writer, path
        writer.release()
        LOGGER.debug("Codec acilamadi: %s", codec)

    LOGGER.error("Video yazici acilamadi (%s). Kayit yapilmayacak.", path)
    return None, None
