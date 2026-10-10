"""
OpenCV DNN yardimcilari.

Model dosyasi diskten Python ile (np.fromfile) okunup OpenCV'ye BELLEKTEN
verilir. cv2.dnn.readNetFromONNX(yol) Windows'ta ASCII disi karakter iceren
yollari (ör. "C:\\Users\\MELSA ÖZKAYA\\...") acamaz; bellekten yukleme bu
sorunu tamamen ortadan kaldirir.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def load_onnx_net(path: str, label: str = "model") -> "cv2.dnn.Net":
    """ONNX modelini yolundan bagimsiz olarak (unicode guvenli) yukler."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} dosyasi bulunamadi: {path}\n"
            "models/ klasorunun proje kokunde oldugundan emin olun "
            "(yeniden uretmek icin: python tools/prepare_plate_models.py)."
        )
    data = np.fromfile(path, dtype=np.uint8)
    net = cv2.dnn.readNetFromONNX(data)
    if int(cv2.__version__.split(".")[0]) < 5:
        # OpenCV 5'in yeni motoru hedef secimini desteklemiyor (uyari basar);
        # 4.x'te varsayilan zaten bu, yalnizca acikca belirtiliyor.
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    LOGGER.debug("%s yuklendi: %s (OpenCV %s)", label, path, cv2.__version__)
    return net
