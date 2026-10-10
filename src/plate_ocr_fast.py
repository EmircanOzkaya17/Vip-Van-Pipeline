"""
Plakaya ozel OCR: fast-plate-ocr CCT-S v2 (global, 65 ulke - Turkiye dahil).

Neden PaddleOCR yerine bu?
  * PaddleOCR genel amacli bir belge OCR'idir: once metin satiri ARAR, sonra
    okur. 80-100 piksellik, egik, bulanik bir plakada en sik basarisiz olan
    adim tam olarak bu arama adimidir.
  * Bu model dogrudan plaka KESITLERI uzerinde egitilmistir: girdinin tamami
    plakadir, cikti sabit 10 karakter yuvasidir. Tespit adimi yoktur; kucuk ve
    dusuk cozunurluklu plakalarda cok daha kararlidir.
  * ~5 MB, CPU'da milisaniyeler; OpenCV DNN ile calisir (onnxruntime yok).

Model: https://github.com/ankandrew/fast-plate-ocr (MIT). models/ icindeki
dosya OpenCV uyumu icin yamalanmistir (bkz. tools/prepare_plate_models.py).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .config import (
    PLATE_OCR_FAST_ALPHABET,
    PLATE_OCR_FAST_INPUT_SIZE,
    PLATE_OCR_FAST_ONNX,
    PLATE_OCR_FAST_PAD_CHAR,
    PLATE_OCR_FAST_SLOTS,
)
from .dnn_utils import load_onnx_net

LOGGER = logging.getLogger(__name__)

# [(metin, güven), ...]
OcrResult = List[Tuple[str, float]]


class FastPlateOCR:
    """Tek satirlik plaka kesitini sabit yuvali CCT modeliyle okur."""

    name = "fastplate"

    def __init__(self, model_path: str = PLATE_OCR_FAST_ONNX) -> None:
        self.net = load_onnx_net(model_path, "plaka OCR modeli")
        self.width, self.height = PLATE_OCR_FAST_INPUT_SIZE
        self.alphabet = np.array(list(PLATE_OCR_FAST_ALPHABET))
        # Saglamlik testi: OpenCV surumune bagli sayisal hatalari erken yakala
        probe = self._forward(np.full((self.height, self.width, 3), 128, np.uint8))
        if not np.isfinite(probe).all():
            raise RuntimeError(
                f"Plaka OCR modeli OpenCV {cv2.__version__} ile gecersiz (NaN) "
                "cikti uretti. models/ klasorundeki yamali modeli kullandiginizdan "
                "emin olun (tools/prepare_plate_models.py)."
            )
        LOGGER.info("Plaka OCR hazir (fast-plate-ocr CCT-S v2, OpenCV %s).",
                    cv2.__version__)

    # ------------------------------------------------------------------ #
    def _forward(self, rgb: np.ndarray) -> np.ndarray:
        blob = rgb.astype(np.float32)[None, ...]            # (1, H, W, 3)
        self.net.setInput(blob)
        # Yamali modelde tek cikis ("plate") vardir. Orijinal model verilirse
        # (plate + region) tum cikislar istenir ve dogru boyuttaki secilir;
        # OpenCV 5 cikislarin bir kismini istemeye izin vermez.
        names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(names) if len(names) > 1 else [self.net.forward()]
        expected = PLATE_OCR_FAST_SLOTS * len(self.alphabet)
        output = next((o for o in outputs if o.size == expected), outputs[0])
        return output.reshape(PLATE_OCR_FAST_SLOTS, len(self.alphabet))

    # ------------------------------------------------------------------ #
    def read_with_probs(self, plate_bgr: np.ndarray) -> Tuple[str, np.ndarray]:
        """Metni ve her karakterin olasiligini doner (dolgu yuvalari haric)."""
        if plate_bgr is None or plate_bgr.size == 0:
            return "", np.zeros(0)
        if plate_bgr.ndim == 2:
            plate_bgr = cv2.cvtColor(plate_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        probs = self._forward(rgb)
        indices = probs.argmax(axis=1)
        chars = self.alphabet[indices]
        char_probs = probs.max(axis=1)
        keep = chars != PLATE_OCR_FAST_PAD_CHAR
        return "".join(chars[keep]), char_probs[keep]

    # ------------------------------------------------------------------ #
    def read(self, plate_bgr: np.ndarray) -> OcrResult:
        """
        PlateOCR ile ayni arayuz: [(metin, guven)]. Guven, okunan karakterlerin
        ortalama olasiligidir; tek bir belirsiz karakter bunu belirgin dusurur.
        """
        text, char_probs = self.read_with_probs(plate_bgr)
        if not text:
            return []
        return [(text, float(np.mean(char_probs)))]
