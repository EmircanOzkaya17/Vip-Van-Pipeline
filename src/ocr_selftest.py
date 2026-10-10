"""
Plaka zinciri tanı modu (--test-ocr).

Video işlemeden, tek bir görüntü üzerinde plaka zincirinin her adımını ayrı
ayrı çalıştırır ve ara çıktıları konsola yazar.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .bbox import crop_image
from .config import OCR_MIN_CONFIDENCE, PLATE_CROP_PADDING
from .plate_image import preprocess_plate, refine_plate_region
from .plate_reader import PlateReader
from .plate_text import correct_and_validate_plate
from .reporting import SEPARATOR_WIDTH

LOGGER = logging.getLogger(__name__)


def read_image_unicode(path: str) -> Optional[np.ndarray]:
    """
    Goruntuyu ASCII disi karakter iceren yollardan da okur.

    cv2.imread Windows'ta "C:\\Users\\MELSA ÖZKAYA\\..." gibi yollari acamaz
    ve sessizce None doner. np.fromfile Python'un unicode dosya API'sini
    kullandigi icin bu sorun yasanmaz.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def run_ocr_selftest(image_path: str) -> int:
    """
    Tek bir goruntu uzerinde plaka zincirini ayrintili calistirir.

    Plaka okunamadiginda hangi asamanin sorumlu oldugunu kesin olarak gormek
    icindir. Arac kesiti, plaka kesiti veya duzlestirilmis kesit verilebilir;
    her adimin ciktisi ve PaddleOCR'in ham donusu ekrana yazilir.
    """
    image = read_image_unicode(image_path)
    if image is None:
        LOGGER.error("Goruntu okunamadi: %s", image_path)
        return 1

    print("=" * SEPARATOR_WIDTH)
    print(f" OCR KENDI KENDINE TEST: {image_path}")
    print("=" * SEPARATOR_WIDTH)
    print(f" Girdi boyutu: {image.shape[1]}x{image.shape[0]}")

    try:
        reader = PlateReader()
    except RuntimeError as exc:
        LOGGER.error("Plaka okuyucu baslatilamadi: %s", exc)
        return 1

    print(f" Dedektor: {reader.detector.backend_name} | OCR: "
          f"{'+'.join(name for name, _ in reader.engines)} | OpenCV {cv2.__version__}")

    # --- 1) Plaka tespiti ---
    boxes = reader.detector.detect(image)
    print(f"\n[1] Plaka tespiti: {len(boxes)} kutu "
          f"(geometri suzgecinin eledigi: {reader.rejected_boxes}, "
          f"yakinlastirmayla bulunan: {reader.detector.zoom_hits})")
    for box, conf in boxes:
        width, height = box[2] - box[0], box[3] - box[1]
        print(f"    {box} guven={conf:.3f} {width}x{height} "
              f"oran={width / max(height, 1):.2f}")

    # Kutu bulunamazsa goruntunun kendisi zaten plaka kabul edilir
    crops = [crop_image(image, box, PLATE_CROP_PADDING) for box, _ in boxes]
    crops = [c for c in crops if c is not None] or [image]

    for index, crop in enumerate(crops):
        print(f"\n--- Kesit {index}: {crop.shape[1]}x{crop.shape[0]} ---")

        # --- 2) Bolge iyilestirme (egiklik duzeltme) ---
        variants = [("orijinal", crop)]
        refined = refine_plate_region(crop)
        if refined is not None:
            ratio = refined.shape[1] / max(refined.shape[0], 1)
            print(f"[2] Iyilestirme: {refined.shape[1]}x{refined.shape[0]} "
                  f"(oran {ratio:.2f})")
            variants.insert(0, ("duzlestirilmis", refined))
        else:
            print("[2] Iyilestirme: bolge bulunamadi, yalnizca orijinal kesit")

        # --- 3) Her OCR motoru x her kesit varyanti ---
        any_text = []
        for engine_name, engine in reader.engines:
            for variant_name, variant in variants:
                if engine_name == "paddle":
                    prepared = preprocess_plate(variant)
                    results = (engine.detect_and_recognize(prepared)
                               + engine.recognize_only(prepared))
                else:
                    results = engine.read(variant)
                print(f"[3] {engine_name:<9} / {variant_name:<14}: "
                      f"{results if results else 'BOS'}")
                any_text.extend(results)
                if len(results) > 1:
                    merged = "".join(text for text, _ in results)
                    any_text.append((merged, float(np.mean([c for _, c in results]))))

        # --- 4) Format dogrulama ---
        print("[4] Format dogrulama:")
        if not any_text:
            print("    (dogrulanacak metin yok)")
        for text, conf in any_text:
            plate = correct_and_validate_plate(text)
            status = plate if plate else "GECERSIZ FORMAT"
            flag = "" if conf >= OCR_MIN_CONFIDENCE else "  <- guven esigin altinda"
            print(f"    '{text}' ({conf:.3f}) -> {status}{flag}")

    # --- 5) Videoda kullanilan zincirin tamami ---
    attempt = reader.read_plate(image)
    print(f"\n[5] Zincirin tamami (videodaki ile ayni): {attempt.describe()}")

    print("=" * SEPARATOR_WIDTH)
    return 0
