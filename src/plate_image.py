"""
Plaka görüntüsü işleme: bölge iyileştirme (eğiklik düzeltme) ve OCR öncesi
ön işleme.

Yalnızca OpenCV + NumPy kullanır; model bağımlılığı yoktur. Durumsuz (stateless)
fonksiyonlardan oluştuğu için plaka zincirinden bağımsız olarak da kullanılabilir.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    PLATE_MIN_CROP_WIDTH,
    PLATE_OCR_MAX_UPSCALE,
    PLATE_OCR_MIN_WIDTH,
    PLATE_OCR_TARGET_HEIGHT,
    PLATE_REFINE_CENTER_BAND,
    PLATE_REFINE_IDEAL_RATIO,
    PLATE_REFINE_MARGIN_X,
    PLATE_REFINE_MARGIN_Y,
    PLATE_REFINE_MAX_AREA_FRAC,
    PLATE_REFINE_MAX_RATIO,
    PLATE_REFINE_MIN_AREA_FRAC,
    PLATE_REFINE_MIN_FILL,
    PLATE_REFINE_MIN_RATIO,
    PLATE_REFINE_WORK_SIZE,
)

LOGGER = logging.getLogger(__name__)

# cv2.minAreaRect biçimi: ((cx, cy), (w, h), açı)
RotatedRect = Tuple[Tuple[float, float], Tuple[float, float], float]


# =============================================================================
# ORTAK YARDIMCILAR
# =============================================================================

def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Ölçek anlamlı ölçüde 1'den büyükse görüntüyü INTER_CUBIC ile büyütür."""
    if scale > 1.01:
        height, width = image.shape[:2]
        return cv2.resize(
            image, (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    return image


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Gri görüntüye yerel kontrast dengeleme (CLAHE) uygular."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _normalize_rect(rect: Any) -> RotatedRect:
    """minAreaRect ciktisini 'uzun kenar yatay' olacak sekilde duzenler."""
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h, angle = h, w, angle + 90.0
    return (cx, cy), (w, h), angle


def _warp_rect(image: np.ndarray, rect: RotatedRect) -> Optional[np.ndarray]:
    """Egik dikdortgeni perspektif donusumu ile duz bir kesite cevirir."""
    (cx, cy), (w, h), angle = rect
    out_w = int(round(w * PLATE_REFINE_MARGIN_X))
    out_h = int(round(h * PLATE_REFINE_MARGIN_Y))
    if out_w < PLATE_MIN_CROP_WIDTH or out_h < 8:
        return None
    points = cv2.boxPoints(((cx, cy), (out_w, out_h), angle)).astype(np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    source = np.array(
        [points[np.argmin(total)], points[np.argmin(diff)],
         points[np.argmax(total)], points[np.argmax(diff)]],
        dtype=np.float32,
    )
    target = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


# =============================================================================
# BÖLGE İYİLEŞTİRME STRATEJİLERİ
# =============================================================================

def _refine_by_plate_body(work: np.ndarray) -> Optional[RotatedRect]:
    """
    Birinci strateji: parlak plaka govdesini tek parca olarak yakala.

    TR plakasi acik zeminlidir, karakterler koyudur. Otsu esiklemesinden
    sonra RETR_EXTERNAL zaten plakanin DIS hattini verir; karakter
    bosluklarini kapatmak icin morfolojik kapama GEREKMEZ. Eski surumdeki
    9x3'luk kapama cekirdegi, 93 piksellik bir kesitte plakayi yanindaki
    krom gril ile birlestirip tek bir 92x73 blob uretiyor ve bu blob oran
    suzgecine takildigi icin geriye yalnizca kosedeki bir kirinti kaliyordu.
    """
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)   # kenarlari koruyan yumusatma
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = work.shape[:2]
    crop_area = float(width * height)
    best_rect, best_score = None, 0.0
    for contour in contours:
        rect = _normalize_rect(cv2.minAreaRect(contour))
        (cx, cy), (w, h), _ = rect
        if h < 6 or w < PLATE_MIN_CROP_WIDTH:
            continue
        ratio = w / h
        if not PLATE_REFINE_MIN_RATIO <= ratio <= PLATE_REFINE_MAX_RATIO:
            continue
        rect_area = w * h
        area_frac = rect_area / crop_area
        if not PLATE_REFINE_MIN_AREA_FRAC <= area_frac <= PLATE_REFINE_MAX_AREA_FRAC:
            continue
        # Dolgu orani: plaka govdesi ICI DOLU bir dikdortgendir. Gril
        # yansimasi / tampon seridi gibi parcali bolgeler burada elenir.
        fill = cv2.contourArea(contour) / max(rect_area, 1.0)
        if fill < PLATE_REFINE_MIN_FILL:
            continue
        # Merkez, kesidin orta bandinda olmali: dedektor plakayi ortaladigi
        # icin kose kirintilari boylece elenir (eski hatanin ta kendisi).
        band = PLATE_REFINE_CENTER_BAND
        if not (band * width <= cx <= (1.0 - band) * width
                and band * height <= cy <= (1.0 - band) * height):
            continue
        score = rect_area * fill * (
            1.0 - abs(ratio - PLATE_REFINE_IDEAL_RATIO) / PLATE_REFINE_IDEAL_RATIO
        )
        if score > best_score:
            best_score, best_rect = score, rect
    return best_rect


def _refine_by_characters(work: np.ndarray) -> Optional[RotatedRect]:
    """
    Ikinci strateji: plaka govdesi yakalanamazsa KARAKTERLERDEN yola cik.

    Koyu karakter lekelerinin merkezlerine bir dogru oturtulur; bu dogrunun
    egimi plakanin egim acisini verir. Lekeler bu aciya gore dondurulup
    sinirlari olculerek plaka dikdortgeni yeniden insa edilir. Plaka zemini
    kirli / golgeli oldugunda govde esiklemesi calismaz ama karakterler
    genelde hala ayirt edilebilir.
    """
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = _apply_clahe(gray)
    inverted = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 12,
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(inverted, 8)

    height, width = work.shape[:2]
    blobs: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if bh < 0.10 * min(height, width) or bh > 0.75 * height:
            continue
        if bw < 2 or bw > 0.35 * width:
            continue
        if not 0.9 <= bh / max(bw, 1) <= 5.0:      # karakterler dik dikdortgen
            continue
        if area / float(max(bw * bh, 1)) < 0.20:
            continue
        blobs.append((centroids[index], (x, y, bw, bh)))

    if len(blobs) < 3:                              # plaka en az 7 karakter
        return None

    points = np.array([c for c, _ in blobs], dtype=np.float32)
    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    angle = float(np.degrees(np.arctan2(vy, vx)))
    if abs(angle) > 60.0:                           # dikey dizilim -> plaka degil
        return None

    rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    corners: List[Tuple[float, float]] = []
    for _, (x, y, bw, bh) in blobs:
        for px, py in ((x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)):
            corners.append((rotation[0, 0] * px + rotation[0, 1] * py + rotation[0, 2],
                            rotation[1, 0] * px + rotation[1, 1] * py + rotation[1, 2]))
    array = np.array(corners)
    x1, y1 = float(array[:, 0].min()), float(array[:, 1].min())
    x2, y2 = float(array[:, 0].max()), float(array[:, 1].max())
    rect_w, rect_h = x2 - x1, y2 - y1
    if rect_h < 6 or rect_w < PLATE_MIN_CROP_WIDTH:
        return None
    if not PLATE_REFINE_MIN_RATIO <= rect_w / rect_h <= PLATE_REFINE_MAX_RATIO:
        return None

    # Merkezi dondurulmemis koordinat sistemine geri tasi
    inverse = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle, 1.0)
    mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx = inverse[0, 0] * mid_x + inverse[0, 1] * mid_y + inverse[0, 2]
    cy = inverse[1, 0] * mid_x + inverse[1, 1] * mid_y + inverse[1, 2]
    return (cx, cy), (rect_w, rect_h), angle


# =============================================================================
# GENEL API
# =============================================================================

def refine_plate_region(plate_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Dedektor kutusunun icindeki gercek plaka alanini izole eder ve
    perspektif donusumu ile yatay hale getirir.

    Dedektor kutusu EKSENE HIZALIDIR. Plaka kadrajda egik durdugunda
    (donen arac, yandan bakan kamera) kutu neredeyse kare gelir ve icinde
    plakanin yanisira gril/tampon parcalari kalir. Bu fonksiyon plakayi o
    kutunun icinden cekip cikarir.

    Iki strateji sirayla denenir:
      1) Parlak plaka govdesini tek parca olarak yakalamak
      2) Koyu karakter lekelerinden plaka eksenini kestirmek

    Ikisi de guvenilir bir bolge bulamazsa None doner ve cagiran taraf
    ORIJINAL kesitle devam eder. Bu geri dusus kritiktir: yanlis secilmis
    bir bolge, saglam bir kesitten cok daha kotudur.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    try:
        height, width = plate_crop.shape[:2]
        # Kucuk kesitlerde konturlar birkac piksele sikisir; once buyut.
        scale = max(1.0, PLATE_REFINE_WORK_SIZE / float(max(width, height)))
        work = _upscale(plate_crop, scale)

        for strategy in (_refine_by_plate_body, _refine_by_characters):
            rect = strategy(work)
            if rect is None:
                continue
            refined = _warp_rect(work, _normalize_rect(rect))
            if refined is not None and refined.size > 0:
                return refined
        return None
    except Exception as exc:
        LOGGER.debug("Plaka bolgesi iyilestirilemedi: %s", exc)
        return None


def preprocess_plate(plate_crop: np.ndarray) -> np.ndarray:
    """
    OCR oncesi plaka kesitini iyilestirir.

    Olcekleme hem genislige hem yukseklige bakar. Bu onemlidir: dedektor
    kutusu kare geldiginde yalnizca yukseklige bakan bir esik devreye
    girmez, metin 12-15 piksel kalir ve PaddleOCR'in metin tespit modeli
    hicbir sey bulamaz.
    """
    height, width = plate_crop.shape[:2]

    scale = max(
        PLATE_OCR_MIN_WIDTH / max(width, 1),
        PLATE_OCR_TARGET_HEIGHT / max(height, 1),
        1.0,
    )
    scale = min(scale, PLATE_OCR_MAX_UPSCALE)
    image = _upscale(plate_crop, scale)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _apply_clahe(gray)
    # Buyutme sonrasi yumusayan karakter kenarlarini keskinlestir
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gray = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
