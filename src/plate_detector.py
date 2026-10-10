"""
Plaka tespiti: YOLOv9-t plaka dedektörü + geometrik akıl süzgeci.

İki çalıştırma yolu vardır:
  * OpenCV DNN (varsayılan): models/ klasöründeki NMS'i ayrılmış model.
    onnxruntime / open_image_models GEREKMEZ.
  * onnxruntime: open_image_models paketi (eski yol, isteğe bağlı).

İkisi de aynı ağırlıkları kullanır; çıktıları birebir aynıdır (doğrulandı).
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .bbox import Box, clamp_box
from .config import (
    PLATE_BOX_MAX_RATIO,
    PLATE_BOX_MAX_WIDTH_FRAC,
    PLATE_BOX_MIN_RATIO,
    PLATE_BOX_MIN_WIDTH_FRAC,
    PLATE_BOX_MIN_Y_FRAC,
    PLATE_DETECTION_CONF,
    PLATE_DETECTION_NMS_IOU,
    PLATE_DETECTOR_BACKEND,
    PLATE_DETECTOR_INPUT_SIZE,
    PLATE_DETECTOR_MODEL,
    PLATE_DETECTOR_ONNX,
    PLATE_DETECT_ZOOM_ON_MISS,
    PLATE_MAX_CANDIDATES,
    PLATE_ZOOM_TILE_FRAC,
    PLATE_ZOOM_TILES,
    PLATE_ZOOM_TOP_FRAC,
)
from .dnn_utils import load_onnx_net

LOGGER = logging.getLogger(__name__)

# [((x1, y1, x2, y2), confidence), ...]
PlateDetections = List[Tuple[Box, float]]
RawDetections = List[Tuple[Tuple[float, float, float, float], float]]


def check_plate_box_geometry(
    box: Box, vehicle_width: int, vehicle_height: int
) -> Tuple[bool, str]:
    """
    Bir dedektor kutusunun fiziksel olarak plaka OLABILIR mi oldugunu soyler.

    Dedektor jant, far, tampon araligi gibi bolgeleri de yuksek guvenle
    plaka isaretleyebiliyor. Bu kutular OCR'a girdiginde yalnizca gurultu
    uretir. Asagidaki uc kriter, plakanin degismez fiziksel ozelliklerine
    dayanir ve modelden bagimsizdir.

    Donus: (gecti_mi, elenme_sebebi)
    """
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return False, "gecersiz kutu"

    # 1) En/boy orani: egik plaka bile eksene hizali kutuda >= 1.0 verir.
    ratio = box_w / float(box_h)
    if not PLATE_BOX_MIN_RATIO <= ratio <= PLATE_BOX_MAX_RATIO:
        return False, f"oran {ratio:.2f}"

    # 2) Arac kutusuna gore genislik payi: plaka aracin yaklasik 1/4'u kadar
    #    genistir; jant/far cok daha kucuk, on cam cok daha buyuk kalir.
    width_frac = box_w / float(max(vehicle_width, 1))
    if not PLATE_BOX_MIN_WIDTH_FRAC <= width_frac <= PLATE_BOX_MAX_WIDTH_FRAC:
        return False, f"genislik payi {width_frac:.3f}"

    # 3) Dikey konum: plaka aracin ust ucte birinde bulunmaz.
    center_y_frac = ((y1 + y2) / 2.0) / float(max(vehicle_height, 1))
    if center_y_frac < PLATE_BOX_MIN_Y_FRAC:
        return False, f"cok yukarida ({center_y_frac:.2f})"

    return True, ""


# =============================================================================
# OPENCV DNN YOLU (VARSAYILAN)
# =============================================================================

def _letterbox(
    image: np.ndarray, size: int, color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    open-image-models'in YOLOv9 on islemesinin birebir aynisi: en/boy oranini
    koruyarak olcekle, kalan alani gri ile doldur.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    pad_w, pad_h = (size - new_w) / 2.0, (size - new_h) / 2.0
    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (pad_w, pad_h)


class _OpenCVYoloPlateModel:
    """NMS'i ayrilmis YOLOv9 plaka modelini OpenCV DNN ile calistirir."""

    name = "opencv-dnn"

    def __init__(self, model_path: str, input_size: int) -> None:
        self.net = load_onnx_net(model_path, "plaka dedektoru")
        self.input_size = input_size
        # Saglamlik testi: bos goruntude sonlu cikti uretmeli
        probe = self._forward(np.full((input_size, input_size, 3), 114, np.uint8))
        if not np.isfinite(probe).all():
            raise RuntimeError("Plaka dedektoru OpenCV'de gecersiz (NaN) cikti uretti")

    def _forward(self, letterboxed_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            letterboxed_bgr, scalefactor=1.0 / 255.0, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()                  # (1, 4 + sinif, N)
        return output[0].T                           # (N, 4 + sinif)

    def predict(self, image: np.ndarray) -> RawDetections:
        letterboxed, ratio, (pad_w, pad_h) = _letterbox(image, self.input_size)
        rows = self._forward(letterboxed)
        scores = rows[:, 4:].max(axis=1)
        keep = scores >= PLATE_DETECTION_CONF
        if not keep.any():
            return []
        rows, scores = rows[keep], scores[keep]
        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        boxes_xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(),
            PLATE_DETECTION_CONF, PLATE_DETECTION_NMS_IOU,
        )
        results: RawDetections = []
        for index in np.array(indices).reshape(-1):
            x, y, bw, bh = boxes_xywh[index]
            results.append((
                (float((x - pad_w) / ratio), float((y - pad_h) / ratio),
                 float((x + bw - pad_w) / ratio), float((y + bh - pad_h) / ratio)),
                float(scores[index]),
            ))
        return results


# =============================================================================
# ONNXRUNTIME YOLU (open_image_models, ISTEGE BAGLI)
# =============================================================================

def _extract_detection(det: Any) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
    """
    Farklı kütüphane sürümlerinin tespit nesnelerinden kutu ve güven
    değerini güvenli biçimde çıkarır.
    """
    try:
        bounding_box = getattr(det, "bounding_box", None)
        if bounding_box is not None:
            box = (
                float(bounding_box.x1), float(bounding_box.y1),
                float(bounding_box.x2), float(bounding_box.y2),
            )
            return box, float(getattr(det, "confidence", 1.0))
        if isinstance(det, dict):
            raw_box = det.get("bounding_box") or det.get("box") or det.get("bbox")
            if raw_box is not None:
                box = tuple(float(v) for v in raw_box)  # type: ignore[assignment]
                return box, float(det.get("confidence", 1.0))  # type: ignore[return-value]
        if isinstance(det, (list, tuple)) and len(det) >= 4:
            box = tuple(float(v) for v in det[:4])  # type: ignore[assignment]
            conf = float(det[4]) if len(det) > 4 else 1.0
            return box, conf  # type: ignore[return-value]
    except Exception:
        pass
    return None, 0.0


def _onnxruntime_import_error() -> Optional[str]:
    """onnxruntime yuklenemiyorsa GERCEK sebebi doner (yuklenebiliyorsa None)."""
    try:
        import onnxruntime  # noqa: F401
        return None
    except Exception as exc:                     # ImportError, OSError (DLL) ...
        return f"{type(exc).__name__}: {exc}"


class _OnnxRuntimePlateModel:
    """open_image_models (onnxruntime) uzerinden ayni modeli calistirir."""

    name = "onnxruntime"

    def __init__(self, detector_model: str) -> None:
        # Eskiden burada her hata "ONNX destegi olmadan kurulmus" diye
        # raporlaniyordu. Artik onnxruntime'in kendi hatasi aynen gosterilir.
        ort_error = _onnxruntime_import_error()
        if ort_error:
            raise RuntimeError(
                f"onnxruntime yuklenemedi -> {ort_error}. "
                "Not: varsayilan OpenCV yolu onnxruntime gerektirmez "
                "(models/plate_det_yolov9t384_raw.onnx mevcut olmali)."
            )
        try:
            from open_image_models import create_detector  # type: ignore
            self.model = create_detector(detector_model)
        except ImportError:
            from open_image_models import LicensePlateDetector  # type: ignore
            self.model = LicensePlateDetector(detection_model=detector_model)

    def predict(self, image: np.ndarray) -> RawDetections:
        results: RawDetections = []
        for det in self.model.predict(image) or []:
            box, conf = _extract_detection(det)
            if box is not None:
                results.append((box, conf))
        return results


def _create_backend(backend: str, detector_model: str, onnx_path: str):
    """Ayarlanan yola gore dedektor modelini kurar."""
    backend = (backend or "auto").lower()
    if backend in ("auto", "opencv"):
        if os.path.isfile(onnx_path):
            try:
                return _OpenCVYoloPlateModel(onnx_path, PLATE_DETECTOR_INPUT_SIZE)
            except Exception as exc:
                if backend == "opencv":
                    raise RuntimeError(f"OpenCV plaka dedektoru yuklenemedi: {exc}") from exc
                LOGGER.warning("OpenCV plaka dedektoru yuklenemedi (%s); "
                               "onnxruntime deneniyor.", exc)
        elif backend == "opencv":
            raise RuntimeError(f"Plaka dedektoru model dosyasi yok: {onnx_path}")
        else:
            LOGGER.warning("Model dosyasi yok (%s); onnxruntime deneniyor.", onnx_path)
    return _OnnxRuntimePlateModel(detector_model)


class PlateDetector:
    """
    Araç kesiti içinde plaka kutularını bulan dedektör.

    Ham tespitler `check_plate_box_geometry` süzgecinden geçirilir; elenen kutu
    sayısı tanı amaçlı `rejected_boxes` sayacında tutulur.
    """

    def __init__(
        self,
        model_name: str = PLATE_DETECTOR_MODEL,
        backend: str = PLATE_DETECTOR_BACKEND,
        onnx_path: str = PLATE_DETECTOR_ONNX,
    ) -> None:
        self.model = _create_backend(backend, model_name, onnx_path)
        self.backend_name = self.model.name
        self.rejected_boxes = 0              # Geometri suzgecinin eledigi kutu sayisi
        self.zoom_hits = 0                   # Plakayi yalnizca 2. gecisin buldugu deneme
        LOGGER.info("Plaka dedektoru hazir (%s).", self.backend_name)

    # ------------------------------------------------------------------ #
    def detect(self, vehicle_crop: np.ndarray) -> PlateDetections:
        """
        Araç kesiti içinde plaka kutularını tespit eder.

        Ham tespitler geometrik akil suzgecinden gecirilir; plaka olmasi fiziksel
        olarak mumkun olmayan kutular (jant, far, tampon seridi) elenir. Geriye
        kalanlardan yalnizca en guvenilir PLATE_MAX_CANDIDATES tanesi dondurulur.

        Dönüş: [((x1, y1, x2, y2), confidence), ...]
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []
        results = self._filter(self._predict(vehicle_crop), vehicle_crop)
        if not results and PLATE_DETECT_ZOOM_ON_MISS:
            results = self._filter(self._predict_zoomed(vehicle_crop), vehicle_crop)
            if results:
                self.zoom_hits += 1
        return results

    # ------------------------------------------------------------------ #
    def _predict(self, image: np.ndarray) -> RawDetections:
        try:
            return self.model.predict(image)
        except Exception as exc:
            LOGGER.debug("Plaka tespit hatasi: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    def _predict_zoomed(self, vehicle_crop: np.ndarray) -> RawDetections:
        """
        Aracin alt bolgesini ust uste binen dosemelere bolup her birini ayri
        tarar; kutulari arac koordinatlarina geri tasir ve NMS ile birlestirir.
        """
        height, width = vehicle_crop.shape[:2]
        top = int(height * PLATE_ZOOM_TOP_FRAC)
        tile_w = max(1, int(width * PLATE_ZOOM_TILE_FRAC))
        if PLATE_ZOOM_TILES <= 1 or tile_w >= width:
            starts = [0]
        else:
            step = (width - tile_w) / float(PLATE_ZOOM_TILES - 1)
            starts = [int(round(i * step)) for i in range(PLATE_ZOOM_TILES)]

        merged: RawDetections = []
        edge = 3                                     # px: kesim kenarina temas payi
        for x0 in starts:
            tile = vehicle_crop[top:height, x0:x0 + tile_w]
            if tile.size == 0:
                continue
            tile_right = tile.shape[1]
            for (x1, y1, x2, y2), conf in self._predict(tile):
                # Doseme sinirinda kesilmis YARIM plakalari at: plaka zaten
                # %50 ust uste binen komsu dosemede butun olarak gorunur.
                cut_left = x0 > 0 and x1 <= edge
                cut_right = x0 + tile_right < width and x2 >= tile_right - edge
                if cut_left or cut_right:
                    continue
                merged.append(((x1 + x0, y1 + top, x2 + x0, y2 + top), conf))
        if len(merged) <= 1:
            return merged
        boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b, _ in merged]
        keep = cv2.dnn.NMSBoxes(boxes, [c for _, c in merged],
                                PLATE_DETECTION_CONF, PLATE_DETECTION_NMS_IOU)
        return [merged[i] for i in np.array(keep).reshape(-1)]

    # ------------------------------------------------------------------ #
    def _filter(self, detections: RawDetections, vehicle_crop: np.ndarray) -> PlateDetections:
        """Guven + geometri suzgeci; en guvenilir N kutuyu doner."""
        height, width = vehicle_crop.shape[:2]
        results: PlateDetections = []
        for box, conf in detections:
            if conf < PLATE_DETECTION_CONF:
                continue
            clamped = clamp_box(box, width, height)
            accepted, reason = check_plate_box_geometry(clamped, width, height)
            if not accepted:
                self.rejected_boxes += 1
                LOGGER.debug("Plaka kutusu elendi (%s): %s conf=%.2f",
                             reason, clamped, conf)
                continue
            results.append((clamped, conf))

        # En güvenilir plakadan başlayacak şekilde sırala ve tavani uygula
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:PLATE_MAX_CANDIDATES]
