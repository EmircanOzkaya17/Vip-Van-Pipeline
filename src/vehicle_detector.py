"""
Araç tespiti ve takibi: Ultralytics YOLO + dahili BoT-SORT takipçisi.

ultralytics, yalnızca bu sınıf oluşturulduğunda içe aktarılır (lazy import).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .bbox import clamp_box
from .config import (
    COCO_VEHICLE_CLASSES,
    MIN_BOX_AREA,
    TRACKER_CONFIG,
    YOLO_CONF_THRESHOLD,
    YOLO_HALF_PRECISION,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL_PATH,
)

LOGGER = logging.getLogger(__name__)


class VehicleDetector:
    """
    YOLO11m + BoT-SORT sarmalayıcısı.

    Ultralytics'in dahili BoT-SORT takipçisini `persist=True` ile kullanır;
    böylece aynı araç kadrajda kaldığı sürece track_id korunur.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        tracker_config: str = TRACKER_CONFIG,
        device: Optional[str] = None,
        imgsz: int = YOLO_IMAGE_SIZE,
        conf: float = YOLO_CONF_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
    ) -> None:
        try:
            from ultralytics import YOLO  # Lazy import: net hata mesajı için
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ultralytics kurulu degil. Kurulum: pip install ultralytics"
            ) from exc

        LOGGER.info("YOLO modeli yukleniyor: %s", model_path)
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"YOLO modeli yuklenemedi ({model_path}): {exc}") from exc

        self.tracker_config = tracker_config
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.class_ids = list(COCO_VEHICLE_CLASSES.keys())
        self.half = YOLO_HALF_PRECISION and bool(device and "cpu" not in str(device))

    # ------------------------------------------------------------------ #
    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tek bir frame üzerinde tespit + takip yapar.

        Dönüş: [{track_id, bbox, class_name, conf}, ...]
        Hata durumunda boş liste döner (akış kesilmez).
        """
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                classes=self.class_ids,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.warning("Tespit/takip hatasi: %s", exc)
            return []

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            clss = boxes.cls.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy()
        except Exception as exc:
            LOGGER.warning("Tespit sonuclari okunamadi: %s", exc)
            return []

        height, width = frame.shape[:2]
        detections: List[Dict[str, Any]] = []
        for box, track_id, cls_id, conf in zip(xyxy, ids, clss, confs):
            class_name = COCO_VEHICLE_CLASSES.get(int(cls_id))
            if class_name is None:          # İstenmeyen sınıflar göz ardı edilir
                continue
            bbox = clamp_box(box, width, height)
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < MIN_BOX_AREA:
                continue
            detections.append(
                {
                    "track_id": int(track_id),
                    "bbox": bbox,
                    "class_name": class_name,
                    "conf": float(conf),
                }
            )
        return detections
