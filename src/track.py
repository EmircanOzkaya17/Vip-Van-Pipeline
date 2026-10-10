"""
Track veri modeli: tek bir aracın (track_id) yaşam döngüsü boyunca biriken
tüm bilgileri tutar ve rapor/CSV için düz kayda dönüştürür.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .bbox import Box

# CSV / rapor kolonları; to_record() anahtarlarıyla aynı sırada tutulur.
RECORD_FIELDS: Tuple[str, ...] = (
    "track_id", "vehicle_class", "vehicle_type",
    "final_positive", "final_negative", "final_margin",
    "plate_text", "ocr_confidence",
    "first_frame", "cross_frame", "last_frame", "bbox",
    # Tani kolonlari: plaka okunamadiginda sebebi CSV'den de gorulebilsin
    "decision", "plate_attempts", "plate_candidates", "plate_stages",
)


def _round_or_blank(value: Optional[float], digits: int = 4) -> Any:
    """Değer varsa yuvarlar; yoksa CSV için boş metin döner."""
    return round(value, digits) if value is not None else ""


@dataclass
class TrackInfo:
    """
    Tek bir aracın (track_id) tüm yaşam döngüsü bilgilerini tutar.
    Spesifikasyonda istenen tüm alanlar bu yapıda saklanır.
    """

    # --- Kimlik ve konum ---
    track_id: int
    first_frame: int
    last_frame: int
    bbox: Box = (0, 0, 0, 0)
    detection_conf: float = 0.0

    # --- Araç sınıfı (7 frame oylaması) ---
    class_votes: Counter = field(default_factory=Counter)
    vehicle_class: Optional[str] = None

    # --- Çizgi geçişi ---
    previous_center: Optional[Tuple[float, float]] = None
    cross_frame: Optional[int] = None

    # --- OpenCLIP skorları ---
    clip_frame_scores: List[Dict[str, float]] = field(default_factory=list)
    processed_clip_offsets: Set[int] = field(default_factory=set)
    final_positive: Optional[float] = None
    final_negative: Optional[float] = None
    final_margin: Optional[float] = None
    vehicle_type: Optional[str] = None
    is_target: Optional[bool] = None       # None = karar bekliyor
    decision_offset: Optional[int] = None  # Kararin verildigi gecis sonrasi kare
    decided_on_close: bool = False         # Karar track kapanirken mi verildi?

    # --- Plaka ---
    plate_buffer: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    plate_candidates: List[Tuple[str, float]] = field(default_factory=list)
    plate_attempts: int = 0
    plate_stages: Counter = field(default_factory=Counter)  # Asama -> deneme sayisi
    last_plate_offset: Optional[int] = None  # Son plaka denemesi/tamponlama karesi
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    plate_finalized: bool = False

    # --- Yardımcı ---
    last_crop: Optional[np.ndarray] = None
    last_crop_offset: Optional[int] = None
    archived: bool = False

    # ------------------------------------------------------------------ #
    @property
    def age(self) -> int:
        """Track'in kaç frame'dir yaşadığını döner."""
        return self.last_frame - self.first_frame + 1

    def offset_after_cross(self, frame_index: int) -> Optional[int]:
        """Çizgi geçişinden bu yana geçen frame sayısını döner."""
        if self.cross_frame is None:
            return None
        return frame_index - self.cross_frame

    def release_buffers(self) -> None:
        """Bellek tasarrufu için geçici görüntü tamponlarını temizler."""
        self.plate_buffer.clear()
        self.last_crop = None

    def decision_label(self) -> str:
        """Siniflandirma kararinin ne zaman verildigini kisa metinle doner."""
        if self.is_target is None:
            return "-"
        if self.decided_on_close:
            return "kapanis"
        if self.decision_offset is None:
            return "?"
        return f"+{self.decision_offset}"

    def plate_stage_summary(self) -> str:
        """Plaka asama sayaclarini 'asama:sayi' biciminde tek satira indirger."""
        return " ".join(f"{stage}:{count}"
                        for stage, count in self.plate_stages.most_common())

    def to_record(self) -> Dict[str, Any]:
        """CSV / rapor için düz bir sözlük üretir (anahtarlar: RECORD_FIELDS)."""
        return {
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class or "unknown",
            "vehicle_type": self.vehicle_type or "unclassified",
            "final_positive": _round_or_blank(self.final_positive),
            "final_negative": _round_or_blank(self.final_negative),
            "final_margin": _round_or_blank(self.final_margin),
            "plate_text": self.plate_text or "",
            "ocr_confidence": _round_or_blank(self.ocr_confidence),
            "first_frame": self.first_frame,
            "cross_frame": self.cross_frame if self.cross_frame is not None else "",
            "last_frame": self.last_frame,
            "bbox": "{},{},{},{}".format(*self.bbox),
            "decision": self.decision_label(),
            "plate_attempts": self.plate_attempts,
            "plate_candidates": " ".join(text for text, _ in self.plate_candidates),
            "plate_stages": self.plate_stage_summary(),
        }
