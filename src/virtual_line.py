"""
Sanal tetik çizgisi (line crossing) geometrisi.

Aracın bounding box merkezinin çizgiyi hangi frame'de ve hangi yönde geçtiğini
işaretli alan (cross product) yöntemiyle tespit eder. Bu modül yalnızca
geometri içerir; çizim `visualization.draw_virtual_line` içindedir.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .config import (
    LINE_AUTO_ORIENT,
    LINE_CROSS_DIRECTION,
    LINE_EXPECTED_TRAVEL_DIRECTION,
    LINE_SEGMENT_EXTENSION,
)

LOGGER = logging.getLogger(__name__)

Point = Tuple[float, float]


class VirtualLine:
    """
    Kadrajdaki mavi sanal çizgiyi temsil eder.

    Aracın bounding box merkezinin çizgiyi hangi frame'de geçtiğini,
    işaretli alan (cross product) yöntemiyle tespit eder.
    """

    def __init__(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        reference_resolution: Optional[Tuple[int, int]] = None,
        extension: float = LINE_SEGMENT_EXTENSION,
        direction: str = LINE_CROSS_DIRECTION,
        travel_direction: Tuple[float, float] = LINE_EXPECTED_TRAVEL_DIRECTION,
        auto_orient: bool = LINE_AUTO_ORIENT,
    ) -> None:
        self._ref_start = (float(start[0]), float(start[1]))
        self._ref_end = (float(end[0]), float(end[1]))
        self._ref_resolution = reference_resolution
        self.extension = extension
        self.direction = direction
        self.travel_direction = travel_direction
        self.auto_orient = auto_orient
        # Ölçeklenmiş (gerçek) koordinatlar - fit_to_frame() ile atanır
        self.start = self._ref_start
        self.end = self._ref_end
        self._auto_orient()

    def fit_to_frame(self, width: int, height: int) -> None:
        """
        Referans çözünürlüğe göre tanımlanmış koordinatları gerçek video
        çözünürlüğüne ölçekler. Referans verilmemişse koordinatlar aynen kullanılır.
        """
        if self._ref_resolution:
            ref_w, ref_h = self._ref_resolution
            sx, sy = width / float(ref_w), height / float(ref_h)
            self.start = (self._ref_start[0] * sx, self._ref_start[1] * sy)
            self.end = (self._ref_end[0] * sx, self._ref_end[1] * sy)
        else:
            self.start, self.end = self._ref_start, self._ref_end
        self._auto_orient()
        LOGGER.info(
            "Sanal cizgi: (%.0f, %.0f) -> (%.0f, %.0f)",
            self.start[0], self.start[1], self.end[0], self.end[1],
        )

    # ------------------------------------------------------------------ #
    def _auto_orient(self) -> None:
        """
        Çizginin uç nokta sırasını, aracın beklenen ilerleme yönünde hareket
        ettiğinde işaretin negatiften pozitife geçeceği şekilde normalize eder.

        signed_side fonksiyonunun bir nokta hareket ettikçe nasıl değiştiği
        (-dy, dx) vektörü ile belirlenir. Bu vektörün beklenen hareket yönüyle
        skaler çarpımı negatifse uç noktalar yer değiştirilir. Böylece kullanıcı
        çizgiyi hangi uçtan çizerse çizsin "soldan sağa geçiş" doğru algılanır.
        """
        if not self.auto_orient:
            return
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        vx, vy = self.travel_direction
        if (-dy * vx + dx * vy) < 0:
            self.start, self.end = self.end, self.start
            self._ref_start, self._ref_end = self._ref_end, self._ref_start
            LOGGER.debug("Sanal cizgi uc noktalari yon icin ters cevrildi.")

    # ------------------------------------------------------------------ #
    def signed_side(self, point: Point) -> float:
        """
        Noktanın çizginin hangi tarafında olduğunu veren işaretli çapraz çarpım.
        Negatif = çizginin solu (araç giriş tarafı), pozitif = sağı.
        """
        (sx, sy), (ex, ey) = self.start, self.end
        dx, dy = ex - sx, ey - sy
        px, py = point[0] - sx, point[1] - sy
        return dx * py - dy * px

    def projection_t(self, point: Point) -> float:
        """Noktanın çizgi üzerindeki normalize izdüşüm parametresini (t) verir."""
        (sx, sy), (ex, ey) = self.start, self.end
        dx, dy = ex - sx, ey - sy
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return 0.0
        return ((point[0] - sx) * dx + (point[1] - sy) * dy) / denom

    def _direction_ok(self, previous_side: float, current_side: float) -> bool:
        """Geçişin istenen yönde olup olmadığını kontrol eder."""
        if self.direction == "neg_to_pos":
            return previous_side < 0.0 <= current_side
        if self.direction == "pos_to_neg":
            return previous_side > 0.0 >= current_side
        return (previous_side < 0.0) != (current_side < 0.0)

    def is_crossed(self, previous_point: Optional[Point], current_point: Point) -> bool:
        """
        İki ardışık merkez noktası arasında çizgi geçişi olup olmadığını döner.
        Segment dışında kalan (çok yukarı/aşağı) geçişler elenir.
        """
        if previous_point is None:
            return False
        previous_side = self.signed_side(previous_point)
        current_side = self.signed_side(current_point)
        if not self._direction_ok(previous_side, current_side):
            return False
        t = self.projection_t(current_point)
        return -self.extension <= t <= 1.0 + self.extension
