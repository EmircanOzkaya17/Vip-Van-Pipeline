"""
Plaka okuma zinciri:
Detection -> Crop -> Region refinement -> OCR -> Regex/Düzeltme -> Confidence Check.

Varsayilan motorlar yalnizca OpenCV DNN kullanir (bkz. plate_detector,
plate_ocr_fast); PaddleOCR istege bagli ikinci motordur.

Her deneme, zincirin nerede durduğunu anlatan bir `PlateAttempt` döndürür.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .bbox import crop_image
from .config import (
    OCR_FORMAT_VALID_MIN_CONF,
    OCR_MIN_CONFIDENCE,
    PLATE_CROP_PADDING,
    PLATE_DETECTOR_MODEL,
    PLATE_MIN_CROP_WIDTH,
    PLATE_OCR_ENGINES,
    PLATE_OCR_RECOGNIZE_ON_REGEX_FAIL,
    PLATE_OCR_TRY_BOTH_CROPS,
    PLATE_REFINE_REGION,
)
from .plate_detector import PlateDetector
from .plate_image import preprocess_plate, refine_plate_region
from .plate_ocr_fast import FastPlateOCR, OcrResult
from .plate_text import correct_and_validate_plate

LOGGER = logging.getLogger(__name__)


@dataclass
class PlateAttempt:
    """
    Tek bir plaka okuma denemesinin tani (diagnostic) sonucu.

    Plaka okunamadiginda sorunun HANGI asamada oldugunu gorebilmek icin
    zincirin her adimi kaydedilir. `stage` degeri asagidakilerden biridir:
      no_plate_detected : Dedektor arac kesitinde plaka bulamadi
      plate_too_small   : Plaka bulundu ama OCR icin fazla dusuk cozunurluklu
      no_ocr_text       : OCR hicbir metin dondurmedi
      low_confidence    : Metin var ama guven esigin altinda
      regex_failed      : Metin var ama gecerli plaka formatina oturmadi
      ok                : Basarili
    """

    stage: str = "no_plate_detected"
    plate_count: int = 0                       # Bulunan plaka kutusu sayisi
    best_plate_width: int = 0                  # En genis plaka kutusunun eni (px)
    raw_texts: List[Tuple[str, float]] = field(default_factory=list)
    plate_text: Optional[str] = None
    confidence: Optional[float] = None

    @property
    def ok(self) -> bool:
        """Denemenin gecerli bir plaka uretip uretmedigini doner."""
        return self.plate_text is not None

    def describe(self) -> str:
        """Log icin tek satirlik ozet uretir."""
        parts = [f"asama={self.stage}", f"plaka_kutusu={self.plate_count}"]
        if self.best_plate_width:
            parts.append(f"genislik={self.best_plate_width}px")
        if self.raw_texts:
            preview = ", ".join(f"'{t}'({c:.2f})" for t, c in self.raw_texts[:3])
            parts.append(f"ocr=[{preview}]")
        if self.plate_text:
            parts.append(f"-> {self.plate_text} ({self.confidence:.2f})")
        return " | ".join(parts)


class PlateReader:
    """
    Plaka dedektörü + plakaya özel OCR (isteğe bağlı PaddleOCR) birleşimi.

    Akış: Plate Detection -> Plate Crop -> (Deskew) -> OCR -> Regex/Düzeltme ->
          Confidence Check

    OCR motorlari PLATE_OCR_ENGINES sirasiyla denenir; bir motor gecerli plaka
    verirse sonrakiler o kesit icin calistirilmaz. Bir motor yuklenemezse
    digerleriyle devam edilir; hicbiri yuklenemezse RuntimeError atilir.

    debug_dir verilirse her denemenin arac kesiti ve plaka kirpintisi diske
    yazilir; boylece plaka okunamadiginda sorunun gorsel kaynagi incelenebilir.
    """

    def __init__(
        self,
        detector_model: str = PLATE_DETECTOR_MODEL,
        debug_dir: Optional[str] = None,
        engines: Tuple[str, ...] = PLATE_OCR_ENGINES,
    ) -> None:
        # Yükleme sırası korunur: önce dedektör, ardından OCR.
        self.detector = PlateDetector(detector_model)
        self.engines: List[Tuple[str, object]] = []
        errors: List[str] = []
        for name in engines:
            try:
                if name == "fastplate":
                    self.engines.append((name, FastPlateOCR()))
                elif name == "paddle":
                    from .plate_ocr import PlateOCR  # paddle yalnizca istenirse
                    self.engines.append((name, PlateOCR()))
                else:
                    errors.append(f"{name}: bilinmeyen OCR motoru")
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                LOGGER.warning("OCR motoru '%s' yuklenemedi: %s", name, exc)
        if not self.engines:
            raise RuntimeError("Hicbir OCR motoru yuklenemedi -> " + " | ".join(errors))
        LOGGER.info("Plaka okuyucu hazir: dedektor=%s, OCR=%s",
                    self.detector.backend_name,
                    "+".join(name for name, _ in self.engines))
        self.debug_dir = debug_dir
        self._debug_counter = 0
        if self.debug_dir:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
                LOGGER.info("Plaka hata ayiklama kesitleri: %s", self.debug_dir)
            except OSError as exc:
                LOGGER.warning("Hata ayiklama dizini olusturulamadi: %s", exc)
                self.debug_dir = None

    # ------------------------------------------------------------------ #
    @property
    def rejected_boxes(self) -> int:
        """Geometri süzgecinin elediği sahte plaka kutusu sayısı."""
        return self.detector.rejected_boxes

    # ------------------------------------------------------------------ #
    def _save_debug_crop(self, image: np.ndarray, track_id: int, suffix: str) -> None:
        """Hata ayiklama modunda bir kesiti diske yazar."""
        if not self.debug_dir or image is None or image.size == 0:
            return
        self._debug_counter += 1
        filename = f"t{track_id:04d}_{self._debug_counter:05d}_{suffix}.jpg"
        path = os.path.join(self.debug_dir, filename)
        try:
            # cv2.imwrite Windows'ta ASCII disi yollara (ör. "MELSA ÖZKAYA")
            # HATA VERMEDEN yazamaz ve False doner. imencode + tofile Python'un
            # unicode dosya API'sini kullandigi icin her yolda calisir.
            ok, encoded = cv2.imencode(".jpg", image)
            if ok:
                encoded.tofile(path)
            else:
                LOGGER.debug("Hata ayiklama kesiti kodlanamadi: %s", path)
        except Exception as exc:
            LOGGER.debug("Hata ayiklama kesiti yazilamadi: %s", exc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare(plate_crop: np.ndarray) -> np.ndarray:
        """OCR oncesi on isleme; basarisiz olursa ham kesit kullanilir."""
        try:
            return preprocess_plate(plate_crop)
        except Exception as exc:
            LOGGER.debug("Plaka on isleme hatasi: %s", exc)
            return plate_crop

    # ------------------------------------------------------------------ #
    @property
    def ocr(self):
        """Geriye uyumluluk: varsa PaddleOCR motoru, yoksa ilk motor."""
        for name, engine in self.engines:
            if name == "paddle":
                return engine
        return self.engines[0][1]

    # ------------------------------------------------------------------ #
    def run_ocr(self, plate_crop: np.ndarray) -> OcrResult:
        """Kesiti ilk OCR motoruyla okur (tani / test amacli)."""
        name, engine = self.engines[0]
        if name == "paddle":
            return engine.read(self._prepare(plate_crop))  # type: ignore[attr-defined]
        return engine.read(plate_crop)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    def _crop_variants(
        self, plate_crop: np.ndarray, track_id: int, index: int
    ) -> List[Tuple[str, np.ndarray]]:
        """
        OCR'a gonderilecek kesit varyantlarini hazirlar.

        Kutu icindeki gercek plaka alanini izole edip egikligi duzelt.
        Iyilestirme basarisiz olursa (veya PLATE_OCR_TRY_BOTH_CROPS acikken
        her halukarda) orijinal kesit de denenir. Iyilestirme nadiren
        yanlis bolge secebilir; iki kesidi birlikte denemek bu riski
        tamamen ortadan kaldirir ve maliyeti yalnizca bir OCR cagrisidir.
        """
        crop_variants: List[Tuple[str, np.ndarray]] = []
        if PLATE_REFINE_REGION:
            refined = refine_plate_region(plate_crop)
            if refined is not None:
                self._save_debug_crop(refined, track_id, f"refined{index}")
                crop_variants.append(("duzlestirilmis", refined))
        if not crop_variants or PLATE_OCR_TRY_BOTH_CROPS:
            crop_variants.append(("orijinal", plate_crop))
        return crop_variants

    # ------------------------------------------------------------------ #
    @staticmethod
    def _evaluate_ocr_results(
        attempt: PlateAttempt,
        ocr_results: OcrResult,
        det_conf: float,
        variant_name: str,
        best_score: float,
    ) -> Tuple[float, bool]:
        """
        Bir kesitin OCR sonuclarini regex + guven kontrolunden gecirir ve
        `attempt` nesnesini gunceller.

        Donus: (guncel en iyi skor, bu sonuclarda gecerli plaka bulundu mu)
        """
        attempt.raw_texts.extend(ocr_results)
        found_valid = False

        # Cok parcali okumalari birlestirerek de degerlendir
        merged_text = "".join(text for text, _ in ocr_results)
        merged_conf = float(np.mean([conf for _, conf in ocr_results]))
        candidates = list(ocr_results)
        if len(ocr_results) > 1:
            candidates.append((merged_text, merged_conf))

        for text, conf in candidates:
            # --- 4) Regex + karakter duzeltme ---
            # SIRA ONEMLI: once format dogrulanir. Karakter duzeltmesinin
            # ardindan gecerli bir TR plaka formatina oturan metin, ham
            # OCR guveni dusuk olsa bile guclu bir kanittir; guven
            # esigini once uygulamak dogru okumalari da cope atar.
            plate = correct_and_validate_plate(text)
            if plate is None:
                if conf >= OCR_MIN_CONFIDENCE:
                    attempt.stage = "regex_failed"
                elif attempt.stage not in ("regex_failed", "ok"):
                    attempt.stage = "low_confidence"
                continue

            # --- 5) Confidence kontrolu (formata oturana gevsek esik) ---
            if conf < OCR_FORMAT_VALID_MIN_CONF:
                if attempt.stage not in ("regex_failed", "ok"):
                    attempt.stage = "low_confidence"
                continue

            found_valid = True
            score = conf * det_conf
            if score > best_score:
                best_score = score
                attempt.plate_text = plate
                attempt.confidence = conf
                attempt.stage = "ok"
                LOGGER.debug("Plaka '%s' -> %s (%s kesit, conf=%.2f)",
                             text, plate, variant_name, conf)

        # Daha onceki bir varyant basarili olduysa asama "ok" kalmalidir
        if attempt.plate_text is not None:
            attempt.stage = "ok"
        return best_score, found_valid

    # ------------------------------------------------------------------ #
    def _paddle_variant(
        self,
        engine,
        attempt: PlateAttempt,
        variant: np.ndarray,
        det_conf: float,
        variant_name: str,
        best_score: float,
    ) -> Tuple[float, bool]:
        """
        Tek bir kesit varyantini PaddleOCR'dan gecirir.

        Once tam boru hatti (metin tespiti + tanima) calisir. Hic metin
        donmezse YA DA metin donup gecerli plaka cikmazsa yalnizca-tanima
        modu denenir. Ikinci durum onemlidir: tespit modeli TR seridini,
        civata golgesini veya karakter boslugunu ayri kutu yapip plakayi
        parcaladiginda eski kod tanima moduna hic gecmiyordu.
        """
        prepared = self._prepare(variant)

        full = engine.detect_and_recognize(prepared)
        found = False
        if full:
            best_score, found = self._evaluate_ocr_results(
                attempt, full, det_conf, f"{variant_name}/paddle", best_score
            )

        if not full or (not found and PLATE_OCR_RECOGNIZE_ON_REGEX_FAIL):
            rec_only = engine.recognize_only(prepared)
            if rec_only:
                best_score, found_rec = self._evaluate_ocr_results(
                    attempt, rec_only, det_conf, f"{variant_name}/paddle-tanima",
                    best_score,
                )
                found = found or found_rec
            elif not full and attempt.stage in ("no_plate_detected", "plate_too_small"):
                attempt.stage = "no_ocr_text"
        return best_score, found

    # ------------------------------------------------------------------ #
    def _ocr_variant(
        self,
        attempt: PlateAttempt,
        variant: np.ndarray,
        det_conf: float,
        variant_name: str,
        best_score: float,
    ) -> float:
        """
        Kesit varyantini OCR motorlarindan sirayla gecirir. Bir motor gecerli
        plaka bulursa sonraki (daha yavas) motorlar bu kesit icin atlanir.
        """
        for name, engine in self.engines:
            if name == "paddle":
                best_score, found = self._paddle_variant(
                    engine, attempt, variant, det_conf, variant_name, best_score
                )
            else:
                # Plakaya ozel model ham (renkli) kesit bekler; Paddle icin
                # yapilan gri + keskinlestirme on islemesi burada kullanilmaz.
                results = engine.read(variant)  # type: ignore[attr-defined]
                found = False
                if results:
                    best_score, found = self._evaluate_ocr_results(
                        attempt, results, det_conf, f"{variant_name}/{name}", best_score
                    )
                elif attempt.stage in ("no_plate_detected", "plate_too_small"):
                    attempt.stage = "no_ocr_text"
            if found:
                break
        return best_score

    # ------------------------------------------------------------------ #
    def read_plate(self, vehicle_crop: np.ndarray, track_id: int = 0) -> PlateAttempt:
        """
        Tam plaka okuma zinciri:
        Detection -> Crop -> OCR -> Regex/Düzeltme -> Confidence Check.

        Basari ya da basarisizlik fark etmeksizin, zincirin nerede durdugunu
        anlatan bir PlateAttempt doner. Bu sayede plaka okunamadiginda hangi
        asamanin sorumlu oldugu loglardan ve ozet tablosundan gorulebilir.
        """
        attempt = PlateAttempt()
        self._save_debug_crop(vehicle_crop, track_id, "vehicle")

        # --- 1) Plaka tespiti ---
        plate_boxes = self.detector.detect(vehicle_crop)
        attempt.plate_count = len(plate_boxes)
        if not plate_boxes:
            attempt.stage = "no_plate_detected"
            return attempt

        attempt.best_plate_width = max(box[2] - box[0] for box, _ in plate_boxes)
        best_score = 0.0

        for index, (box, det_conf) in enumerate(plate_boxes):
            plate_width = box[2] - box[0]
            if plate_width < PLATE_MIN_CROP_WIDTH:
                if attempt.stage == "no_plate_detected":
                    attempt.stage = "plate_too_small"
                continue

            # --- 2) Plaka kirpma ve bolge iyilestirme ---
            plate_crop = crop_image(vehicle_crop, box, PLATE_CROP_PADDING)
            if plate_crop is None:
                continue
            self._save_debug_crop(plate_crop, track_id, f"plate{index}")

            # --- 3) OCR (her kesit varyanti icin) ---
            for variant_name, variant in self._crop_variants(plate_crop, track_id, index):
                best_score = self._ocr_variant(
                    attempt, variant, det_conf, variant_name, best_score
                )
        return attempt
