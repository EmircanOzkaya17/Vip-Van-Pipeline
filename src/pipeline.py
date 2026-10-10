"""
Ana işlem hattı: tespit/takip, çizgi geçişi, OpenCLIP sınıflandırması ve plaka
okuma adımlarını her frame için koordine eder, track yaşam döngüsünü yönetir.

Bileşenler (dedektör, sınıflandırıcı, plaka okuyucu, çizgi) dışarıdan verilir;
bu modül hiçbir modeli kendisi yüklemez.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .bbox import box_center, crop_image
from .config import (
    CLASS_VOTE_FRAMES,
    CLIP_CROP_PADDING,
    CLIP_FALLBACK_ON_TRACK_END,
    CLIP_FRAME_OFFSETS,
    NON_TARGET_VEHICLE_TYPE,
    PLATE_EXTENDED_MAX_ATTEMPTS,
    PLATE_EXTENDED_SEARCH,
    PLATE_EXTENDED_STRIDE,
    PLATE_FRAME_COUNT,
    PLATE_MIN_VOTES,
    REQUIRE_LINE_CROSSING,
    TARGET_VEHICLE_TYPE,
    TRACK_TIMEOUT_FRAMES,
)
from .plate_text import majority_vote_plate
from .track import TrackInfo

if TYPE_CHECKING:  # Yalnızca tip denetimi için; çalışma zamanında import edilmez
    from .plate_reader import PlateReader
    from .vehicle_classifier import VehicleTypeClassifier
    from .vehicle_detector import VehicleDetector
    from .virtual_line import VirtualLine

LOGGER = logging.getLogger(__name__)


class VehicleAnalyticsPipeline:
    """
    Tüm bileşenleri birleştiren ana işlem hattı.

    Her frame için sırasıyla:
      1) Tespit + takip
      2) Sınıf oylaması (ilk 7 frame)
      3) Sanal çizgi geçişi kontrolü
      4) Geçişten sonraki hedef frame'de OpenCLIP sınıflandırması
      5) Hedef araçlar için 7 frame boyunca plaka okuma
      6) Majority voting ile nihai plaka
      7) Kapanan track'lerin arşivlenmesi
    """

    def __init__(
        self,
        detector: VehicleDetector,
        classifier: VehicleTypeClassifier,
        line: VirtualLine,
        plate_reader: Optional[PlateReader] = None,
        extended_plate_search: bool = PLATE_EXTENDED_SEARCH,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.line = line
        self.plate_reader = plate_reader
        self.extended_plate_search = extended_plate_search
        # Plaka okuyucu baslatilamadiysa sebebi (ozet tablosunda gosterilir)
        self.plate_reader_error: Optional[str] = None

        self.tracks: Dict[int, TrackInfo] = {}          # Aktif track'ler
        self.completed: Dict[int, TrackInfo] = {}       # Kapanmış track'ler
        self.frame_index: int = 0

        # İstatistikler
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray) -> List[TrackInfo]:
        """Tek bir frame'i baştan sona işler ve aktif track listesini döner."""
        detections = self.detector.track(frame)
        active_tracks: List[TrackInfo] = []
        seen_ids = set()

        for detection in detections:
            track_id = detection["track_id"]
            seen_ids.add(track_id)
            track = self._get_or_create_track(track_id)

            try:
                self._update_basics(track, detection)
                self._update_line_crossing(track)
                self._process_after_crossing(track, frame)
            except Exception as exc:
                # Tek bir track'teki hata tüm akışı durdurmamalıdır.
                LOGGER.warning("Track %s islenirken hata: %s", track_id, exc)

            active_tracks.append(track)

        self._cleanup_tracks(seen_ids)
        self.frame_index += 1
        return active_tracks

    # ------------------------------------------------------------------ #
    @staticmethod
    def _vehicle_crop(frame: np.ndarray, track: TrackInfo) -> Optional[np.ndarray]:
        """
        Track'in güncel kutusundan araç kesiti alır.
        Aynı kesit hem OpenCLIP'e hem plaka zincirine gider (CLIP_CROP_PADDING).
        """
        return crop_image(frame, track.bbox, CLIP_CROP_PADDING)

    # ------------------------------------------------------------------ #
    def _get_or_create_track(self, track_id: int) -> TrackInfo:
        """Var olan track'i döner, yoksa yeni bir kayıt oluşturur."""
        track = self.tracks.get(track_id)
        if track is None:
            track = TrackInfo(
                track_id=track_id,
                first_frame=self.frame_index,
                last_frame=self.frame_index,
            )
            self.tracks[track_id] = track
            self.stats["total_tracks"] += 1
            LOGGER.debug("Yeni track olusturuldu: #%s", track_id)
        return track

    # ------------------------------------------------------------------ #
    def _update_basics(self, track: TrackInfo, detection: Dict[str, Any]) -> None:
        """
        Kutu bilgisini günceller ve ilk 7 frame boyunca araç sınıfı oylaması yapar.
        7 frame sonunda en çok oy alan sınıf kalıcı olarak atanır.
        """
        track.last_frame = self.frame_index
        track.bbox = detection["bbox"]
        track.detection_conf = detection["conf"]

        if track.vehicle_class is None:
            track.class_votes[detection["class_name"]] += 1
            if sum(track.class_votes.values()) >= CLASS_VOTE_FRAMES:
                track.vehicle_class = track.class_votes.most_common(1)[0][0]
                LOGGER.debug(
                    "Track #%s sinifi belirlendi: %s", track.track_id,
                    track.vehicle_class,
                )

    # ------------------------------------------------------------------ #
    def _update_line_crossing(self, track: TrackInfo) -> None:
        """
        Bounding box merkezinin sanal çizgiyi geçtiği ilk frame'i tespit eder.
        Geçiş frame'i referans alınır; bu frame sınıflandırmaya dahil edilmez.
        """
        center = box_center(track.bbox)
        if track.cross_frame is None:
            if self.line.is_crossed(track.previous_center, center):
                track.cross_frame = self.frame_index
                self.stats["line_crossings"] += 1
                LOGGER.info(
                    "Track #%s sanal cizgiyi gecti (frame %d)",
                    track.track_id, self.frame_index,
                )
            elif not REQUIRE_LINE_CROSSING and track.age >= CLASS_VOTE_FRAMES:
                # Çizgi geçişi zorunlu değilse belirli bir yaştan sonra tetikle
                track.cross_frame = self.frame_index
        track.previous_center = center

    # ------------------------------------------------------------------ #
    def _process_after_crossing(self, track: TrackInfo, frame: np.ndarray) -> None:
        """Çizgi geçişi sonrası sınıflandırma ve plaka adımlarını yürütür."""
        offset = track.offset_after_cross(self.frame_index)
        if offset is None:
            return

        # Son kesiti sakla: karar bekleyenler icin CLIP fallback'i, plakasi
        # henuz kesinlesmemis hedefler icin kapanistaki son plaka denemesi.
        pending = track.is_target is None and CLIP_FALLBACK_ON_TRACK_END
        plate_open = bool(track.is_target) and not track.plate_finalized
        if pending or plate_open:
            track.last_crop = self._vehicle_crop(frame, track)
            track.last_crop_offset = offset

        self._maybe_classify(track, frame, offset)
        self._maybe_read_plate(track, frame, offset)
        self._maybe_finalize_plate(track, offset)

    # ------------------------------------------------------------------ #
    def _maybe_classify(self, track: TrackInfo, frame: np.ndarray, offset: int) -> None:
        """
        Belirlenen offset frame'lerinde OpenCLIP sınıflandırması çalıştırır.
        Tüm offsetler tamamlandığında nihai kararı verir.

        DUZELTME: Eskiden karar yalnizca TAM OLARAK planlanan karede (gecis+2)
        verilebiliyordu. YOLO aracı o tek karede kacirirsa (gecis anında
        kadraj kenarinda oldukca sik) karar track kapanisina kaliyor, bu arada
        plaka penceresi bosa geciyor ve tamponlanan kesitler hic OCR'a
        gitmeden siliniyordu. Artik planlanan kareler gecmisse (gecikmis)
        eldeki skorlarla, hic skor yoksa GUNCEL kare ile hemen karar verilir.
        """
        if track.is_target is not None:      # Karar zaten verilmiş
            return

        scheduled = (offset in CLIP_FRAME_OFFSETS
                     and offset not in track.processed_clip_offsets)
        overdue = offset > max(CLIP_FRAME_OFFSETS)
        if not scheduled and not overdue:
            return

        if scheduled or not track.clip_frame_scores:
            crop = self._vehicle_crop(frame, track)
            scores = self.classifier.score_crop(crop) if crop is not None else None
            if scheduled:
                track.processed_clip_offsets.add(offset)
            if scores is not None:
                track.clip_frame_scores.append(scores)
                if not scheduled:
                    self.stats["late_classifications"] += 1
                    LOGGER.info(
                        "Track #%s: planlanan CLIP karesi kacirildi, "
                        "gecis+%d karesinde siniflandiriliyor.",
                        track.track_id, offset,
                    )

        # Planlanan tüm offsetler işlendiyse (ya da süreleri geçtiyse) karar ver
        all_done = track.processed_clip_offsets >= set(CLIP_FRAME_OFFSETS)
        if (all_done or overdue) and track.clip_frame_scores:
            self._finalize_classification(track, offset)

    # ------------------------------------------------------------------ #
    def _finalize_classification(
        self, track: TrackInfo, offset: Optional[int] = None
    ) -> None:
        """
        Nihai skorları hesaplar ve hedef araç olup olmadığına karar verir.
        Hedef değilse track'in ileri işlemleri sonlandırılır.

        offset None ise karar track kapanirken verilmistir (tani amacli kayit).
        """
        aggregated = self.classifier.aggregate(track.clip_frame_scores)
        if aggregated is None:
            return

        track.final_positive = aggregated["final_positive"]
        track.final_negative = aggregated["final_negative"]
        track.final_margin = aggregated["final_margin"]
        track.is_target = self.classifier.is_target_vehicle(
            track.final_positive, track.final_margin
        )
        track.vehicle_type = (
            TARGET_VEHICLE_TYPE if track.is_target else NON_TARGET_VEHICLE_TYPE
        )
        track.decision_offset = offset
        track.decided_on_close = offset is None

        if track.is_target:
            self.stats["target_vehicles"] += 1
            LOGGER.info(
                "Track #%s -> HEDEF ARAC | pos=%.3f neg=%.3f margin=%.3f | karar: %s",
                track.track_id, track.final_positive,
                track.final_negative, track.final_margin,
                track.decision_label(),
            )
        else:
            self.stats["non_target_vehicles"] += 1
            track.release_buffers()          # Takip sonlandırıldı, tampon boşalt
            LOGGER.debug(
                "Track #%s -> hedef degil | pos=%.3f margin=%.3f",
                track.track_id, track.final_positive, track.final_margin,
            )

    # ------------------------------------------------------------------ #
    def _in_plate_window(self, track: TrackInfo, offset: int) -> bool:
        """
        Bu karede plaka denemesi yapilip yapilmayacagini soyler.

        Temel pencere : cizgi gecisinden sonraki ilk PLATE_FRAME_COUNT kare
                        (spesifikasyon geregi her zaman tamamen kullanilir).
        Uzatilmis     : temel pencere gecerli bir plaka uretmediyse, arac
                        kameraya yaklastikca her PLATE_EXTENDED_STRIDE karede
                        bir denemeye devam edilir. Kare sayaci yerine "son
                        denemeden bu yana gecen kare" kullanilir; boylece YOLO
                        tam o karede araci kacirsa da deneme kaybolmaz.
        """
        if 1 <= offset <= PLATE_FRAME_COUNT:
            return True
        if not self.extended_plate_search or offset <= PLATE_FRAME_COUNT:
            return False
        if track.plate_attempts >= PLATE_EXTENDED_MAX_ATTEMPTS:
            return False
        last = track.last_plate_offset if track.last_plate_offset is not None else 0
        return offset - last >= PLATE_EXTENDED_STRIDE

    # ------------------------------------------------------------------ #
    def _maybe_read_plate(self, track: TrackInfo, frame: np.ndarray, offset: int) -> None:
        """
        Çizgi geçişinden sonraki ilk 7 frame'de plaka okuma zincirini çalıştırır.

        Sınıflandırma kararı 2. frame'de verildiğinden, karardan önceki
        frame'lerin kesitleri tamponlanır ve karar olumlu çıkarsa geriye
        dönük olarak işlenir. Böylece 7 frame'in tamamı değerlendirilir.

        DUZELTME: Tampon artik pencere kontrolunden ONCE bosaltilir. Eskiden
        karar pencere disindaki bir karede verilirse (ör. gecis+9) fonksiyon
        pencere kontrolunde donuyor ve tampondaki 7 kesit hic islenmiyordu.
        """
        if self.plate_reader is None or track.plate_finalized:
            return
        if track.is_target is False:         # Hedef değil -> işlem yok
            return

        # Karar olumlu ve bekleyen kesit varsa once onlari isle
        if track.is_target and track.plate_buffer:
            self._flush_plate_buffer(track)

        if not self._in_plate_window(track, offset):
            return

        crop = self._vehicle_crop(frame, track)
        if crop is None:
            return
        track.last_plate_offset = offset

        if track.is_target is None:
            # Karar henüz verilmedi -> kesiti tamponla
            if len(track.plate_buffer) < PLATE_FRAME_COUNT:
                track.plate_buffer.append((offset, crop))
            return

        self._run_plate_pipeline(track, crop)

    # ------------------------------------------------------------------ #
    def _flush_plate_buffer(self, track: TrackInfo) -> None:
        """Karar oncesi tamponlanan kesitleri sirayla plaka zincirinden gecirir."""
        if not track.plate_buffer or self.plate_reader is None:
            return
        buffered = list(track.plate_buffer)
        track.plate_buffer.clear()
        self.stats["plate_buffer_flushed"] += len(buffered)
        LOGGER.debug("Track #%s: tampondaki %d kesit isleniyor.",
                     track.track_id, len(buffered))
        for _, buffered_crop in buffered:
            self._run_plate_pipeline(track, buffered_crop)

    # ------------------------------------------------------------------ #
    def _run_plate_pipeline(self, track: TrackInfo, crop: np.ndarray) -> None:
        """
        Tek bir kesit uzerinde plaka okuma zincirini calistirir ve sonucun
        hangi asamada durdugunu istatistiklere isler.
        """
        track.plate_attempts += 1
        try:
            attempt = self.plate_reader.read_plate(  # type: ignore[union-attr]
                crop, track.track_id
            )
        except Exception as exc:
            LOGGER.warning("Plaka okuma hatasi (track #%s): %s", track.track_id, exc)
            self.stats["plate_stage_error"] += 1
            return

        # Asama bazli sayaclar: ozet tablosunda darbogazi gosterir
        self.stats[f"plate_stage_{attempt.stage}"] += 1
        track.plate_stages[attempt.stage] += 1
        LOGGER.debug("Track #%s plaka denemesi %d | %s",
                     track.track_id, track.plate_attempts, attempt.describe())

        if attempt.ok:
            track.plate_candidates.append((attempt.plate_text, attempt.confidence))
            LOGGER.info("Track #%s plaka adayi: %s (%.2f)",
                        track.track_id, attempt.plate_text, attempt.confidence)

    # ------------------------------------------------------------------ #
    def _maybe_finalize_plate(self, track: TrackInfo, offset: Optional[int] = None) -> None:
        """
        7 frame tamamlandığında (veya track kapanırken) majority voting
        uygulayarak nihai plakayı belirler.

        Temel pencere (ilk 7 kare) her zaman tamamlanir. Uzatilmis arama
        aciksa ve bu pencere gecerli bir plaka uretmediyse, plaka onaylanana
        ya da deneme tavanina ulasilana kadar karar ertelenir. offset None
        (track kapanisi) ise beklemeden karar verilir.
        """
        if track.plate_finalized or not track.is_target:
            return
        if offset is not None:
            if offset < PLATE_FRAME_COUNT:
                return
            if self.extended_plate_search and self.plate_reader is not None:
                # Uzatilmis modda erken bitirmek icin AYNI metnin en az
                # PLATE_MIN_VOTES kez okunmasi gerekir. Tek bir yuksek guvenli
                # okuma yetmez: uzaktaki kucuk plakada model "16RT2877" gibi
                # yanlis ama formata uyan bir metni 0.97 guvenle verebiliyor.
                # Arac yaklastikca gelen okumalar beklenir; track kapanirken
                # normal oylama (tek yuksek guvenli oy dahil) uygulanir.
                if (not self._has_consensus(track)
                        and track.plate_attempts < PLATE_EXTENDED_MAX_ATTEMPTS):
                    return

        plate_text, confidence = majority_vote_plate(track.plate_candidates)
        track.plate_text = plate_text
        track.ocr_confidence = confidence
        track.plate_finalized = True
        track.release_buffers()

        if plate_text:
            self.stats["plates_read"] += 1
            LOGGER.info(
                "Track #%s -> FINAL PLAKA: %s (conf=%.2f, %d aday)",
                track.track_id, plate_text, confidence or 0.0,
                len(track.plate_candidates),
            )
        else:
            self.stats["plates_failed"] += 1
            LOGGER.info(
                "Track #%s -> plaka okunamadi (%d deneme, %d aday%s)",
                track.track_id, track.plate_attempts, len(track.plate_candidates),
                "" if self.plate_reader is not None else ", PLAKA OKUYUCU KAPALI",
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_consensus(track: TrackInfo) -> bool:
        """Ayni plaka metni en az PLATE_MIN_VOTES kez okunduysa True."""
        if not track.plate_candidates:
            return False
        counts = Counter(text for text, _ in track.plate_candidates)
        return counts.most_common(1)[0][1] >= PLATE_MIN_VOTES

    # ------------------------------------------------------------------ #
    def _cleanup_tracks(self, seen_ids: set) -> None:
        """
        Belirli süredir görünmeyen track'leri kapatır, sonuçlarını arşivler
        ve bellek tamponlarını serbest bırakır.
        """
        expired = [
            track_id for track_id, track in self.tracks.items()
            if track_id not in seen_ids
            and (self.frame_index - track.last_frame) > TRACK_TIMEOUT_FRAMES
        ]
        for track_id in expired:
            track = self.tracks.pop(track_id)
            self._close_track(track)

    # ------------------------------------------------------------------ #
    def _close_track(self, track: TrackInfo) -> None:
        """
        Kapanan bir track için son kararları verir ve arşivler.

        DUZELTME: Karar burada (kapanista) olumlu cikarsa, eskiden tampondaki
        kesitler hic islenmeden release_buffers() ile siliniyordu. Artik once
        tampon, ardindan (hala aday yoksa) aracin son kesiti plaka zincirinden
        gecirilir; ancak bundan sonra nihai plaka oylamasi yapilir.
        """
        # Sınıflandırma yapılamadıysa son kesit ile son bir deneme
        if track.is_target is None and CLIP_FALLBACK_ON_TRACK_END:
            if not track.clip_frame_scores and track.last_crop is not None:
                scores = self.classifier.score_crop(track.last_crop)
                if scores is not None:
                    track.clip_frame_scores.append(scores)
            if track.clip_frame_scores:
                self._finalize_classification(track, None)

        if track.is_target and not track.plate_finalized:
            if self.plate_reader is not None:
                self._flush_plate_buffer(track)
                last_untried = (track.last_crop is not None
                                and track.last_crop_offset != track.last_plate_offset)
                if not track.plate_candidates and last_untried:
                    self._run_plate_pipeline(track, track.last_crop)
            self._maybe_finalize_plate(track)

        track.release_buffers()
        track.archived = True
        self.completed[track.track_id] = track
        LOGGER.debug("Track #%s kapatildi.", track.track_id)

    # ------------------------------------------------------------------ #
    def finalize_all(self) -> None:
        """Video bittiğinde açık kalan tüm track'leri kapatır."""
        for track in list(self.tracks.values()):
            self._close_track(track)
        self.tracks.clear()

    # ------------------------------------------------------------------ #
    def all_results(self) -> List[TrackInfo]:
        """Tüm (aktif + kapanmış) track kayıtlarını track_id sırasıyla döner."""
        merged = dict(self.completed)
        merged.update(self.tracks)
        return [merged[key] for key in sorted(merged.keys())]
