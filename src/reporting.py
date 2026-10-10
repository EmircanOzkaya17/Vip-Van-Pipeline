"""
Sonuç raporlama: track sonuçlarının CSV'ye yazılması ve konsola özet tablo.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from .file_utils import describe_saved_file
from .track import RECORD_FIELDS, TrackInfo

if TYPE_CHECKING:  # Yalnızca tip denetimi için
    from .pipeline import VehicleAnalyticsPipeline

LOGGER = logging.getLogger(__name__)

SEPARATOR_WIDTH: int = 74

# Plaka zincirinin hangi asamada takildigini gosteren dokum etiketleri.
# Plaka okunamadiginda once ozetteki bu tabloya bakilmalidir.
PLATE_STAGE_LABELS: Dict[str, str] = {
    "plate_stage_no_plate_detected": "Plaka kutusu bulunamadi",
    "plate_stage_plate_too_small":   "Plaka cok kucuk (dusuk cozunurluk)",
    "plate_stage_no_ocr_text":       "OCR metin dondurmedi",
    "plate_stage_low_confidence":    "OCR guveni esigin altinda",
    "plate_stage_regex_failed":      "Format dogrulamasi basarisiz",
    "plate_stage_ok":                "Basarili okuma",
    "plate_stage_error":             "Beklenmeyen hata",
}


def export_results_csv(path: str, tracks: List[TrackInfo]) -> None:
    """Tüm track sonuçlarını CSV dosyasına yazar."""
    # Çıktı klasörü videonunkiyle aynı olabilir ve henüz oluşturulmamış olabilir.
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            LOGGER.error("CSV dizini olusturulamadi (%s): %s", directory, exc)
            return
    try:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RECORD_FIELDS))
            writer.writeheader()
            for track in tracks:
                writer.writerow(track.to_record())
        LOGGER.info("Sonuclar kaydedildi: %s (%d kayit)", path, len(tracks))
    except OSError as exc:
        LOGGER.error("CSV yazilamadi (%s): %s", path, exc)


def _print_target_table(targets: List[TrackInfo]) -> None:
    """
    Hedef araçları (VIP van) tablo halinde basar.

    KARAR : siniflandirmanin verildigi kare (+2 = planlanan, kapanis = track
            kaybolurken). DENEME: plaka zincirinin bu arac icin kac kez
            calistigi. DENEME 0 ise zincir hic calismamistir.
    """
    print("-" * SEPARATOR_WIDTH)
    print(f" {'TRACK':<7}{'SINIF':<7}{'PLAKA':<11}"
          f"{'POS':<7}{'MARGIN':<8}{'OCR':<6}{'KARAR':<9}{'DENEME':<7}{'ADAY':<5}")
    print("-" * SEPARATOR_WIDTH)
    for track in targets:
        print(
            f" {track.track_id:<7}{(track.vehicle_class or '-'):<7}"
            f"{(track.plate_text or '-'):<11}"
            f"{(track.final_positive or 0):<7.3f}"
            f"{(track.final_margin or 0):<8.3f}"
            f"{(track.ocr_confidence or 0):<6.2f}"
            f"{track.decision_label():<9}"
            f"{track.plate_attempts:<7}"
            f"{len(track.plate_candidates):<5}"
        )
        if not track.plate_text and track.plate_candidates:
            # Aday var ama oylamadan gecmedi -> hangi metinler okundu?
            preview = ", ".join(f"{t}({c:.2f})" for t, c in track.plate_candidates[:6])
            print(f"   adaylar: {preview}")
        if not track.plate_text and track.plate_stages:
            print(f"   asamalar: {track.plate_stage_summary()}")


def _print_plate_reader_status(pipeline: VehicleAnalyticsPipeline) -> None:
    """Plaka okuyucunun acik olup olmadigini ve kapaliysa sebebini basar."""
    reader = pipeline.plate_reader
    if reader is not None:
        mode = "uzatilmis arama" if pipeline.extended_plate_search else "ilk 7 kare"
        detector = getattr(reader, "detector", None)
        backend = getattr(detector, "backend_name", "?")
        engines = "+".join(name for name, _ in getattr(reader, "engines", [])) or "?"
        print(f" Plaka okuyucu           : ACIK ({backend}, OCR={engines}, {mode})")
        zoom_hits = getattr(detector, "zoom_hits", 0)
        if zoom_hits:
            print(f" Yakinlastirmayla bulunan: {zoom_hits} plaka kutusu")
        return
    reason = getattr(pipeline, "plate_reader_error", None) or "bilinmiyor"
    print(" Plaka okuyucu           : *** KAPALI ***")
    print(f"   sebep: {reason[:SEPARATOR_WIDTH * 2]}")


def _print_plate_stage_breakdown(pipeline: VehicleAnalyticsPipeline) -> None:
    """Plaka denemelerinin hangi aşamada sonuçlandığını yüzdeleriyle basar."""
    stage_counts = [(label, pipeline.stats[key])
                    for key, label in PLATE_STAGE_LABELS.items() if pipeline.stats[key]]
    if not stage_counts:
        return
    total = sum(count for _, count in stage_counts)
    print("-" * SEPARATOR_WIDTH)
    print(f" PLAKA ZINCIRI DOKUMU ({total} deneme)")
    print("-" * SEPARATOR_WIDTH)
    for label, count in sorted(stage_counts, key=lambda x: -x[1]):
        share = 100.0 * count / total
        print(f" {label:<42}{count:>5}  ({share:5.1f}%)")


def print_summary(
    pipeline: VehicleAnalyticsPipeline,
    video_path: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> None:
    """Konsola özet tablo basar ve kaydedilen çıktı dosyalarını gösterir."""
    results = pipeline.all_results()
    targets = [t for t in results if t.is_target]
    print("\n" + "=" * SEPARATOR_WIDTH)
    print(" OZET")
    print("=" * SEPARATOR_WIDTH)
    print(f" Toplam track            : {len(results)}")
    print(f" Cizgi gecisi            : {pipeline.stats['line_crossings']}")
    print(f" Hedef arac (VIP van)    : {len(targets)}")
    print(f" Okunan plaka            : {pipeline.stats['plates_read']}")
    _print_plate_reader_status(pipeline)
    if pipeline.stats["late_classifications"]:
        print(f" Gec siniflandirma       : {pipeline.stats['late_classifications']}"
              "  (planlanan CLIP karesinde tespit yoktu)")
    if pipeline.plate_reader is not None and pipeline.plate_reader.rejected_boxes:
        print(f" Elenen sahte plaka kutusu: {pipeline.plate_reader.rejected_boxes}"
              "  (jant / far / tampon)")

    saved_video = describe_saved_file(video_path)
    if saved_video:
        print(f" Kaydedilen video        : {saved_video}")
    if csv_path and os.path.exists(csv_path):
        print(f" Kaydedilen CSV          : {csv_path}")
    if targets:
        _print_target_table(targets)

    _print_plate_stage_breakdown(pipeline)

    # Hedef var ama plaka zinciri hic calismadiysa bunu acikca soyle;
    # aksi halde ozet "OCR okuyamadi" gibi yanlis yorumlanir.
    never_tried = [t.track_id for t in targets if t.plate_attempts == 0]
    if never_tried and pipeline.plate_reader is not None:
        print("-" * SEPARATOR_WIDTH)
        print(f" UYARI: {len(never_tried)} hedef aracta plaka zinciri HIC calismadi "
              f"(track {', '.join(map(str, never_tried[:10]))}).")
        print(" Arac gecisten hemen sonra kadrajdan cikmis ve son kesiti de "
              "kaydedilememis olabilir.")
    print("=" * SEPARATOR_WIDTH + "\n")
