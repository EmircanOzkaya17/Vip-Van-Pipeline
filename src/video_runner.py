"""
Video işleme akışı.

Sıra: kaynak açma -> ilk frame -> sanal çizgi -> modeller -> çıktı yolları ve
video yazıcı -> frame döngüsü -> CSV / özet.

Bileşenlerin CLI argümanlarından kurulması (`build_virtual_line`,
`build_pipeline`) da burada yapılır; böylece `main.py` yalnızca uygulamayı
başlatır.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import (
    AUTO_SAVE_OUTPUT_VIDEO,
    FPS_WINDOW,
    LINE_REFERENCE_END,
    LINE_REFERENCE_RESOLUTION,
    LINE_REFERENCE_START,
    OUTPUT_FPS_FALLBACK,
    PLATE_READING_ENABLED,
)
from .file_utils import describe_saved_file
from .output_paths import (
    build_output_stem,
    resolve_output_csv_path,
    resolve_output_video_path,
)
from .pipeline import VehicleAnalyticsPipeline
from .plate_reader import PlateReader
from .reporting import export_results_csv, print_summary
from .vehicle_classifier import VehicleTypeClassifier
from .vehicle_detector import VehicleDetector
from .video_io import create_video_writer, open_video_source
from .virtual_line import VirtualLine
from .visualization import annotate_frame, pick_line_interactively

LOGGER = logging.getLogger(__name__)

WINDOW_NAME: str = "VIP Van Detection"


# =============================================================================
# BİLEŞENLERİN KURULUMU
# =============================================================================

def build_virtual_line(args: argparse.Namespace) -> VirtualLine:
    """CLI argümanlarına göre sanal çizgi nesnesini oluşturur."""
    if args.line:
        try:
            x1, y1, x2, y2 = (int(v) for v in args.line.split(","))
            return VirtualLine((x1, y1), (x2, y2), reference_resolution=None)
        except ValueError:
            LOGGER.warning(
                "--line formati hatali (%s). Varsayilan cizgi kullanilacak.",
                args.line,
            )
    return VirtualLine(
        LINE_REFERENCE_START, LINE_REFERENCE_END, LINE_REFERENCE_RESOLUTION
    )


def build_pipeline(
    args: argparse.Namespace, line: VirtualLine
) -> VehicleAnalyticsPipeline:
    """
    Tüm modelleri yükleyip pipeline'ı kurar.

    SIRA ONEMLI: plaka okuyucu torch'tan (YOLO / OpenCLIP) ONCE yuklenir.
    Windows'ta onnxruntime / paddle gibi kutuphaneler torch kendi DLL'lerini
    yukledikten sonra ice aktarilirsa "DLL load failed" hatasina dusebiliyor.
    Varsayilan OpenCV yolu bundan etkilenmez; Paddle motoru acilirsa korur.
    """
    plate_reader: Optional[PlateReader] = None
    plate_error: Optional[str] = None
    if not PLATE_READING_ENABLED:
        plate_error = "config.PLATE_READING_ENABLED = False"
    elif args.no_plate:
        plate_error = "--no-plate bayragi verildi"
    else:
        try:
            plate_reader = PlateReader(debug_dir=args.debug_plates)
        except Exception as exc:
            # Plaka modülü yoksa sistem sınıflandırma ile çalışmaya devam eder.
            # Yalnizca RuntimeError degil HER hata yakalanir: paddle/onnxruntime
            # DLL ve model dosyasi hatalari farkli tiplerde gelebiliyor.
            plate_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("=" * 60)
            LOGGER.error("PLAKA OKUMA DEVRE DISI: %s", plate_error)
            LOGGER.error("Video yine islenecek ama HICBIR plaka okunmayacak.")
            LOGGER.error("Teshis icin: python main.py --test-ocr <plaka.jpg> --verbose")
            LOGGER.error("=" * 60)

    detector = VehicleDetector(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    classifier = VehicleTypeClassifier(device=args.device)

    pipeline = VehicleAnalyticsPipeline(
        detector, classifier, line, plate_reader,
        extended_plate_search=args.extended_plate,
    )
    pipeline.plate_reader_error = plate_error
    LOGGER.info(
        "Plaka okuma: %s | uzatilmis arama: %s",
        "ACIK" if plate_reader is not None else "KAPALI",
        "acik" if args.extended_plate else "kapali",
    )
    return pipeline


# =============================================================================
# AKIŞ ADIMLARI
# =============================================================================

def _prepare_virtual_line(
    args: argparse.Namespace, first_frame: np.ndarray
) -> VirtualLine:
    """Sanal çizgiyi oluşturur, frame'e ölçekler; istenirse fare ile seçtirir."""
    height, width = first_frame.shape[:2]
    line = build_virtual_line(args)
    line.fit_to_frame(width, height)
    if args.pick_line:
        try:
            line = pick_line_interactively(first_frame, line)
            line.fit_to_frame(width, height)
        except cv2.error as exc:
            LOGGER.warning("Interaktif secim yapilamadi (%s).", exc)
    return line


def _resolve_output_paths(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Video ve CSV çıktı yollarını belirler; ikisi aynı dosya gövdesini paylaşır.

    Varsayılan davranış: işlenen videoyu kaynağın bulunduğu dizindeki alt
    klasöre kaydet. Böylece bayrak verilmeyi unutulsa bile çıktı kaybolmaz.
    --no-save-video verilmişse video yolu None olur.
    """
    output_stem = build_output_stem(args.source)
    video_path: Optional[str] = None
    if not args.no_save_video:
        video_path = resolve_output_video_path(
            args.source, args.save_video, args.output_dir, output_stem,
            auto_save=AUTO_SAVE_OUTPUT_VIDEO,
        )
    csv_path = resolve_output_csv_path(
        args.source, args.csv, args.output_dir, output_stem
    )
    return video_path, csv_path


def _run_frame_loop(
    capture: cv2.VideoCapture,
    first_frame: np.ndarray,
    pipeline: VehicleAnalyticsPipeline,
    line: VirtualLine,
    writer: Optional[cv2.VideoWriter],
    show: bool,
    max_frames: int,
) -> None:
    """
    Kaynak bitene, kullanıcı durdurana veya frame sınırına ulaşılana kadar
    frame'leri işler. Çıkışta açık track'leri kapatır ve kaynakları serbest bırakır.
    """
    frame = first_frame
    durations: deque = deque(maxlen=FPS_WINDOW)
    processed = 0

    try:
        while True:
            start_time = time.perf_counter()
            active_tracks = pipeline.process_frame(frame)
            durations.append(time.perf_counter() - start_time)
            processed += 1

            if show or writer is not None:
                fps = 1.0 / max(float(np.mean(durations)), 1e-6)
                annotate_frame(frame, line, active_tracks, processed, fps,
                               pipeline.stats)

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    LOGGER.info("Kullanici tarafindan durduruldu.")
                    break

            if max_frames and processed >= max_frames:
                LOGGER.info("Maksimum frame sayisina ulasildi.")
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.info("Video akisi sona erdi.")
                break

    except KeyboardInterrupt:
        LOGGER.info("Klavye kesmesi ile durduruldu.")
    except Exception as exc:
        LOGGER.exception("Beklenmeyen hata: %s", exc)
    finally:
        pipeline.finalize_all()
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def _report_results(
    pipeline: VehicleAnalyticsPipeline,
    video_path: Optional[str],
    csv_path: Optional[str],
) -> None:
    """CSV'yi yazar, kaydedilen videoyu doğrular ve özet tabloyu basar."""
    if csv_path:
        export_results_csv(csv_path, pipeline.all_results())
    if video_path:
        saved_video = describe_saved_file(video_path)
        if saved_video:
            LOGGER.info("Islenmis video kaydedildi: %s", saved_video)
        else:
            LOGGER.warning("Cikti videosu diskte bulunamadi: %s", video_path)
    print_summary(pipeline, video_path, csv_path)


# =============================================================================
# ANA AKIŞ
# =============================================================================

def run_video(args: argparse.Namespace) -> int:
    """
    Video işleme akışını baştan sona çalıştırır.
    Dönüş: süreç çıkış kodu (0 = başarılı, 1 = başlatma hatası).
    """
    # ---------------- Video kaynağı ----------------
    try:
        capture = open_video_source(args.source)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        LOGGER.error("Ilk frame okunamadi. Kaynak gecersiz olabilir.")
        capture.release()
        return 1

    height, width = first_frame.shape[:2]
    source_fps = capture.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS_FALLBACK
    LOGGER.info("Kaynak: %s | %dx%d @ %.1f FPS", args.source, width, height, source_fps)

    # ---------------- Sanal çizgi ----------------
    line = _prepare_virtual_line(args, first_frame)

    # ---------------- Modeller ----------------
    try:
        pipeline = build_pipeline(args, line)
    except RuntimeError as exc:
        LOGGER.error("Baslatma hatasi: %s", exc)
        capture.release()
        return 1

    # ---------------- Çıktı yolları ve video yazıcı ----------------
    output_video_path, csv_path = _resolve_output_paths(args)
    writer: Optional[cv2.VideoWriter] = None
    if output_video_path:
        writer, output_video_path = create_video_writer(
            output_video_path, source_fps, (width, height)
        )
    else:
        LOGGER.info("Video kaydi kapali; sonuclar yalnizca ekrana/CSV'ye gidecek.")

    # ---------------- Ana döngü ----------------
    _run_frame_loop(
        capture, first_frame, pipeline, line, writer,
        show=args.show, max_frames=args.max_frames,
    )

    # ---------------- Çıktılar ----------------
    _report_results(pipeline, output_video_path, csv_path)
    return 0
