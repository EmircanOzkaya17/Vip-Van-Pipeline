"""
Komut satırı arayüzü: argüman tanımları ve ayrıştırma.

Varsayılan değerler `config` modülünden (dolayısıyla .env'den) gelir.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import (
    OUTPUT_DIR_NAME,
    PLATE_EXTENDED_SEARCH,
    YOLO_CONF_THRESHOLD,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL_PATH,
)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description="VIP/Yolcu Vani tespit, siniflandirma, takip ve plaka okuma",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=None,
                        help="Video dosyasi, RTSP adresi veya kamera indeksi")
    parser.add_argument("--test-ocr", default=None,
                        help="Tek bir goruntu uzerinde plaka zincirini test et "
                             "(video islenmez, ayrintili tani ciktisi verir)")
    parser.add_argument("--model", default=YOLO_MODEL_PATH, help="YOLO model yolu")
    parser.add_argument("--device", default=None,
                        help="Cihaz: cuda, cuda:0, cpu (bos = otomatik)")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE,
                        help="YOLO inference boyutu")
    parser.add_argument("--conf", type=float, default=YOLO_CONF_THRESHOLD,
                        help="Tespit guven esigi")
    parser.add_argument("--line", default=None,
                        help="Sanal cizgi koordinatlari: x1,y1,x2,y2 (piksel)")
    parser.add_argument("--pick-line", action="store_true",
                        help="Ilk frame uzerinde fare ile cizgi sec")
    parser.add_argument("--show", action="store_true",
                        help="Sonuclari pencerede goster (q ile cikis)")
    parser.add_argument("--save-video", nargs="?", const="", default=None,
                        metavar="YOL",
                        help="Anotasyonlu videonun yazilacagi dosya veya dizin. "
                             "Hic verilmezse video yine de kaynagin yanindaki "
                             f"'{OUTPUT_DIR_NAME}' klasorune kaydedilir")
    parser.add_argument("--no-save-video", action="store_true",
                        help="Otomatik video kaydini kapat (yalnizca izleme)")
    parser.add_argument("--output-dir", default=None, metavar="DIZIN",
                        help="Tum ciktilarin yazilacagi klasor "
                             "(varsayilan: kaynak videonun yanindaki alt klasor)")
    parser.add_argument("--csv", nargs="?", const="", default=None, metavar="YOL",
                        help="Sonuc CSV dosyasi. Degersiz verilirse video ile "
                             "ayni klasore ayni isimle yazilir")
    parser.add_argument("--no-plate", action="store_true",
                        help="Plaka okumayi devre disi birak")
    # --extended-plate / --no-extended-plate (varsayilan config'den gelir).
    # Eski komutlardaki --extended-plate bayragi aynen calismaya devam eder.
    parser.add_argument("--extended-plate", action=argparse.BooleanOptionalAction,
                        default=PLATE_EXTENDED_SEARCH,
                        help="Ilk 7 karede plaka okunamazsa arac yaklastikca "
                             "denemeye devam et")
    parser.add_argument("--debug-plates", default=None,
                        help="Plaka kesitlerinin yazilacagi hata ayiklama dizini")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Islenecek maksimum frame (0 = sinirsiz)")
    parser.add_argument("--verbose", action="store_true", help="Ayrintili log")
    return parser.parse_args(argv)
