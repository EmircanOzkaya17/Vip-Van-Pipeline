"""
Çıktı yolu yönetimi.

Amaç: kullanıcı bir çıktı yolu yazmayı unutsa bile işlenen videonun
kaybolmaması. Çıktılar, kaynak videonun yanında kendi klasöründe ve hangi
videoya ait olduğu adından anlaşılacak şekilde toplanır.

Bu modül yalnızca yol HESAPLAR; diske bir şey yazmaz.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .config import (
    AUTO_SAVE_OUTPUT_VIDEO,
    OUTPUT_ADD_TIMESTAMP,
    OUTPUT_DIR_NAME,
    OUTPUT_NAME_SUFFIX,
    OUTPUT_STREAM_DIR_NAME,
    OUTPUT_VIDEO_EXTENSION,
)
from .file_utils import ensure_unique_path, sanitize_filename

CSV_EXTENSION: str = ".csv"


def is_file_source(source: str) -> bool:
    """Kaynağın diskteki bir video dosyası olup olmadığını söyler."""
    if source.isdigit():                 # Kamera indeksi (0, 1, ...)
        return False
    if "://" in source:                  # rtsp:// http:// vb. akış adresi
        return False
    return True


def stream_output_directory() -> str:
    """Kaynak dosya değilse (RTSP / kamera) kullanılan çıktı klasörü (cwd altında)."""
    return os.path.join(os.getcwd(), OUTPUT_STREAM_DIR_NAME)


def source_output_directory(source: str, override: Optional[str] = None) -> str:
    """
    Çıktıların yazılacağı dizini belirler.

    Öncelik sırası:
      1) --output-dir ile verilen dizin
      2) Kaynak bir video DOSYASI ise -> <videonun dizini>/<OUTPUT_DIR_NAME>
      3) Kaynak RTSP / kamera ise      -> <çalışma dizini>/<OUTPUT_STREAM_DIR_NAME>
    """
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if is_file_source(source):
        base = os.path.dirname(os.path.abspath(os.path.expanduser(source)))
        return os.path.join(base, OUTPUT_DIR_NAME)
    return stream_output_directory()


def build_output_stem(source: str) -> str:
    """
    Çıktı dosyalarının ortak gövde adını üretir (uzantısız).

        "giris.mp4" -> "giris_islenmis_20250109_142530"
        "0"         -> "kamera0_islenmis_20250109_142530"

    Video ve CSV aynı gövdeyi paylaşır; böylece bir koşuya ait dosyalar
    klasörde yan yana ve eşleşmiş halde durur.
    """
    if is_file_source(source):
        stem = os.path.splitext(os.path.basename(source))[0]
    elif source.isdigit():
        stem = f"kamera{source}"
    else:
        stem = "akis"
    stem = sanitize_filename(stem) + OUTPUT_NAME_SUFFIX
    if OUTPUT_ADD_TIMESTAMP:
        stem += time.strftime("_%Y%m%d_%H%M%S")
    return stem


def _looks_like_directory(raw_value: str, absolute_path: str) -> bool:
    """Kullanıcının verdiği değerin dosya mı klasör mü olduğunu tahmin eder."""
    if os.path.isdir(absolute_path):
        return True
    if raw_value.endswith(("/", "\\", os.sep)):
        return True
    return os.path.splitext(absolute_path)[1] == ""     # Uzantısız -> klasör


def _explicit_target(explicit: str, stem: str, extension: str) -> str:
    """
    Kullanıcının verdiği yolu mutlak yola çevirir. Yol bir klasör gibi
    görünüyorsa dosya adı `stem + extension` olarak eklenir.
    """
    target = os.path.abspath(os.path.expanduser(explicit))
    if _looks_like_directory(explicit, target):
        target = os.path.join(target, stem + extension)
    return target


def _auto_target(
    source: str, output_dir: Optional[str], stem: str, extension: str
) -> str:
    """Otomatik isimlendirilmiş, var olan dosyanın üzerine yazmayan yol üretir."""
    return ensure_unique_path(
        os.path.join(source_output_directory(source, output_dir), stem + extension)
    )


def resolve_output_video_path(
    source: str,
    explicit: Optional[str],
    output_dir: Optional[str],
    stem: str,
    auto_save: bool = AUTO_SAVE_OUTPUT_VIDEO,
) -> Optional[str]:
    """
    Anotasyonlu videonun yazılacağı tam yolu belirler.

    explicit  : --save-video değeri.
                None -> bayrak hiç verilmedi (otomatik kayda bakılır)
                ""   -> bayrak değersiz verildi (otomatik isimlendirme)
                yol  -> dosya ya da klasör yolu
    auto_save : Bayrak verilmese bile kaydın yapılıp yapılmayacağı.

    Kayıt istenmiyorsa None döner.
    """
    if explicit:
        return ensure_unique_path(
            _explicit_target(explicit, stem, OUTPUT_VIDEO_EXTENSION)
        )

    if explicit is None and not auto_save:
        return None

    return _auto_target(source, output_dir, stem, OUTPUT_VIDEO_EXTENSION)


def resolve_output_csv_path(
    source: str,
    explicit: Optional[str],
    output_dir: Optional[str],
    stem: str,
) -> Optional[str]:
    """
    CSV çıktı yolunu belirler. --csv hiç verilmediyse None döner; değersiz
    verildiyse video ile aynı klasöre aynı gövde adıyla yazılır.

    NOT: Elle verilen CSV yolu (video yolundan farklı olarak) benzersiz hale
    getirilmez; aynı adda dosya varsa üzerine yazılır. Orijinal davranış korunmuştur.
    """
    if explicit is None:
        return None
    if explicit:
        return _explicit_target(explicit, stem, CSV_EXTENSION)
    # Otomatik isimlendirmede sessizce üzerine yazmayı önle
    return _auto_target(source, output_dir, stem, CSV_EXTENSION)
