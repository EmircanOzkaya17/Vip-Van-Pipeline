"""
Genel dosya sistemi yardımcıları.

Çıktı yolları, video yazıcı ve PaddleX model önbelleği gibi farklı bölümler
tarafından ortak kullanılan, alana (domain) özgü olmayan fonksiyonlar.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

LOGGER = logging.getLogger(__name__)

INVALID_FILENAME_CHARS: re.Pattern = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def is_ascii_path(path: Optional[str]) -> bool:
    """Verilen yolun tamamen ASCII karakterlerden oluşup oluşmadığını döner."""
    if not path:
        return False
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ensure_writable_directory(path: str) -> bool:
    """
    Dizini oluşturur ve gerçekten yazılabilir olduğunu doğrular.

    Yalnızca os.makedirs başarısı yeterli değildir (salt okunur disk, ağ
    paylaşımı, izin sorunları); bu yüzden küçük bir deneme dosyası yazılıp
    silinir.
    """
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
        return True
    except OSError as exc:
        LOGGER.debug("Dizin kullanilamadi (%s): %s", path, exc)
        return False


def sanitize_filename(name: str) -> str:
    """Dosya adında kullanılamayan karakterleri temizler."""
    cleaned = INVALID_FILENAME_CHARS.sub("_", name).strip(" .")
    return cleaned or "video"


def ensure_unique_path(path: str) -> str:
    """Aynı adda dosya varsa sonuna sayaç ekleyerek üzerine yazmayı önler."""
    if not os.path.exists(path):
        return path
    stem, extension = os.path.splitext(path)
    for index in range(1, 1000):
        candidate = f"{stem}_{index:03d}{extension}"
        if not os.path.exists(candidate):
            return candidate
    return path


def describe_saved_file(path: Optional[str]) -> Optional[str]:
    """Kaydedilen dosyayı 'yol (boyut)' biçiminde özetler; yoksa None döner."""
    if not path or not os.path.exists(path):
        return None
    try:
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    except OSError:
        return path
    return f"{path} ({size_mb:.1f} MB)"
