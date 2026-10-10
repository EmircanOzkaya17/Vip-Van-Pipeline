"""
Plaka metni doğrulama, karakter düzeltme ve çoğunluk oylaması.

Saf metin işleme modülüdür; görüntü veya model bağımlılığı yoktur. Bu sayede
OCR motoru değişse bile aynen yeniden kullanılabilir ve kolayca test edilebilir.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import (
    PLATE_ACCEPT_SINGLE_CONF,
    PLATE_CHAR_VOTE_ENABLED,
    PLATE_CHAR_VOTE_MIN_AGREEMENT,
    PLATE_CHAR_VOTE_MIN_CANDIDATES,
    PLATE_MIN_VOTES,
)

# Turk plaka formatlari (yonetmelik):
#   il kodu (2 rakam) + harf grubu (1-3 harf) + rakam grubu
# Gecerli kombinasyonlar:
#   1 harf + 4 rakam   |  1 harf + 5 rakam
#   2 harf + 3 rakam   |  2 harf + 4 rakam
#   3 harf + 2 rakam   |  3 harf + 3 rakam
# Toplam uzunluk her zaman 7 veya 8 karakterdir.
# NOT: Plakalarda Q, W, X ve Turkce'ye ozgu C, G, I, O, S, U (noktali/sapkali)
# harfleri KULLANILMAZ. Harf kumesini daraltmak yanlis duzeltmeleri azaltir.
PLATE_LETTERS: str = "ABCDEFGHIJKLMNOPRSTUVYZ"

PLATE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(rf"^(0[1-9]|[1-7][0-9]|8[01])[{PLATE_LETTERS}]\d{{4,5}}$"),
    re.compile(rf"^(0[1-9]|[1-7][0-9]|8[01])[{PLATE_LETTERS}]{{2}}\d{{3,4}}$"),
    re.compile(rf"^(0[1-9]|[1-7][0-9]|8[01])[{PLATE_LETTERS}]{{3}}\d{{2,3}}$"),
)

# OCR'da sık karışan karakterler (harf bölgesi <-> rakam bölgesi)
DIGIT_TO_LETTER: Dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B", "9": "G",
    # Plakada bulunmayan harfler -> gorsel olarak en yakin gecerli harf
    "Q": "O", "W": "V", "X": "K",
}
LETTER_TO_DIGIT: Dict[str, str] = {
    "O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8", "E": "8",
}

# Türkçe karakterlerin latin karşılıkları (OCR bazen bunları üretebilir)
TURKISH_TRANSLATION = str.maketrans("ÇĞİIÖŞÜçğıiöşü", "CGIIOSUCGIIOSU")


def normalize_ocr_text(raw_text: str) -> str:
    """OCR çıktısını büyük harfe çevirir, boşluk ve sembolleri temizler."""
    if not raw_text:
        return ""
    text = raw_text.strip().upper().translate(TURKISH_TRANSLATION)
    return re.sub(r"[^A-Z0-9]", "", text)


def _convert_chunk(chunk: str, mapping: Dict[str, str]) -> Tuple[str, int]:
    """
    Bir karakter bloğunu verilen dönüşüm tablosuna göre çevirir.
    Dönüş: (çevrilmiş_metin, yapılan_değişiklik_sayısı)
    """
    converted: List[str] = []
    changes = 0
    for char in chunk:
        replacement = mapping.get(char, char)
        if replacement != char:
            changes += 1
        converted.append(replacement)
    return "".join(converted), changes


# Plakanin solundaki mavi AB seridinde "TR" yazar. OCR bu seridi plakayla
# ayni satirda okuyup "TR34ABC123" (10 karakter) dondurebilir; eski kod
# 7-8 karakter disindaki her metni dogrudan reddediyordu.
COUNTRY_PREFIX: str = "TR"
# Bu uzunluga kadar olan metinlerde plaka, alt dizi olarak aranir.
# Daha uzun metinler (ör. aracin uzerindeki yazilar) rastlantisal eslesme
# riski tasidigi icin aranmaz.
MAX_SEARCH_TEXT_LENGTH: int = 12


def _validate_exact(text: str) -> Tuple[Optional[str], int]:
    """
    Tam olarak 7-8 karakterlik normalize metni konuma gore duzeltip dogrular.
    Donus: (plaka veya None, yapilan_degisiklik_sayisi)
    """
    if not 7 <= len(text) <= 8:
        return None, 0

    best_candidate: Optional[str] = None
    best_changes: Optional[int] = None

    for letter_count in (1, 2, 3):
        tail_length = len(text) - 2 - letter_count
        if not 2 <= tail_length <= 5:
            continue

        # Bölge 1: il kodu (rakam) | Bölge 2: harf grubu | Bölge 3: rakam grubu
        province, c1 = _convert_chunk(text[:2], LETTER_TO_DIGIT)
        letters, c2 = _convert_chunk(text[2:2 + letter_count], DIGIT_TO_LETTER)
        digits, c3 = _convert_chunk(text[2 + letter_count:], LETTER_TO_DIGIT)

        candidate = province + letters + digits
        if not any(pattern.match(candidate) for pattern in PLATE_PATTERNS):
            continue

        total_changes = c1 + c2 + c3
        if best_changes is None or total_changes < best_changes:
            best_candidate, best_changes = candidate, total_changes

    return best_candidate, best_changes or 0


def correct_and_validate_plate(raw_text: str) -> Optional[str]:
    """
    Regex + karakter düzeltme adımı.

    OCR metnini temizler, olası harf/rakam karışıklıklarını konuma göre
    düzeltir ve geçerli bir Türk plaka formatına oturuyorsa döner.

    Harf bloğunun uzunluğu (1-3) önceden bilinmediğinden tüm olasılıklar
    denenir. Birden fazla olasılık geçerli formata otursa bile OCR çıktısına
    EN AZ müdahale gerektiren aday seçilir; böylece gereksiz karakter
    değişimiyle yanlış plaka üretilmesi engellenir.

    Metin 8 karakterden uzunsa (ör. "TR34ABC123", "TR 06 AB 1234", sonda
    civata golgesinden gelen fazladan bir karakter) once bastaki "TR" ulke
    kodu atilir, ardindan 7-8 karakterlik alt diziler denenir. Secim
    sirasi: en az karakter duzeltmesi, sonra en az atilan karakter.

    Geçerli bir format bulunamazsa None döner.
    """
    text = normalize_ocr_text(raw_text)
    if len(text) < 7 or len(text) > MAX_SEARCH_TEXT_LENGTH:
        return None

    plate, _ = _validate_exact(text)
    if plate is not None:
        return plate

    sources = [text]
    if text.startswith(COUNTRY_PREFIX):
        sources.insert(0, text[len(COUNTRY_PREFIX):])

    best: Optional[Tuple[int, int, str]] = None   # (degisiklik, atilan, plaka)
    for source in sources:
        for length in (8, 7):
            for start in range(0, len(source) - length + 1):
                window = source[start:start + length]
                candidate, changes = _validate_exact(window)
                if candidate is None:
                    continue
                # Ulke kodu atilmis kaynakta "TR" atilan karakter sayilmaz;
                # boylece "TR" sonrasi tam eslesme her zaman one gecer.
                dropped = len(source) - length
                key = (changes, dropped, candidate)
                if best is None or key < best:
                    best = key
    return best[2] if best else None


def majority_vote_plate(
    candidates: List[Tuple[str, float]]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Majority Voting: birden fazla frame'den gelen plaka okumalarını birleştirir.

    Seçim kriteri: en çok tekrar eden metin; eşitlik durumunda toplam
    confidence değeri yüksek olan kazanır. Az oy alan sonuçlar ancak çok
    yüksek güven değerine sahipse kabul edilir.
    """
    if not candidates:
        return None, None

    vote_counts: Counter = Counter()
    confidence_sums: Dict[str, float] = {}
    for text, conf in candidates:
        vote_counts[text] += 1
        confidence_sums[text] = confidence_sums.get(text, 0.0) + conf

    best_text = max(
        vote_counts.keys(),
        key=lambda t: (vote_counts[t], confidence_sums[t]),
    )
    votes = vote_counts[best_text]
    mean_conf = confidence_sums[best_text] / votes

    if votes >= PLATE_MIN_VOTES or mean_conf >= PLATE_ACCEPT_SINGLE_CONF:
        return best_text, mean_conf
    if PLATE_CHAR_VOTE_ENABLED:
        return character_vote_plate(candidates)
    return None, None


def _layout(text: str) -> str:
    """Plakanin harf/rakam dizilimini doner: '34ABC123' -> 'DDLLLDDD'."""
    return "".join("D" if char.isdigit() else "L" for char in text)


def character_vote_plate(
    candidates: List[Tuple[str, float]]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Karakter bazli oylama: tam metin cogunlugu olusmadiginda devreye girer.

    Adaylar zaten format dogrulamasindan gecmis plakalardir. En sik gorulen
    harf/rakam diziliminden (ör. DDLLLDDD) olanlar alinir ve her konumda
    cogunluk karakteri secilir. Herhangi bir konumda yeterli uzlasma yoksa
    ya da sonuc gecerli bir plaka degilse None doner.
    """
    if len(candidates) < PLATE_CHAR_VOTE_MIN_CANDIDATES:
        return None, None

    layout_counts = Counter(_layout(text) for text, _ in candidates)
    layout, _ = layout_counts.most_common(1)[0]
    group = [(text, conf) for text, conf in candidates if _layout(text) == layout]
    if len(group) < PLATE_CHAR_VOTE_MIN_CANDIDATES:
        return None, None

    chars: List[str] = []
    for position in range(len(layout)):
        position_votes: Counter = Counter()
        position_weights: Dict[str, float] = {}
        for text, conf in group:
            char = text[position]
            position_votes[char] += 1
            position_weights[char] = position_weights.get(char, 0.0) + conf
        winner = max(position_votes,
                     key=lambda c: (position_votes[c], position_weights[c]))
        if position_votes[winner] / len(group) < PLATE_CHAR_VOTE_MIN_AGREEMENT:
            return None, None
        chars.append(winner)

    plate = "".join(chars)
    if not any(pattern.match(plate) for pattern in PLATE_PATTERNS):
        return None, None
    mean_conf = sum(conf for _, conf in group) / len(group)
    return plate, mean_conf
