"""
PaddleOCR sarmalayıcısı.

Farklı PaddleOCR sürümlerinin (2.x / 3.x) imza ve çıktı biçimi farklarını tek
bir yerde toplar; dışarıya yalnızca (metin, güven) listesi döner.

paddleocr, yalnızca PlateOCR oluşturulduğunda içe aktarılır. Windows'ta ASCII
dışı kullanıcı klasörü sorununu çözen `ensure_ascii_model_cache()` bu importtan
ÖNCE çağrılır.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, List, Optional, Tuple

import numpy as np

from .config import (
    PADDLEOCR_DET_BOX_THRESH,
    PADDLEOCR_DET_LIMIT_SIDE_LEN,
    PADDLEOCR_DET_LIMIT_TYPE,
    PADDLEOCR_DET_THRESH,
    PADDLEOCR_DET_UNCLIP_RATIO,
    PADDLEOCR_LANG,
    PADDLEOCR_REC_SCORE_THRESH,
    PADDLEOCR_USE_ANGLE_CLS,
    PADDLEOCR_USE_DOC_PREPROCESS,
    PADDLEX_ASCII_FALLBACKS,
    PADDLEX_CACHE_HOME,
)
from .file_utils import ensure_writable_directory, is_ascii_path

LOGGER = logging.getLogger(__name__)

# [(metin, güven), ...]
OcrResult = List[Tuple[str, float]]


# =============================================================================
# MODEL ÖNBELLEĞİ VE YÜKLEME
# =============================================================================

def ensure_ascii_model_cache() -> None:
    """
    PaddleX model onbellegini gerekiyorsa ASCII bir dizine yonlendirir.

    Paddle'in C++ cikarim motoru (paddle_inference.create_predictor) dosya
    yollarini dar karakter (ANSI) API'si ile acar. Kullanici klasoru Turkce
    karakter iceriyorsa model dosyasi acilamaz; icerik bos gorunur ve
    "attempting to parse an empty input" JSON hatasi alinir. Model dosyalari
    aslinda saglam indirilmistir, sorun tamamen yol kodlamasindadir.

    Bu fonksiyon PADDLE_PDX_CACHE_HOME degiskenini ayarlar ve MUTLAKA
    paddleocr import edilmeden once cagrilmalidir.
    """
    if os.environ.get("PADDLE_PDX_CACHE_HOME"):
        return                                   # Kullanici zaten belirlemis

    default_home = os.path.expanduser("~")
    if is_ascii_path(default_home) and not PADDLEX_CACHE_HOME:
        return                                   # Varsayilan yol zaten sorunsuz

    # Aday dizinler: once elle verilen, sonra bilinen ASCII konumlar
    candidates: List[str] = []
    if PADDLEX_CACHE_HOME:
        candidates.append(PADDLEX_CACHE_HOME)
    if os.name == "nt":
        public_dir = os.environ.get("PUBLIC")
        if public_dir:
            candidates.append(os.path.join(public_dir, "paddlex_cache"))
        candidates.extend(PADDLEX_ASCII_FALLBACKS)
    candidates.append(os.path.join(tempfile.gettempdir(), "paddlex_cache"))

    for candidate in candidates:
        if not is_ascii_path(candidate):
            continue
        if not ensure_writable_directory(candidate):
            continue                             # Yazilamiyor -> sonraki aday

        os.environ["PADDLE_PDX_CACHE_HOME"] = candidate
        LOGGER.warning(
            "Kullanici dizini ASCII disi karakter iceriyor (%s). "
            "PaddleX model onbellegi su dizine yonlendirildi: %s",
            default_home, candidate,
        )
        return

    LOGGER.error(
        "ASCII bir model onbellek dizini olusturulamadi. PaddleOCR "
        "yuklenirken 'empty input' hatasi alinabilir. Cozum: "
        "PADDLE_PDX_CACHE_HOME degiskenini elle ayarlayin."
    )


def _create_paddle_ocr(lang: str = PADDLEOCR_LANG):
    """
    PaddleOCR örneğini oluşturur.
    Farklı PaddleOCR sürümlerinin imza değişikliklerine karşı toleranslıdır.
    """
    # ONEMLI: Bu cagri paddleocr import edilmeden ONCE yapilmalidir,
    # aksi halde PaddleX onbellek yolunu import aninda sabitler.
    ensure_ascii_model_cache()

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "paddleocr kurulu degil. Kurulum: pip install paddlepaddle paddleocr"
        ) from exc

    LOGGER.info("PaddleOCR yukleniyor (lang=%s)", lang)
    # Sirasiyla denenir: once PaddleOCR 3.x (gereksiz alt modeller kapali),
    # ardindan 2.x imzalari. Ilk basarili olan kullanilir.
    doc_flag = PADDLEOCR_USE_DOC_PREPROCESS
    init_variants = (
        # PaddleOCR 3.x - belge on isleme ve satir yonu kapali
        {"lang": lang,
         "use_doc_orientation_classify": doc_flag,
         "use_doc_unwarping": doc_flag,
         "use_textline_orientation": PADDLEOCR_USE_ANGLE_CLS},
        # PaddleOCR 3.x - yalnizca satir yonu parametresi destekleniyorsa
        {"lang": lang,
         "use_textline_orientation": PADDLEOCR_USE_ANGLE_CLS},
        # PaddleOCR 2.x
        {"lang": lang, "use_angle_cls": PADDLEOCR_USE_ANGLE_CLS,
         "show_log": False},
        {"lang": lang, "use_angle_cls": PADDLEOCR_USE_ANGLE_CLS},
        {"lang": lang},
    )
    last_error: Optional[Exception] = None
    for kwargs in init_variants:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            # Desteklenmeyen parametre veya surum farki -> bir sonrakini dene
            last_error = exc
            LOGGER.debug("PaddleOCR imzasi basarisiz (%s): %s",
                         sorted(kwargs), exc)
            continue
    raise RuntimeError(
        f"PaddleOCR baslatilamadi: {last_error}\n"
        "Ipucu: model onbellegi bozulmus olabilir. "
        "~/.paddlex/official_models klasorunu silip tekrar deneyin."
    )


# =============================================================================
# ÇIKTI AYRIŞTIRMA
# =============================================================================

def _result_field(result: Any, key: str) -> Any:
    """Sozluk benzeri sonuc nesnelerinden guvenli alan okuma."""
    try:
        return result[key]
    except Exception:
        return getattr(result, key, None)


def parse_ocr_result(raw_result: Any) -> OcrResult:
    """
    PaddleOCR ciktisini (metin, guven) listesine donusturur.

    Uc farkli format desteklenir:
      * 3.x OCR pipeline    -> rec_texts / rec_scores  (COGUL)
      * 3.x TextRecognition -> rec_text  / rec_score   (TEKIL)
      * 2.x                 -> [[box, (text, score)], ...]

    Tekil/cogul ayrimi kritiktir: yalnizca-tanima modu TEKIL anahtar
    kullanir; sadece cogul anahtar aranirsa basarili bir okuma bile
    sessizce bos gorunur ve "OCR metin dondurmedi" sanilir.
    """
    parsed: OcrResult = []
    if raw_result is None:
        return parsed
    try:
        pages = list(raw_result)
    except TypeError:
        pages = [raw_result]

    for page in pages:
        if page is None:
            continue

        # --- 3.x cogul anahtarlar (OCR pipeline) ---
        texts = _result_field(page, "rec_texts")
        if texts is not None:
            scores = _result_field(page, "rec_scores") or []
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 0.0
                parsed.append((str(text), score))
            continue

        # --- 3.x tekil anahtarlar (TextRecognition modulu) ---
        text = _result_field(page, "rec_text")
        if text is not None:
            score = _result_field(page, "rec_score")
            if isinstance(text, (list, tuple)):
                score_list = score if isinstance(score, (list, tuple)) else []
                for index, item in enumerate(text):
                    value = float(score_list[index]) if index < len(score_list) else 0.0
                    parsed.append((str(item), value))
            else:
                parsed.append(
                    (str(text), float(score) if score is not None else 0.0)
                )
            continue

        # --- 2.x liste formati ---
        if isinstance(page, (list, tuple)):
            for line in page:
                try:
                    parsed.append((str(line[1][0]), float(line[1][1])))
                except Exception:
                    continue
    return parsed


# =============================================================================
# OCR MOTORU
# =============================================================================

class PlateOCR:
    """
    Plaka kesitleri için PaddleOCR motoru.

    `read()` önce tam boru hattını (metin tespiti + tanıma), sonuç boşsa
    yalnızca-tanıma modunu dener. İki aşama tanı amacıyla ayrı ayrı da
    çağrılabilir (bkz. `--test-ocr`).
    """

    def __init__(self, lang: str = PADDLEOCR_LANG) -> None:
        self.engine = _create_paddle_ocr(lang)
        self._recognizer = None              # Tembel olusturulan TextRecognition
        self._recognizer_failed = False
        self._failure_reported = False       # Ilk OCR hatasi WARNING olarak gosterilir

    # ------------------------------------------------------------------ #
    def read(self, image: np.ndarray) -> OcrResult:
        """
        Once tam boru hatti (metin tespiti + tanima) denenir. Sonuc bos gelirse
        yalnizca-tanima moduna dusulur: kesit zaten plakaya kirpildigi icin
        metin tespiti asamasi gereksizdir ve kucuk/bulanik goruntulerde
        basarisiz olan asama genellikle tam olarak budur.
        """
        results = self.detect_and_recognize(image)
        if not results:
            results = self.recognize_only(image)
        return results

    # ------------------------------------------------------------------ #
    def detect_and_recognize(self, image: np.ndarray) -> OcrResult:
        """
        Metin tespiti + tanima birlikte calistirilir.

        Esikler plaka kesitine gore gevsetilerek gecirilir; desteklenmeyen
        surumler icin kademeli olarak sade cagrilara dusulur.
        """
        detection_kwargs = {
            "text_det_limit_side_len": PADDLEOCR_DET_LIMIT_SIDE_LEN,
            "text_det_limit_type": PADDLEOCR_DET_LIMIT_TYPE,
            "text_det_thresh": PADDLEOCR_DET_THRESH,
            "text_det_box_thresh": PADDLEOCR_DET_BOX_THRESH,
            "text_det_unclip_ratio": PADDLEOCR_DET_UNCLIP_RATIO,
            "text_rec_score_thresh": PADDLEOCR_REC_SCORE_THRESH,
        }
        # Sirasiyla: 3.x esikli -> 3.x sade -> 2.x ocr()
        strategies = (
            ("predict+esikler", lambda: self.engine.predict(image, **detection_kwargs)),
            ("predict", lambda: self.engine.predict(image)),
            ("ocr", lambda: self.engine.ocr(image)),
        )
        for name, call in strategies:
            try:
                results = parse_ocr_result(call())
            except Exception as exc:
                self._log_failure(f"{name} basarisiz: {exc}")
                continue
            if results:
                return results
        return []

    # ------------------------------------------------------------------ #
    def recognize_only(self, image: np.ndarray) -> OcrResult:
        """
        Yalnizca-tanima modu: goruntunun tamaminin tek bir metin satiri
        oldugu varsayilir. Plaka kesiti icin ideal, cunku konum zaten bilinir.
        """
        # --- PaddleOCR 3.x: TextRecognition modulu ---
        recognizer = self._get_recognizer()
        if recognizer is not None:
            try:
                results = parse_ocr_result(recognizer.predict(image))
                if results:
                    return results
            except Exception as exc:
                self._log_failure(f"yalnizca-tanima basarisiz: {exc}")

        # --- PaddleOCR 2.x: ocr(..., det=False) ---
        try:
            return parse_ocr_result(self.engine.ocr(image, det=False))
        except Exception as exc:
            LOGGER.debug("det=False modu kullanilamadi: %s", exc)
        return []

    # ------------------------------------------------------------------ #
    def _get_recognizer(self):
        """TextRecognition modelini tembel (lazy) olarak olusturur."""
        if self._recognizer is not None or self._recognizer_failed:
            return self._recognizer
        try:
            from paddleocr import TextRecognition
            self._recognizer = TextRecognition()
            LOGGER.info("Yalnizca-tanima modu hazir (TextRecognition).")
        except Exception as exc:
            self._recognizer_failed = True
            LOGGER.debug("TextRecognition kullanilamiyor: %s", exc)
        return self._recognizer

    # ------------------------------------------------------------------ #
    def _log_failure(self, message: str) -> None:
        """
        OCR hatalarini loglar. Ilk hata WARNING seviyesindedir; boylece
        --verbose olmadan da kullanici sorunu fark eder. Sonrakiler her
        frame'de tekrarlanmamasi icin DEBUG seviyesine duser.
        """
        if not self._failure_reported:
            self._failure_reported = True
            LOGGER.warning("OCR sorunu: %s", message)
            LOGGER.warning("Ayrinti icin --verbose ile calistirin.")
        else:
            LOGGER.debug("OCR sorunu: %s", message)
