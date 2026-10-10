"""
Uygulama konfigürasyonu.

Bu modül iki tür ayarı tek bir yerde toplar:

1) Ortama bağlı değerler (model dosyaları / adları, dizin yolları)
   `.env` dosyasından veya gerçek ortam değişkenlerinden okunur. Tanımsız ya da
   boş bırakılmışsa koddaki varsayılan kullanılır.
   Öncelik: gerçek ortam değişkeni > .env > koddaki varsayılan.

2) Algoritma eşikleri ve ayar parametreleri
   Sistemin davranışını belirleyen ve kodla birlikte sürümlenmesi gereken
   sabitler. "Tüm ayarlanabilir parametreler tek bir yerde" ilkesi korunmuştur;
   her değerin gerekçesi yanında açıklanmıştır.

Bu modül ağır kütüphanelere (cv2, torch vb.) bağımlı değildir.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# =============================================================================
# .ENV YÜKLEME
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> None:
    """
    .env dosyasındaki değerleri ortam değişkenlerine yükler.

    Önceden tanımlı ortam değişkenleri ezilmez. python-dotenv kurulu değilse
    yalnızca gerçek ortam değişkenleri ve koddaki varsayılanlar kullanılır.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        if path.is_file():
            logging.getLogger(__name__).warning(
                "python-dotenv kurulu degil, %s okunmadi "
                "(kurulum: pip install python-dotenv).", path,
            )
        return
    load_dotenv(path, override=False)


def _env_str(name: str, default: str) -> str:
    """Ortam değişkenini okur; tanımsız veya boşsa varsayılanı döner."""
    return os.environ.get(name, "").strip() or default


def _env_optional_str(name: str) -> Optional[str]:
    """Ortam değişkenini okur; tanımsız veya boşsa None döner."""
    return os.environ.get(name, "").strip() or None


_load_env_file(ENV_FILE)


# =============================================================================
# YOLO / TESPİT
# =============================================================================
YOLO_MODEL_PATH: str = _env_str("YOLO_MODEL_PATH", "yolo11m.pt")  # (.env)
YOLO_CONF_THRESHOLD: float = 0.35          # Minimum tespit güveni
YOLO_IOU_THRESHOLD: float = 0.50           # NMS IoU eşiği
YOLO_IMAGE_SIZE: int = 960                 # Inference çözünürlüğü
# FP16 kullanımı. Yalnızca --device ile CUDA cihazı AÇIKÇA verildiğinde
# devreye girer; --device boş (otomatik) bırakılırsa FP32 ile çalışılır.
YOLO_HALF_PRECISION: bool = True

# Yalnızca bu COCO sınıfları işlenir; diğer tüm sınıflar göz ardı edilir.
COCO_VEHICLE_CLASSES: Dict[int, str] = {2: "car", 5: "bus", 7: "truck"}

MIN_BOX_AREA: int = 900                    # Çok küçük kutuları ele (px^2)

# =============================================================================
# TAKİP
# =============================================================================
TRACKER_CONFIG: str = _env_str("TRACKER_CONFIG", "botsort.yaml")  # (.env)
CLASS_VOTE_FRAMES: int = 7                 # İlk 7 frame boyunca sınıf oylaması
TRACK_TIMEOUT_FRAMES: int = 30             # Bu kadar frame görünmezse track kapanır

# =============================================================================
# SANAL ÇİZGİ
# =============================================================================
# Referans görseldeki mavi çizginin piksel koordinatları ve referans çözünürlük.
# Farklı çözünürlükte video verilirse koordinatlar otomatik ölçeklenir.
LINE_REFERENCE_RESOLUTION: Tuple[int, int] = (704, 396)   # (genişlik, yükseklik)
LINE_REFERENCE_START: Tuple[int, int] = (43, 222)         # Çizgi alt uç noktası
LINE_REFERENCE_END: Tuple[int, int] = (165, 108)          # Çizgi üst uç noktası

# Çizgi segmentinin iki ucundan yüzde kaç uzatılarak geçiş kabul edileceği.
LINE_SEGMENT_EXTENSION: float = 0.45
# Geçiş yönü: "neg_to_pos" (beklenen yönde), "pos_to_neg" (ters yön), "any".
LINE_CROSS_DIRECTION: str = "neg_to_pos"

# Araçların kadrajdaki beklenen ilerleme yönü. Referans senaryoda araçlar
# soldan girip sağa doğru ilerlediği için (1, 0) kullanılır.
# Bu vektör sayesinde, çizginin uç noktaları hangi sırada verilirse verilsin
# (ör. --pick-line ile ters sırada tıklansın) geçiş yönü doğru hesaplanır.
LINE_EXPECTED_TRAVEL_DIRECTION: Tuple[float, float] = (1.0, 0.0)
LINE_AUTO_ORIENT: bool = True
# Çizgi geçişi zorunlu mu? False ise araç N frame sonra yine sınıflandırılır.
REQUIRE_LINE_CROSSING: bool = True

# =============================================================================
# OPENCLIP
# =============================================================================
CLIP_MODEL_NAME: str = _env_str("CLIP_MODEL_NAME", "ViT-B-32")               # (.env)
CLIP_PRETRAINED: str = _env_str("CLIP_PRETRAINED", "laion2b_s34b_b79k")      # (.env)

# Çizgiyi geçtikten sonra sınıflandırmada kullanılacak frame offsetleri.
# Geçiş frame'i sayılmaz; (2,) => geçişten sonraki 2. frame kullanılır.
# Çoklu frame ile ortalama almak istenirse örn. (2, 4, 6) yapılabilir.
CLIP_FRAME_OFFSETS: Tuple[int, ...] = (2,)

# Araç kutusu CLIP'e gönderilmeden önce oransal olarak genişletilir.
# NOT: Aynı pay, plaka zincirine giden araç kesitinde de kullanılır.
CLIP_CROP_PADDING: float = 0.06
CLIP_MIN_CROP_SIZE: int = 32               # Bu boyutun altındaki cropler atlanır

# Skor toplama yöntemi:
#   "softmax_group_sum" -> softmax olasılıkları grup bazında toplanır [0..1]
#                          (0.80 / 0.25 eşikleri bu ölçekte anlamlıdır)
#   "mean"              -> ham cosine benzerliklerinin grup ortalaması
CLIP_SCORE_MODE: str = "softmax_group_sum"
CLIP_LOGIT_SCALE: float = 100.0            # Softmax öncesi benzerlik ölçeği

# Nihai karar eşikleri
FINAL_POSITIVE_THRESHOLD: float = 0.80
FINAL_MARGIN_THRESHOLD: float = 0.25
TARGET_VEHICLE_TYPE: str = "VIP Passenger Shuttle Van"
NON_TARGET_VEHICLE_TYPE: str = "non-target"

# Track kaybolurken hâlâ sınıflandırılmadıysa son crop ile karar ver.
CLIP_FALLBACK_ON_TRACK_END: bool = True

# Prompt şablonları (prompt ensembling -> daha kararlı skorlar)
CLIP_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}",
    "a street surveillance photo of a {}",
    "a {} seen from the side on a road",
)

# Pozitif prompt grubu: insan taşımacılığı yapan büyük yolcu vanları
POSITIVE_PROMPTS: Tuple[str, ...] = (
    "Mercedes-Benz Vito passenger van",
    "Volkswagen Transporter passenger van",
    "Volkswagen Caravelle minibus",
    "Ford Tourneo Custom passenger van",
    "Renault Trafic passenger van",
    "Fiat Scudo passenger van",
    "passenger shuttle van",
    "people carrier van with side windows",
    "VIP shuttle van",
    "passenger van",
    "commuter van",
)

# Negatif prompt grubu: hedef olmayan araç tipleri
NEGATIVE_PROMPTS: Tuple[str, ...] = (
    "sedan car",
    "hatchback car",
    "SUV",
    "crossover car",
    "pickup truck",
    "cargo van without windows",
    "delivery van",
    "panel van",
    "truck",
    "city bus",
    "motorcycle",
)

# =============================================================================
# PLAKA TESPİTİ / OCR
# =============================================================================
PLATE_READING_ENABLED: bool = True
PLATE_FRAME_COUNT: int = 7                 # Çizgi geçişinden sonraki ilk 7 frame

# --- Model dosyalari (proje icindeki models/ klasoru) ---
# Plaka zinciri varsayilan olarak YALNIZCA OpenCV DNN ile calisir; onnxruntime,
# open-image-models ve PaddleOCR GEREKMEZ. Windows'ta onnxruntime torch'tan
# sonra yuklenince DLL catismasina dusebiliyor ya da hic kurulu olmayabiliyor;
# bu durum plaka okuyucuyu tamamen kapatiyordu. Modeller hazir olarak
# models/ klasorunde gelir (yeniden uretmek: tools/prepare_plate_models.py).
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Dedektor: open-image-models YOLOv9-t 384 plaka modeli (NMS'i ayrilmis hali).
#   "auto"        -> models/ icindeki dosya varsa OpenCV, yoksa onnxruntime
#   "opencv"      -> yalnizca OpenCV DNN
#   "onnxruntime" -> eski yol (open_image_models paketi)
PLATE_DETECTOR_BACKEND: str = _env_str("PLATE_DETECTOR_BACKEND", "auto")     # (.env)
PLATE_DETECTOR_ONNX: str = _env_str(                                         # (.env)
    "PLATE_DETECTOR_ONNX", str(MODELS_DIR / "plate_det_yolov9t384_raw.onnx")
)
PLATE_DETECTOR_INPUT_SIZE: int = 384       # Modelin egitim girdi boyutu
PLATE_DETECTION_NMS_IOU: float = 0.50      # OpenCV yolunda NMS IoU esigi
PLATE_DETECTOR_MODEL: str = _env_str(                                        # (.env)
    "PLATE_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end"
)                                          # yalnizca "onnxruntime" yolunda
PLATE_DETECTION_CONF: float = 0.25         # Plaka tespiti minimum güven

# OCR motorlari (sirayla denenir; ilki gecerli plaka verirse digeri calismaz)
#   "fastplate" -> fast-plate-ocr CCT-S v2 (plakaya OZEL egitilmis, 65 ulke,
#                  Turkiye dahil). OpenCV DNN ile calisir, ~5 MB, cok hizli.
#   "paddle"    -> PaddleOCR (genel amacli). Kuruluysa ikinci gorus olarak
#                  eklenebilir: PLATE_OCR_ENGINES=fastplate,paddle
PLATE_OCR_ENGINES: Tuple[str, ...] = tuple(
    name.strip().lower()
    for name in _env_str("PLATE_OCR_ENGINES", "fastplate").split(",")
    if name.strip()
)                                                                            # (.env)
PLATE_OCR_FAST_ONNX: str = _env_str(                                         # (.env)
    "PLATE_OCR_FAST_ONNX", str(MODELS_DIR / "plate_ocr_cct_s_v2_cv.onnx")
)
PLATE_OCR_FAST_ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PLATE_OCR_FAST_PAD_CHAR: str = "_"
PLATE_OCR_FAST_SLOTS: int = 10
PLATE_OCR_FAST_INPUT_SIZE: Tuple[int, int] = (128, 64)   # (genislik, yukseklik)
PLATE_CROP_PADDING: float = 0.08           # Plaka kutusu genişletme oranı
PLATE_MIN_CROP_WIDTH: int = 24             # OCR için minimum plaka genişliği

# --- Dedektor kutusu geometri suzgeci ---
# Plaka dedektoru zaman zaman JANT / FAR / TAMPON ARALIGI gibi bolgeleri de
# plaka sanip yuksek guvenle dondurur (olcum: bir jant 0.625 guvenle plaka
# olarak isaretlendi). Bu kutular OCR'a girdiginde hem bosuna zaman harcanir
# hem de dusuk guvenli cop metinler uretilir.
#
# Fiziksel gercek: TR plakasi 52x11 cm -> en/boy ~4.7. Plaka kadrajda theta
# kadar dondugunde eksene hizali kutunun orani
#       (4.7*cos + sin) / (4.7*sin + cos)
# olur ve bu ifade theta=45 derecede TAM 1.0'a iner, altina asla inmez.
# Yani bir plakanin kutusu hicbir acida "dikey" (oran < 1) olamaz.
# Jant tam da bu yuzden elenir: oran 0.57.
# Yukaridaki hesap yalnizca DUZLEMDE donmeyi kapsar. Arac yana donerken
# (yaw) plakanin genisligi de kisalir; videoda donus anindaki plaka kutusu
# 0.86 olculdu. Bu yuzden alt sinir 0.90'dan 0.75'e cekildi; jant (0.57)
# hala elenir.
PLATE_BOX_MIN_RATIO: float = 0.75
PLATE_BOX_MAX_RATIO: float = 8.00          # Asiri uzun seritler (tampon cizgisi)
# Alt sinir 0.07'den 0.04'e cekildi: arac yandan / capraz gorundugunde kutu
# genisligi aracin BOYUNU da icerir. ~60 derece acida 5 m'lik bir van icin
# kutu ~5.3 m, plakanin izdusumu ~0.26 m -> oran ~0.05; 0.07 bu plakayi
# eliyordu. Jant/far zaten oran (1.) kriterine takilir.
PLATE_BOX_MIN_WIDTH_FRAC: float = 0.04     # Arac kutusu genisligine gore alt sinir
PLATE_BOX_MAX_WIDTH_FRAC: float = 0.60     # ... ve ust sinir
PLATE_BOX_MIN_Y_FRAC: float = 0.30         # Plaka aracin ust ucte birinde olmaz
PLATE_MAX_CANDIDATES: int = 2              # En guvenilir N kutu OCR'a gider

# --- Yakinlastirmali ikinci tespit gecisi ---
# Dedektor arac kesitini 384 piksele kucultur; 640 px genisligindeki bir
# arac kesitinde 75 px'lik plaka ~45 px'e iner ve egik plakalarda guven
# esigin altina duser. Ilk gecis plaka bulamazsa aracin ALT bolgesi (plaka
# ust ucte birde olmaz) ust uste binen 3 dosemeye bolunur ve her biri ayri
# taranir: efektif cozunurluk ~2 kat artar. Yalnizca iska durumunda calisir.
PLATE_DETECT_ZOOM_ON_MISS: bool = True
PLATE_ZOOM_TOP_FRAC: float = 0.25          # Bu oranin ustu taranmaz
PLATE_ZOOM_TILE_FRAC: float = 0.50         # Doseme genisligi / arac genisligi
PLATE_ZOOM_TILES: int = 3                  # Doseme sayisi (%50 ust uste binme)

# --- Plaka bolgesi iyilestirme (refinement) ---
# Dedektorun dondurdugu kutu eksene hizalidir; plaka egik durdugunda kutu
# neredeyse KARE gelir ve icinde plakanin yanisira gril/tampon parcalari kalir.
# Bu adim kutu icindeki gercek plaka govdesini izole eder ve perspektif
# donusumu ile yatay hale getirir. OCR basarisini ciddi olcude artirir.
#
# ONEMLI: iyilestirme YANLIS bir bolge secerse saglam bir kesit copa doner.
# Bu yuzden aday bolge; oran + DOLGU ORANI (govde ne kadar dolu) + alan payi +
# merkez konumu olmak uzere dort kriterden gecmek zorundadir. Hicbiri gecmezse
# fonksiyon None doner ve cagiran taraf orijinal kesitle devam eder.
PLATE_REFINE_REGION: bool = True
PLATE_REFINE_WORK_SIZE: int = 220          # Kesit bu uzun kenara buyutulerek islenir
PLATE_REFINE_MIN_RATIO: float = 2.0        # Kabul edilen en dusuk en/boy orani
PLATE_REFINE_MAX_RATIO: float = 7.0        # Kabul edilen en yuksek en/boy orani
PLATE_REFINE_IDEAL_RATIO: float = 4.7      # TR plakasi 52x11 cm -> ~4.7
PLATE_REFINE_MIN_FILL: float = 0.68        # Kontur alani / dikdortgen alani
PLATE_REFINE_MIN_AREA_FRAC: float = 0.06   # Bolge, kesidin en az bu kadari olmali
PLATE_REFINE_MAX_AREA_FRAC: float = 0.92   # ... ve en fazla bu kadari
PLATE_REFINE_CENTER_BAND: float = 0.15     # Merkez kesidin kenarina bu kadar yakin olamaz
PLATE_REFINE_MARGIN_X: float = 1.06        # Izole edilen bolgeye yatay pay
PLATE_REFINE_MARGIN_Y: float = 1.22        # Izole edilen bolgeye dikey pay
# Iyilestirme basarili olsa bile orijinal kesit de OCR'a gonderilsin mi?
# Acik olmasi, iyilestirmenin nadiren yanildigi durumlarda emniyet kemeridir.
PLATE_OCR_TRY_BOTH_CROPS: bool = True
# Tam boru hatti (tespit+tanima) METIN DONDURDU ama gecerli plaka cikmadiysa
# yalnizca-tanima modu da denensin mi? Tespit modeli TR seridini / civata
# golgesini ayri kutu yapip plakayi parcalayabiliyor; kesit zaten plakaya
# kirpildigi icin tek satir tanima cogu zaman dogru sonucu verir.
PLATE_OCR_RECOGNIZE_ON_REGEX_FAIL: bool = True

# --- OCR oncesi olcekleme ---
# ONEMLI: Olcekleme GENISLIGE gore yapilir. Kutu kare seklinde geldiginde
# yukseklige bakan bir esik hic devreye girmez ve metin 12-15 piksel kalarak
# PaddleOCR'in metin tespit modelini bosa dusurur.
PLATE_OCR_MIN_WIDTH: int = 320             # OCR oncesi hedeflenen minimum genislik
PLATE_OCR_TARGET_HEIGHT: int = 96          # OCR oncesi hedeflenen minimum yukseklik
PLATE_OCR_MAX_UPSCALE: float = 8.0         # Buyutme tavani (asiri bulanikligi onler)

# OCR guven esikleri.
# ONEMLI: guven kontrolu artik regex'ten SONRA yapilir. Bir metin, karakter
# duzeltmesinin ardindan gecerli bir TR plaka formatina oturuyorsa bu, tek
# basina cok guclu bir kanittir; ham OCR guveni 0.55 istemek 90 piksellik bir
# plakada dogru okumalari da cope atar. Format dogrulanmissa daha dusuk esik
# uygulanir, gurultuyu zaten majority voting temizler.
OCR_MIN_CONFIDENCE: float = 0.55           # Formata oturmayan metinler icin
OCR_FORMAT_VALID_MIN_CONF: float = 0.35    # Formata oturan metinler icin
PLATE_MIN_VOTES: int = 2                   # Majority voting için minimum oy
PLATE_ACCEPT_SINGLE_CONF: float = 0.85     # Tek oyu kabul ettiren yüksek güven
# Tam metin cogunlugu yoksa ayni formattaki adaylar KARAKTER KARAKTER oylanir.
# Ornek: 34ABC123 / 34ABC128 / 34A8C123 -> her konumda cogunluk -> 34ABC123.
# Tek bir karakterin titremesi yuzunden tum okumalarin cope gitmesini onler.
PLATE_CHAR_VOTE_ENABLED: bool = True
PLATE_CHAR_VOTE_MIN_CANDIDATES: int = 3    # Karakter oylamasi icin en az aday
PLATE_CHAR_VOTE_MIN_AGREEMENT: float = 0.5 # Her konumda kazanan oy payi (>=)

# --- Uzatilmis plaka aramasi (varsayilan ACIK; --no-extended-plate ile kapanir) ---
# Sanal cizgi kadrajin girisinde oldugu icin, cizgiyi gectikten hemen sonraki
# 7 frame araclarin EN UZAK ve EN KUCUK gorundugu andir; plaka o noktada
# genellikle OCR icin fazla dusuk cozunurluklu kalir. Bu mod acikken ilk 7
# kare her zaman kullanilir; bu kareler gecerli bir plaka URETMEDIYSE arac
# kameraya yaklastikca denemeye devam edilir. Ek maliyet yalnizca hedef
# araclarda (VIP van) ve yalnizca plaka okunamadigi surece odenir.
PLATE_EXTENDED_SEARCH: bool = True
PLATE_EXTENDED_STRIDE: int = 3             # Uzatilmis modda her N frame'de bir dene
PLATE_EXTENDED_MAX_ATTEMPTS: int = 40      # Track basina toplam deneme tavani

PADDLEOCR_LANG: str = _env_str("PADDLEOCR_LANG", "en")                        # (.env)
# Metin satiri yon sinifi: plaka metni ters durmadigi icin gereksiz.
# Kapali olmasi hem bir modelin daha inmesini engeller hem de hizlandirir.
PADDLEOCR_USE_ANGLE_CLS: bool = False
# Belge yonu tespiti (PP-LCNet_x1_0_doc_ori) ve belge duzlestirme (unwarping)
# modelleri taranmis evraklar icindir; plaka kirpintisinda hicbir faydasi yoktur.
# Kapatmak PaddleOCR 3.x'te yukleme hatalarinin da onune gecer.
PADDLEOCR_USE_DOC_PREPROCESS: bool = False

# PaddleX model onbellek dizini (PADDLE_PDX_CACHE_HOME).
# Paddle'in C++ cikarim katmani ASCII olmayan dosya yollarini acamaz. Kullanici
# klasoru Turkce karakter iceriyorsa (ör. C:\Users\MELSA ÖZKAYA) model dosyasi
# acilamaz, bos akis olarak okunur ve su hata alinir:
#   "[json.exception.parse_error.101] ... attempting to parse an empty input"
# Bu durumda onbellek asagidaki ASCII yollardan ilk yazilabilir olana tasinir.
# Elle bir yol vermek isterseniz .env icindeki PADDLEX_CACHE_HOME degerini doldurun.
PADDLEX_CACHE_HOME: Optional[str] = _env_optional_str("PADDLEX_CACHE_HOME")  # (.env)
PADDLEX_ASCII_FALLBACKS: Tuple[str, ...] = (
    r"C:\Users\Public\paddlex_cache",
    r"C:\paddlex_cache",
)

# --- PaddleOCR metin tespiti esikleri ---
# PaddleOCR'in varsayilanlari belge taramasi icin ayarlanmistir ve kucuk,
# bulanik plaka kesitlerinde fazla katidir. Ozellikle box_thresh=0.60:
# metin kutusu bu barajı gecemezse TANIMA HIC CALISMAZ ve sonuc bos doner
# ("OCR metin dondurmedi"). Asagidaki degerler plaka icin gevsetilmistir.
# Parantez icindekiler PaddleOCR'in kendi varsayilanlaridir.
PADDLEOCR_DET_LIMIT_SIDE_LEN: int = 320    # (64)  kisa kenar en az bu kadar olur
PADDLEOCR_DET_LIMIT_TYPE: str = "min"      # (min) "min" = kucuk goruntuyu buyutur
PADDLEOCR_DET_THRESH: float = 0.20         # (0.30) piksel bazli esik
PADDLEOCR_DET_BOX_THRESH: float = 0.30     # (0.60) kutu bazli esik - asil darbogaz
PADDLEOCR_DET_UNCLIP_RATIO: float = 2.0    # (1.5)  kutuyu genislet, karakter kirpma
PADDLEOCR_REC_SCORE_THRESH: float = 0.0    # (0.0)  tanima sonucu eleme yok

# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================
DRAW_NON_TARGET_BOXES: bool = True         # Hedef olmayanlar da çizilsin mi?
COLOR_LINE: Tuple[int, int, int] = (255, 200, 0)      # BGR - mavi/cyan çizgi
COLOR_PENDING: Tuple[int, int, int] = (0, 215, 255)   # Karar bekleyen (sarı)
COLOR_TARGET: Tuple[int, int, int] = (0, 230, 0)      # Hedef araç (yeşil)
COLOR_NON_TARGET: Tuple[int, int, int] = (150, 150, 150)  # Hedef değil (gri)
COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)

# =============================================================================
# ÇIKTI DOSYALARI (OTOMATİK KAYIT)
# =============================================================================
# İşlenen (anotasyonlu) video, hiçbir bayrak verilmese bile kaynak videonun
# bulunduğu dizin altındaki bir alt klasöre kaydedilir:
#
#   D:\kayitlar\giris.mp4
#     -> D:\kayitlar\islenmis\giris_islenmis_20250109_142530.mp4
#
#   --save-video YOL  : dosya veya dizin yolunu elle belirler
#   --output-dir  YOL : tüm çıktıların yazılacağı klasörü değiştirir
#   --no-save-video   : otomatik kaydı tamamen kapatır
AUTO_SAVE_OUTPUT_VIDEO: bool = True
# Kaynak videonun yanındaki alt klasör (.env)
OUTPUT_DIR_NAME: str = _env_str("OUTPUT_DIR_NAME", "islenmis")
OUTPUT_NAME_SUFFIX: str = "_islenmis"      # Dosya adına eklenen ek
OUTPUT_VIDEO_EXTENSION: str = ".mp4"
OUTPUT_ADD_TIMESTAMP: bool = True          # Aynı videoyu yeniden işlerken çakışmayı önler
# Kaynak dosya değilse (RTSP / kamera) çıktının yazılacağı klasör adı.
# Çalışma dizini (cwd) altında oluşturulur. (.env)
OUTPUT_STREAM_DIR_NAME: str = _env_str("OUTPUT_STREAM_DIR_NAME", "vip_van_kayitlari")
# Sırayla denenir; açılabilen ilk codec kullanılır. Bazı OpenCV kurulumlarında
# mp4v bulunmaz, bu yüzden tek bir codec'e bel bağlamak kayıt kaybına yol açar.
OUTPUT_FOURCC_CANDIDATES: Tuple[str, ...] = ("mp4v", "avc1", "XVID", "MJPG")
OUTPUT_FPS_FALLBACK: float = 25.0          # Kaynak FPS okunamazsa kullanılır
OUTPUT_FPS_MAX: float = 240.0              # Bozuk metadata'ya karşı üst sınır

# =============================================================================
# GENEL
# =============================================================================
LOG_FORMAT: str = "[%(asctime)s] %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"
FPS_WINDOW: int = 30                       # FPS ortalaması için pencere boyutu
