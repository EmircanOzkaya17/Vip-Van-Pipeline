<div align="center">

# 🚐 VIP Van Pipeline

**Video akışında VIP / yolcu vanlarını tespit eden, takip eden, sınıflandıran ve plakasını okuyan uçtan uca görüntü işleme hattı**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-6E40C9)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?logo=opencv&logoColor=white)
![Plaka](https://img.shields.io/badge/Plaka-TR%20format-E30A17)

<img src="docs/images/02_vip_van_tespit.jpg" alt="VIP van tespiti ve plaka okuma" width="100%">

</div>

---

## İçindekiler

- [Ne yapar?](#ne-yapar)
- [Nasıl çalışır?](#nasıl-çalışır)
- [Örnek çıktılar](#örnek-çıktılar)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Yapılandırma](#yapılandırma)
- [Çıktılar](#çıktılar)
- [Testler ve simülasyonlar](#testler-ve-simülasyonlar)
- [Proje yapısı](#proje-yapısı)
- [Modeller ve lisanslar](#modeller-ve-lisanslar)

---

## Ne yapar?

Sabit bir kameradan gelen video (dosya, RTSP veya kamera indeksi) üzerinde:

| Adım | Ne olur | Kullanılan araç |
|---|---|---|
| 1. Tespit + takip | `car`, `bus`, `truck` sınıfındaki araçlar bulunur ve her birine kalıcı bir ID verilir | Ultralytics YOLO11 + BoT-SORT |
| 2. Sınıf oylaması | İlk 7 karede en çok çıkan COCO sınıfı araca atanır | Çoğunluk oyu |
| 3. Sanal çizgi | Araç kutusunun merkezi **TRIGGER LINE** çizgisini beklenen yönde geçtiğinde araç tetiklenir | Segment kesişimi |
| 4. Araç tipi | Geçişten sonraki 2. karede araç kesiti, pozitif/negatif prompt gruplarıyla sıfır atışlı (zero-shot) sınıflandırılır | OpenCLIP ViT-B-32 |
| 5. Plaka okuma | Yalnızca **hedef araçlarda** (VIP van) plaka tespit edilir, düzeltilir ve okunur | YOLOv9-t (plaka) + fast-plate-ocr CCT-S v2 |
| 6. Oylama | Birden çok karedeki okumalar tam metin ve **karakter bazında** oylanarak tek plakaya indirilir | TR plaka formatı doğrulaması |
| 7. Rapor | Anotasyonlu video, CSV ve konsolda özet tablo üretilir | OpenCV, CSV |

> **Öne çıkan nokta:** Plaka zinciri varsayılan olarak **yalnızca OpenCV DNN** ile çalışır. `onnxruntime`, `open-image-models` veya `PaddleOCR` kurulu olmasa da plaka okunur; bu da Windows'taki DLL çakışmalarını tamamen ortadan kaldırır.

---

## Nasıl çalışır?

```mermaid
flowchart LR
    A[Video / RTSP / Kamera] --> B[YOLO11 tespit<br/>+ BoT-SORT takip]
    B --> C{Sanal çizgi<br/>geçildi mi?}
    C -- hayır --> B
    C -- evet --> D[OpenCLIP<br/>geçiş + 2. kare]
    D --> E{pos ≥ 0.80<br/>margin ≥ 0.25?}
    E -- hayır --> F[non-target<br/>takip biter]
    E -- evet --> G[VIP VAN]
    G --> H[Plaka tespiti<br/>YOLOv9-t 384]
    H --> I[Geometri süzgeci<br/>+ perspektif düzeltme]
    I --> J[fast-plate-ocr<br/>CCT-S v2]
    J --> K[TR format doğrulama<br/>+ karakter düzeltme]
    K --> L[Çoğunluk / karakter<br/>oylaması]
    L --> M[(Video + CSV<br/>+ özet tablo)]
```

### Plaka zincirindeki sağlamlaştırmalar

- **Tampon (buffer):** Sınıflandırma kararı geçiş+2. karede verildiği için önceki karelerin kesitleri saklanır; karar olumlu çıkarsa geriye dönük işlenir. Böylece 7 karenin tamamı değerlendirilir.
- **Geometri süzgeci:** Jant, far ve tampon aralığı gibi sahte plaka kutuları en/boy oranı, araç içindeki konum ve genişlik oranıyla elenir (TR plakası 52×11 cm → oran ≈ 4.7; bir plaka hiçbir açıda dikey kutu üretemez).
- **Yakınlaştırmalı ikinci geçiş:** İlk taramada plaka bulunamazsa aracın alt bölgesi üst üste binen 3 döşemeye bölünür ve her biri ayrı taranır.
- **Bölge iyileştirme:** Eğik plakada eksene hizalı kutunun içindeki gerçek plaka gövdesi izole edilir ve perspektif dönüşümü ile yatay hale getirilir.
- **Uzatılmış arama:** Çizgi kadrajın girişinde olduğundan araç ilk karelerde uzak ve küçüktür. İlk 7 kare geçerli plaka üretmezse araç kameraya yaklaştıkça her 3 karede bir denemeye devam edilir (en fazla 40 deneme).
- **Karakter oylaması:** `34ABC123 / 34ABC128 / 34A8C123` gibi tek karakterde titreyen okumalar konum konum oylanarak `34ABC123`'e indirgenir.

---

## Örnek çıktılar

### 1) Araç sanal çizgiye yaklaşıyor

Sarı kutular karar bekleyen araçlardır. Mavi çizgi **TRIGGER LINE**'dır.

<img src="docs/images/01_yaklasma_cizgi_oncesi.jpg" alt="Araç çizgiye yaklaşıyor" width="100%">

### 2) Çizgi geçildi → VIP VAN olarak sınıflandırıldı

Yeşil kutu hedef aracı gösterir. Alt etikette OpenCLIP pozitif skoru (`p`), marj (`m`) ve plaka durumu yer alır.

<table>
  <tr>
    <td width="62%"><img src="docs/images/02_vip_van_tespit.jpg" alt="VIP van tespit edildi"></td>
    <td width="38%"><img src="docs/images/03_vip_van_yakin.jpg" alt="Yakın görünüm"></td>
  </tr>
  <tr>
    <td align="center"><sub>Tam kare</sub></td>
    <td align="center"><sub>Yakın görünüm: <code>ID:77 car | VIP VAN</code>, <code>p=0.96 m=0.93</code></sub></td>
  </tr>
</table>

### 3) Konsol özeti

Video bittiğinde hedef araçlar ve plaka zincirinin aşama dökümü basılır:

<img src="docs/images/04_ozet_tablosu.png" alt="Konsol özet tablosu" width="720">

| Sütun | Anlamı |
|---|---|
| `POS` / `MARGIN` | OpenCLIP pozitif grup olasılığı ve pozitif–negatif farkı |
| `OCR` | Nihai plakanın oylama güveni |
| `KARAR` | Sınıflandırmanın verildiği kare (`+2` = planlanan, `kapanis` = track kaybolurken) |
| `DENEME` / `ADAY` | Plaka zincirinin kaç kez çalıştığı / kaç geçerli okuma ürettiği |

---

## Kurulum

### Gereksinimler

- Python **3.10+**
- (İsteğe bağlı) CUDA destekli GPU — CPU'da da çalışır

### Adımlar

```bash
# 1) Depoyu klonlayın
git clone https://github.com/<kullanici-adi>/vip_van_pipeline.git
cd vip_van_pipeline

# 2) Sanal ortam oluşturun
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3) Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4) Ortam dosyasını oluşturun
# Windows
copy example.env .env
# Linux / macOS
cp example.env .env
```

> **GPU kullanacaksanız** `torch`'u önce [pytorch.org](https://pytorch.org/get-started/locally/) üzerindeki CUDA sürümünüze uygun komutla kurun, ardından `requirements.txt`'i yükleyin.

`yolo11m.pt` repoda bulunmaz; Ultralytics ilk çalıştırmada otomatik indirir. Plaka modelleri (`models/*.onnx`) repoyla birlikte gelir.

---

## Kullanım

Repoda örnek video bulunmaz. İşlemek istediğiniz videoyu `assets/` klasörüne koyun (bu klasörün içeriği `.gitignore` ile repo dışında tutulur).

```bash
# Temel kullanım: video işlenir, çıktı kaynağın yanındaki islenmis/ klasörüne yazılır
python main.py --source assets/ornek_video.mp4

# Canlı izleme + CSV raporu
python main.py --source assets/ornek_video.mp4 --show --csv

# GPU ile (FP16 yalnızca cihaz açıkça verildiğinde devreye girer)
python main.py --source assets/ornek_video.mp4 --device cuda:0

# Sanal çizgiyi ilk kare üzerinde fareyle seçme
python main.py --source assets/ornek_video.mp4 --pick-line

# Sanal çizgiyi koordinatla verme (x1,y1,x2,y2 piksel)
python main.py --source assets/ornek_video.mp4 --line 110,570,420,280

# RTSP kaynağı
python main.py --source "rtsp://kullanici:sifre@192.168.1.10:554/stream1" --show

# Plaka kesitlerini hata ayıklama için diske yazma
python main.py --source assets/ornek_video.mp4 --debug-plates plaka_debug --verbose

# Tek bir görüntü üzerinde plaka zincirini test etme (video işlenmez)
python main.py --test-ocr plaka.jpg --verbose
```

### Komut satırı argümanları

| Argüman | Açıklama | Varsayılan |
|---|---|---|
| `--source` | Video dosyası, RTSP adresi veya kamera indeksi | — |
| `--test-ocr` | Tek görüntüde plaka zinciri tanısı | — |
| `--model` | YOLO model yolu | `.env` → `yolo11m.pt` |
| `--device` | `cuda`, `cuda:0`, `cpu` (boş = otomatik) | otomatik |
| `--imgsz` | YOLO çıkarım boyutu | `960` |
| `--conf` | Tespit güven eşiği | `0.35` |
| `--line` | Sanal çizgi: `x1,y1,x2,y2` | config referans çizgisi |
| `--pick-line` | Çizgiyi fareyle seç | kapalı |
| `--show` | Pencerede göster (`q` / `Esc` ile çıkış) | kapalı |
| `--save-video [YOL]` | Anotasyonlu videonun dosya/dizin yolu | otomatik kayıt |
| `--no-save-video` | Otomatik video kaydını kapat | — |
| `--output-dir` | Tüm çıktıların yazılacağı klasör | kaynağın yanındaki `islenmis/` |
| `--csv [YOL]` | Sonuç CSV'si (değersiz: video ile aynı isim) | — |
| `--no-plate` | Plaka okumayı kapat | — |
| `--extended-plate` / `--no-extended-plate` | Uzatılmış plaka araması | açık |
| `--debug-plates` | Plaka kesitlerinin yazılacağı dizin | — |
| `--max-frames` | İşlenecek en fazla kare (`0` = sınırsız) | `0` |
| `--verbose` | Ayrıntılı log | kapalı |

---

## Yapılandırma

Ayarlar iki katmandadır:

**1. Ortama bağlı değerler → `.env`** (şablon: [`example.env`](example.env))
Öncelik sırası: *gerçek ortam değişkeni > `.env` > koddaki varsayılan*. Boş bırakılan değerler varsayılana düşer.

| Değişken | Açıklama |
|---|---|
| `YOLO_MODEL_PATH` | Araç dedektörü ağırlıkları (`yolo11n/m/x.pt`) |
| `TRACKER_CONFIG` | `botsort.yaml` veya `bytetrack.yaml` |
| `CLIP_MODEL_NAME`, `CLIP_PRETRAINED` | OpenCLIP mimarisi ve ön eğitim ağırlıkları |
| `PLATE_DETECTOR_BACKEND` | `auto` / `opencv` / `onnxruntime` |
| `PLATE_OCR_ENGINES` | `fastplate` veya `fastplate,paddle` |
| `PLATE_DETECTOR_ONNX`, `PLATE_OCR_FAST_ONNX` | Model yolları (boş = `models/`) |
| `PADDLEX_CACHE_HOME` | Kullanıcı klasöründe Türkçe karakter varsa ASCII bir önbellek yolu |
| `OUTPUT_DIR_NAME`, `OUTPUT_STREAM_DIR_NAME` | Çıktı klasör adları |

**2. Algoritma eşikleri → [`src/config.py`](src/config.py)**
Kodla birlikte sürümlenmesi gereken tüm eşikler tek dosyadadır ve her birinin gerekçesi yanında açıklanmıştır. Sık ayarlananlar:

| Parametre | Değer | Etkisi |
|---|---|---|
| `FINAL_POSITIVE_THRESHOLD` | `0.80` | Hedef sayılmak için minimum pozitif olasılık |
| `FINAL_MARGIN_THRESHOLD` | `0.25` | Pozitif–negatif fark eşiği |
| `CLIP_FRAME_OFFSETS` | `(2,)` | Sınıflandırmanın yapıldığı geçiş sonrası kare(ler) |
| `POSITIVE_PROMPTS` / `NEGATIVE_PROMPTS` | — | Hedef / hedef olmayan araç tipi tarifleri |
| `LINE_REFERENCE_*` | `704×396` referansında | Varsayılan sanal çizgi (çözünürlüğe göre otomatik ölçeklenir) |
| `LINE_CROSS_DIRECTION` | `neg_to_pos` | Geçiş yönü (`any` ile çift yön) |
| `PLATE_FRAME_COUNT` | `7` | Geçişten sonraki temel plaka penceresi |
| `PLATE_EXTENDED_MAX_ATTEMPTS` | `40` | Track başına plaka denemesi tavanı |

> Farklı bir araç tipini hedeflemek için çoğu zaman yalnızca `POSITIVE_PROMPTS`, `NEGATIVE_PROMPTS` ve `TARGET_VEHICLE_TYPE` değerlerini değiştirmek yeterlidir; model yeniden eğitilmez.

---

## Çıktılar

Hiçbir bayrak verilmese bile işlenen video kaynak videonun yanındaki alt klasöre zaman damgasıyla kaydedilir:

```
assets/giris.mp4
  └─> assets/islenmis/giris_islenmis_20261010_174644.mp4
      assets/islenmis/giris_islenmis_20261010_174644.csv   (--csv verildiyse)
```

- **Video:** Tüm araç kutuları, track ID'leri, sanal çizgi, hedef araçların skorları ve plakası çizilir. Codec sırayla denenir: `mp4v → avc1 → XVID → MJPG`.
- **CSV:** Her track için sınıf, çizgi geçiş karesi, OpenCLIP skorları, karar, plaka ve OCR güveni.
- **Konsol:** Özet, hedef araç tablosu ve plaka zinciri aşama dökümü (plaka okunamadığında darboğazın nerede olduğunu gösterir).

RTSP / kamera kaynaklarında çıktılar çalışma dizinindeki `vip_van_kayitlari/` klasörüne yazılır.

---

## Testler ve simülasyonlar

Model veya video gerektirmeyen hızlı doğrulamalar:

```bash
# TR plaka formatı düzeltme/doğrulama ve oylama birim testleri
python tests/test_plate_text.py

# Sahte bileşenlerle pipeline akışı (kaçırılan kare, erken kaybolma vb. senaryolar)
python tests/sim_pipeline.py

# Sentetik plakalarla GERÇEK plaka zinciri (farklı genişlik / açı / yana dönüş)
python tests/test_plate_chain_synthetic.py

# Uzaktan gelip yaklaşan van senaryosu (uzatılmış arama, uçtan uca)
python tests/sim_e2e_approach.py
```

Plaka modellerini resmi kaynaklarından yeniden üretmek için (gereksinim: `pip install onnx`):

```bash
python tools/prepare_plate_models.py
```

---

## Proje yapısı

```
vip_van_pipeline/
├── main.py                     # Giriş noktası
├── requirements.txt
├── example.env                 # .env şablonu
├── assets/                     # Giriş videolarınız (repoda boş, .gitkeep)
├── islenmis/                   # İşlenmiş çıktılar (repoda boş, .gitkeep)
├── docs/images/                # README görselleri
├── models/
│   ├── plate_det_yolov9t384_raw.onnx   # Plaka dedektörü (NMS öncesi kesilmiş)
│   ├── plate_ocr_cct_s_v2_cv.onnx      # Plaka OCR (OpenCV uyumlu)
│   └── KAYNAK_VE_LISANS.txt
├── src/
│   ├── cli.py                  # Argümanlar
│   ├── config.py               # .env + tüm eşikler
│   ├── video_runner.py         # Video akışı, bileşen kurulumu
│   ├── pipeline.py             # Kare bazlı orkestrasyon, track yaşam döngüsü
│   ├── vehicle_detector.py     # YOLO11 + tracker
│   ├── vehicle_classifier.py   # OpenCLIP zero-shot sınıflandırma
│   ├── virtual_line.py         # Sanal çizgi ve geçiş yönü
│   ├── track.py                # TrackInfo veri yapısı
│   ├── plate_reader.py         # Plaka zinciri orkestrasyonu
│   ├── plate_detector.py       # Plaka dedektörü (OpenCV DNN / onnxruntime)
│   ├── plate_image.py          # Kesit, iyileştirme, perspektif düzeltme
│   ├── plate_ocr_fast.py       # fast-plate-ocr (OpenCV DNN)
│   ├── plate_ocr.py            # OCR motorları (PaddleOCR dahil)
│   ├── plate_text.py           # TR format doğrulama + oylama
│   ├── ocr_selftest.py         # --test-ocr tanı aracı
│   ├── visualization.py        # Çizimler, fareyle çizgi seçimi
│   ├── reporting.py            # CSV + konsol özeti
│   ├── video_io.py / output_paths.py / file_utils.py / bbox.py / dnn_utils.py
│   └── __init__.py
├── tests/
└── tools/
    └── prepare_plate_models.py
```

---

## Modeller ve lisanslar

| Bileşen | Kaynak | Lisans |
|---|---|---|
| Araç tespiti / takip | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| Araç tipi sınıflandırma | [OpenCLIP](https://github.com/mlfoundations/open_clip) (`ViT-B-32`, `laion2b_s34b_b79k`) | MIT |
| Plaka tespiti | [open-image-models](https://github.com/ankandrew/open-image-models) YOLOv9-t 384 | MIT |
| Plaka OCR | [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) CCT-S v2 (global) | MIT |
| (İsteğe bağlı) OCR | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 |

`models/` altındaki ONNX dosyaları yukarıdaki MIT lisanslı modellerin OpenCV DNN ile çalışacak şekilde yamalanmış hâlleridir; ayrıntılar `models/KAYNAK_VE_LISANS.txt` ve `tools/prepare_plate_models.py` içindedir.

> ⚠️ Ultralytics YOLO11 **AGPL-3.0** lisanslıdır. Projeyi kapalı kaynak bir üründe veya ağ üzerinden hizmet olarak kullanacaksanız Ultralytics'in lisans koşullarını inceleyin.

> 🔒 Bu proje plaka gibi kişisel veri işler. Gerçek kamera görüntüleriyle kullanırken yürürlükteki veri koruma mevzuatına (ör. KVKK) uygun hareket edin.
