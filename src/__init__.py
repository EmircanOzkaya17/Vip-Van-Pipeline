"""
VIP / yolcu vanı tespit, sınıflandırma, takip ve plaka okuma paketi.

Modüller (bağımlılık yönü yukarıdan aşağıya):
    config             .env + tüm eşik/ayar sabitleri
    file_utils, bbox   alandan bağımsız yardımcılar
    virtual_line       sanal çizgi geometrisi
    track              TrackInfo veri modeli
    vehicle_detector   YOLO + BoT-SORT
    vehicle_classifier OpenCLIP zero-shot sınıflandırma
    plate_text         plaka metni düzeltme / doğrulama / oylama
    dnn_utils          OpenCV DNN model yükleme (unicode güvenli)
    plate_detector     plaka kutusu tespiti (OpenCV DNN) + geometri süzgeci
    plate_image        plaka bölgesi iyileştirme + OCR ön işleme
    plate_ocr_fast     plakaya özel OCR (fast-plate-ocr, OpenCV DNN)
    plate_ocr          PaddleOCR sarmalayıcısı (isteğe bağlı ikinci motor)
    plate_reader       plaka okuma zinciri
    pipeline           frame bazlı ana işlem hattı
    visualization      OpenCV çizim ve çizgi seçimi
    output_paths       çıktı dosya yolları
    video_io           video kaynağı / yazıcı
    reporting          CSV ve konsol özeti
    cli                komut satırı argümanları
    video_runner       uçtan uca video işleme akışı
    ocr_selftest       --test-ocr tanı modu

Ağır kütüphaneler (ultralytics, torch/open_clip, open_image_models, paddleocr)
bilinçli olarak yalnızca ilgili sınıf oluşturulduğunda içe aktarılır.
Plaka zinciri varsayılan olarak yalnızca OpenCV + models/ klasörüyle çalışır.
"""
