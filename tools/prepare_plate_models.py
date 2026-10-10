#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plaka modellerini OpenCV DNN ile calisacak hale getirir (tek seferlik).

Proje models/ klasorunde bu dosyalar HAZIR gelir; bu betik yalnizca onlari
resmi kaynaklarindan yeniden uretmek (dogrulamak / guncellemek) icindir.
Gereksinim: pip install onnx   (onnxruntime varsa ek dogrulama yapilir)

Uretilen dosyalar:
  plate_det_yolov9t384_raw.onnx
      open-image-models "yolo-v9-t-384-license-plate-end2end" modelinin
      NonMaxSuppression oncesinde kesilmis hali. OpenCV DNN, ONNX
      NonMaxSuppression operatorunu desteklemedigi icin end2end model
      cv2.dnn ile ACILAMAZ; NMS'i kod tarafinda cv2.dnn.NMSBoxes yapar.
  plate_ocr_cct_s_v2_cv.onnx
      fast-plate-ocr "cct-s-v2-global-model" (65 ulke, Turkiye dahil).
      Iki yama: (1) giris uint8 -> float32 (OpenCV 4/5 uyumu),
      (2) her Tanh oncesine Clip(-15, 15): OpenCV 5.0'in Tanh cekirdegi
      buyuk girdilerde NaN uretiyor; tanh(15) = 1 - 1e-13 oldugu icin
      sonuc matematiksel olarak degismez,
      (3) yalnizca "plate" cikisi birakilir (ulke basligi atilir).

Kaynaklar (MIT lisansli):
  https://github.com/ankandrew/open-image-models
  https://github.com/ankandrew/fast-plate-ocr

Kullanim:  python tools/prepare_plate_models.py
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

DET_URL = ("https://github.com/ankandrew/open-image-models/releases/download/"
           "assets/yolo-v9-t-384-license-plates-end2end.onnx")
DET_CUT_TENSOR = "/model/model.22/Concat_5_output_0"   # NMS oncesi (1, 5, N)
DET_OUT = MODELS_DIR / "plate_det_yolov9t384_raw.onnx"

OCR_URL = ("https://github.com/ankandrew/cnn-ocr-lp/releases/download/"
           "arg-plates/cct_s_v2_global.onnx")
OCR_OUT = MODELS_DIR / "plate_ocr_cct_s_v2_cv.onnx"
TANH_CLIP = 15.0


def _download(url: str, target: Path) -> None:
    print(f"indiriliyor: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def build_detector(onnx, work: Path) -> None:
    src = work / "det_end2end.onnx"
    _download(DET_URL, src)
    model = onnx.load(str(src))
    cut = onnx.utils.Extractor(model).extract_model(["images"], [DET_CUT_TENSOR])
    onnx.checker.check_model(cut)
    onnx.save(cut, str(DET_OUT))
    print(f"yazildi: {DET_OUT} ({DET_OUT.stat().st_size / 1e6:.1f} MB)")


def build_ocr(onnx, work: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    src = work / "ocr.onnx"
    _download(OCR_URL, src)
    model = onnx.load(str(src))
    graph = model.graph

    # (1) uint8 giris -> float32, sabit batch=1; ilk Cast etkisiz hale gelir
    graph_input = graph.input[0]
    graph_input.type.tensor_type.elem_type = TensorProto.FLOAT
    for dim, value in zip(graph_input.type.tensor_type.shape.dim, (1, 64, 128, 3)):
        dim.ClearField("dim_param")
        dim.dim_value = value
    first = graph.node[0]
    if first.op_type == "Cast" and first.input[0] == graph_input.name:
        first.op_type = "Identity"
        del first.attribute[:]
    for output in graph.output:
        output.type.tensor_type.shape.dim[0].ClearField("dim_param")
        output.type.tensor_type.shape.dim[0].dim_value = 1

    # (2) Tanh girdilerini [-15, 15] araligina kirp (OpenCV 5 NaN hatasi)
    graph.initializer.extend([
        numpy_helper.from_array(np.array(-TANH_CLIP, np.float32), "tanh_clip_min"),
        numpy_helper.from_array(np.array(TANH_CLIP, np.float32), "tanh_clip_max"),
    ])
    nodes = []
    for node in graph.node:
        if node.op_type == "Tanh":
            clipped = node.input[0] + "__clip"
            nodes.append(helper.make_node(
                "Clip", [node.input[0], "tanh_clip_min", "tanh_clip_max"],
                [clipped], name=node.name + "__clip"))
            node.input[0] = clipped
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)

    # (3) Yalnizca "plate" cikisini birak: ulke ("region") basligina ihtiyac
    #     yok ve OpenCV 5 tum cikislarin birlikte istenmesini zorunlu kiliyor.
    model = onnx.utils.Extractor(model).extract_model(["input"], ["plate"])

    onnx.checker.check_model(model)
    onnx.save(model, str(OCR_OUT))
    print(f"yazildi: {OCR_OUT} ({OCR_OUT.stat().st_size / 1e6:.1f} MB)")
    _verify_ocr(src)


def _verify_ocr(original: Path) -> None:
    """OpenCV ciktisini (varsa) onnxruntime referansiyla karsilastirir."""
    import cv2

    net = cv2.dnn.readNetFromONNX(np.fromfile(str(OCR_OUT), np.uint8))
    sample = np.random.default_rng(0).integers(0, 255, (1, 64, 128, 3))
    net.setInput(sample.astype(np.float32))
    ours = net.forward().reshape(10, -1)
    if not np.isfinite(ours).all():
        print("UYARI: OpenCV ciktisi sonlu degil!")
        return
    try:
        import onnxruntime as ort
    except Exception:
        print(f"OpenCV {cv2.__version__}: cikti sonlu (onnxruntime yok, karsilastirma atlandi)")
        return
    reference = ort.InferenceSession(str(original)).run(
        ["plate"], {"input": sample.astype(np.uint8)})[0].reshape(10, -1)
    print(f"OpenCV {cv2.__version__} / onnxruntime en buyuk fark: "
          f"{float(np.abs(ours - reference).max()):.2e}")


def main() -> int:
    try:
        import onnx
    except ImportError:
        print("Bu betik icin 'onnx' paketi gerekir: pip install onnx")
        return 1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        build_detector(onnx, work)
        build_ocr(onnx, work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
