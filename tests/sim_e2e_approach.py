"""
Uctan uca senaryo: videodaki gibi UZAKTAN gelip YAKLASAN ve donen bir van.

  * Arac cizgiyi kucukken gecer (plaka ~45 px, okunamaz),
  * sonra kameraya yaklasir (plaka ~200 px'e kadar buyur) ve egimi azalir.

Arac tespiti / CLIP sahte; plaka zinciri GERCEK (OpenCV modelleri).
onnxruntime ve paddle zorla devre disi: kullanicinin ortamini taklit eder.
"""

import os
import sys

sys.modules["onnxruntime"] = None   # type: ignore
sys.modules["paddle"] = None        # type: ignore
sys.modules["paddleocr"] = None     # type: ignore
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sim_pipeline import FakeClassifier  # noqa: E402
from src.pipeline import VehicleAnalyticsPipeline  # noqa: E402
from src.plate_reader import PlateReader  # noqa: E402
from src.reporting import print_summary  # noqa: E402
from test_plate_chain_synthetic import make_plate, make_vehicle  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
TOTAL_FRAMES = 110
CROSS_X = 600


class Line:
    def is_crossed(self, prev, cur):
        return prev is not None and prev[0] < CROSS_X <= cur[0]


class ApproachingVan:
    """Kare kare buyuyen, saga ilerleyen ve egimi azalan bir arac uretir."""

    def __init__(self, plate_text, seed=0):
        self.plate = make_plate(plate_text)
        self.rng = np.random.default_rng(seed)
        self.i = -1
        self.frame = None

    def track(self, frame):
        self.i += 1
        t = min(self.i / 90.0, 1.0)
        vehicle_w = int(110 + 790 * t)                   # 110 -> 900 px (plaka ~24 -> ~200 px)
        roll = -35 + 25 * t                              # -35 -> -10 derece
        squeeze = 0.75 + 0.2 * t
        scene = make_vehicle(self.plate, 140, roll, squeeze, self.rng)
        vehicle = cv2.resize(scene, (vehicle_w, int(vehicle_w * 420 / 640)),
                             interpolation=cv2.INTER_AREA)
        x1 = 480 + 9 * self.i
        y1 = 200
        x2, y2 = min(x1 + vehicle.shape[1], FRAME_W), min(y1 + vehicle.shape[0], FRAME_H)
        frame[:] = 120
        frame[y1:y2, x1:x2] = vehicle[: y2 - y1, : x2 - x1]
        return [{"track_id": 77, "bbox": (x1, y1, x2, y2),
                 "class_name": "car", "conf": 0.9}]


def run(plate_text, seed):
    detector = ApproachingVan(plate_text, seed)
    reader = PlateReader()
    pipeline = VehicleAnalyticsPipeline(detector, FakeClassifier(), Line(), reader,
                                        extended_plate_search=True)
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    for _ in range(TOTAL_FRAMES):
        pipeline.process_frame(frame)
    pipeline.finalize_all()
    return pipeline


if __name__ == "__main__":
    ok = 0
    plates = ("34ABC123", "06FK4521", "16RT287")
    for seed, text in enumerate(plates):
        pipe = run(text, seed)
        track = pipe.all_results()[0]
        hit = track.plate_text == text
        ok += hit
        print(f"{'OK ' if hit else 'XX '} gercek={text} okunan={track.plate_text} "
              f"karar={track.decision_label()} deneme={track.plate_attempts} "
              f"adaylar={[t for t, _ in track.plate_candidates]}")
    print_summary(pipe)
    print(f"SONUC: {ok}/{len(plates)}")
    sys.exit(0 if ok == len(plates) else 1)
