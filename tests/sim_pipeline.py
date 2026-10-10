"""Sahte bilesenlerle pipeline akisini simule eder (model gerekmez)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.pipeline import VehicleAnalyticsPipeline


class FakeLine:
    """x=200'u soldan saga gecis."""
    def is_crossed(self, prev, cur):
        return prev is not None and prev[0] < 200 <= cur[0]


class FakeClassifier:
    def score_crop(self, crop):
        return {"positive_mean": 0.965, "negative_mean": 0.035, "margin": 0.93}
    @staticmethod
    def aggregate(scores):
        if not scores:
            return None
        p = float(np.mean([s["positive_mean"] for s in scores]))
        n = float(np.mean([s["negative_mean"] for s in scores]))
        return {"final_positive": p, "final_negative": n, "final_margin": p - n}
    @staticmethod
    def is_target_vehicle(p, m):
        return p >= 0.8 and m >= 0.25


class Attempt:
    def __init__(self, ok):
        self.stage = "ok" if ok else "no_plate_detected"
        self.plate_text = "34ABC123" if ok else None
        self.confidence = 0.9 if ok else None
    @property
    def ok(self):
        return self.plate_text is not None
    def describe(self):
        return self.stage


class FakePlateReader:
    rejected_boxes = 0
    def __init__(self, succeed_from_call=1):
        self.calls = 0
        self.succeed_from_call = succeed_from_call
    def read_plate(self, crop, track_id=0):
        self.calls += 1
        return Attempt(self.calls >= self.succeed_from_call)


class FakeDetector:
    """Arac her frame 15 px saga gider; missing_frames'te tespit yok."""
    def __init__(self, missing_frames=(), last_frame=60):
        self.i = -1
        self.missing = set(missing_frames)
        self.last_frame = last_frame
    def track(self, frame):
        self.i += 1
        if self.i in self.missing or self.i > self.last_frame:
            return []
        x = 100 + 15 * self.i
        return [{"track_id": 77, "bbox": (x - 40, 300, x + 40, 360),
                 "class_name": "car", "conf": 0.9}]


def run(name, detector, reader, frames=120, **kw):
    pipe = VehicleAnalyticsPipeline(detector, FakeClassifier(), FakeLine(), reader, **kw)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(frames):
        pipe.process_frame(frame)
    pipe.finalize_all()
    t = pipe.all_results()[0]
    stages = {k: v for k, v in pipe.stats.items() if k.startswith("plate_stage")}
    print(f"{name:<52} cross={t.cross_frame} target={t.is_target} "
          f"okuyucu_cagrisi={reader.calls if reader else '-'} plaka={t.plate_text} "
          f"dokum={stages or 'YOK'}")


if __name__ == "__main__":
    kw = dict(extended_plate_search=False) if len(sys.argv) < 2 else {}
    # Arac x=100+15i: merkez 200'u i=7'de gecer -> cross_frame=7, offset2 = frame 9
    run("A) Kesintisiz takip", FakeDetector(), FakePlateReader(), **kw)
    run("B) Gecis+2. frame'de tespit yok (frame 9)", FakeDetector(missing_frames={9}), FakePlateReader(), **kw)
    run("C) Gecisten hemen sonra kayboldu (frame>8 yok)", FakeDetector(last_frame=8), FakePlateReader(), **kw)
    run("D) Plaka ilk 7 karede okunamiyor, sonra okunuyor", FakeDetector(), FakePlateReader(succeed_from_call=9), **kw)
