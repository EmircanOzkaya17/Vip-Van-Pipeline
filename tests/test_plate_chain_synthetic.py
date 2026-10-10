"""
Plaka zincirinin (dedektor + OCR + duzeltme) GERCEK modellerle uctan uca testi.

Sentetik TR plakalari uretilir; videodaki kosullara benzer sekilde egilir
(duzlemde donme + yana donus), kucultulur, bulaniklastirilir ve JPEG ile
sikistirilir. onnxruntime ve paddle ZORLA devre disi birakilir: zincirin
yalnizca OpenCV ile calistigi dogrulanir.

Calistirma:  python tests/test_plate_chain_synthetic.py
"""

import os
import sys

sys.modules["onnxruntime"] = None          # type: ignore  # import edilemesin
sys.modules["paddle"] = None               # type: ignore
sys.modules["paddleocr"] = None            # type: ignore
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.plate_reader import PlateReader  # noqa: E402

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "DejaVuSansCondensed-Bold.ttf",
)
PLATES = ("34ABC123", "06FK4521", "35D12345", "16RT287", "07AB1234", "41KL352")


def _font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_plate(text):
    """52x11 cm oranli (4.7) TR plakasi: mavi TR seridi + siyah karakterler."""
    width, height = 520, 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width - 1, height - 1], outline="black", width=5)
    draw.rectangle([5, 5, 52, height - 6], fill=(0, 51, 153))
    draw.text((12, 62), "TR", font=_font(26), fill="white")
    province, rest = text[:2], text[2:]
    letters = "".join(c for c in rest if c.isalpha())
    digits = rest[len(letters):]
    label = f"{province} {letters} {digits}"
    size = 84
    while True:                                   # metni plakaya sigdir
        font = _font(size)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        if right - left <= width - 58 - 24 or size <= 30:
            break
        size -= 2
    x = 58 + (width - 58 - (right - left)) / 2 - left
    y = (height - (bottom - top)) / 2 - top
    draw.text((x, y), label, font=font, fill="black")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def make_vehicle(plate, plate_width, roll_deg, yaw_squeeze, rng):
    """Plakayi koyu bir arac onune perspektifle yerlestirir."""
    vehicle = np.full((420, 640, 3), (38, 38, 42), np.uint8)
    cv2.rectangle(vehicle, (150, 120), (500, 230), (25, 25, 28), -1)       # gril
    for y in range(130, 225, 14):
        cv2.line(vehicle, (160, y), (490, y), (90, 90, 95), 3)
    cv2.circle(vehicle, (325, 175), 30, (170, 170, 175), 3)                # amblem
    cv2.rectangle(vehicle, (0, 330), (640, 420), (60, 60, 64), -1)         # tampon alti

    h, w = plate.shape[:2]
    pw = plate_width * yaw_squeeze
    ph = plate_width / 4.7
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-pw / 2, -ph / 2], [pw / 2, -ph / 2],
                      [pw / 2 * 0.97, ph / 2], [-pw / 2 * 0.97, ph / 2]])
    angle = np.deg2rad(roll_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = dst @ rot.T + np.float32([320, 280])
    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(plate, matrix, (640, 420))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (640, 420))
    vehicle[mask > 0] = warped[mask > 0]

    vehicle = cv2.GaussianBlur(vehicle, (0, 0), 0.9)
    noise = rng.normal(0, 4, vehicle.shape)
    vehicle = np.clip(vehicle.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", vehicle, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def main():
    rng = np.random.default_rng(7)
    reader = PlateReader()
    print(f"OpenCV {cv2.__version__} | dedektor={reader.detector.backend_name} | "
          f"OCR={'+'.join(n for n, _ in reader.engines)}")
    settings = [(width, roll, squeeze)
                for width in (140, 100, 75)
                for roll, squeeze in ((0, 1.0), (-15, 0.9), (-30, 0.8))]
    total = correct = 0
    for width, roll, squeeze in settings:
        hits = []
        for text in PLATES:
            vehicle = make_vehicle(make_plate(text), width, roll, squeeze, rng)
            attempt = reader.read_plate(vehicle)
            hits.append(attempt.plate_text == text)
            if attempt.plate_text != text:
                print(f"   kacirildi: {text} -> {attempt.describe()}")
        total += len(hits)
        correct += sum(hits)
        print(f"genislik={width:>3}px aci={roll:>4}deg yana={squeeze:.1f}: "
              f"{sum(hits)}/{len(hits)} dogru")
    print(f"TOPLAM: {correct}/{total} ({100.0 * correct / total:.0f}%)")
    return 0 if correct / total >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
