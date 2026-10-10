"""
OpenCLIP ile zero-shot araç tipi sınıflandırması.

torch / open_clip / PIL yalnızca bu sınıf kullanıldığında içe aktarılır.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .config import (
    CLIP_LOGIT_SCALE,
    CLIP_MIN_CROP_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_PROMPT_TEMPLATES,
    CLIP_SCORE_MODE,
    FINAL_MARGIN_THRESHOLD,
    FINAL_POSITIVE_THRESHOLD,
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
)

LOGGER = logging.getLogger(__name__)


class VehicleTypeClassifier:
    """
    OpenCLIP ile zero-shot araç tipi sınıflandırması.

    Pozitif ve negatif prompt gruplarının metin gömüleri (embedding) bir kez
    hesaplanır ve önbelleğe alınır; her frame için yalnızca görüntü gömüsü
    çıkarılır. Bu sayede gerçek zamanlı çalışma korunur.

    Prompt grupları parametre olarak verilebilir; varsayılanlar config'dedir.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        pretrained: str = CLIP_PRETRAINED,
        device: Optional[str] = None,
        score_mode: str = CLIP_SCORE_MODE,
        positive_prompts: Sequence[str] = POSITIVE_PROMPTS,
        negative_prompts: Sequence[str] = NEGATIVE_PROMPTS,
    ) -> None:
        try:
            import torch
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "open_clip_torch kurulu degil. Kurulum: pip install open_clip_torch"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_mode = score_mode

        LOGGER.info("OpenCLIP yukleniyor: %s / %s", model_name, pretrained)
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise RuntimeError(f"OpenCLIP modeli yuklenemedi: {exc}") from exc

        self.model.eval()
        self.positive_prompts = list(positive_prompts)
        self.negative_prompts = list(negative_prompts)
        self.text_features = self._encode_prompts(
            self.positive_prompts + self.negative_prompts
        )
        self.positive_count = len(self.positive_prompts)

    # ------------------------------------------------------------------ #
    def _encode_prompts(self, prompts: Sequence[str]):
        """
        Her prompt için şablon ortalaması (prompt ensembling) uygulanmış,
        L2-normalize edilmiş metin gömülerini üretir.
        """
        torch = self.torch
        features = []
        with torch.no_grad():
            for prompt in prompts:
                texts = [tpl.format(prompt) for tpl in CLIP_PROMPT_TEMPLATES]
                tokens = self.tokenizer(texts).to(self.device)
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat.mean(dim=0)
                feat = feat / feat.norm()
                features.append(feat)
        return torch.stack(features)

    # ------------------------------------------------------------------ #
    def _encode_image(self, crop_bgr: np.ndarray):
        """BGR numpy görüntüsünü normalize edilmiş CLIP gömüsüne çevirir."""
        from PIL import Image

        torch = self.torch
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    # ------------------------------------------------------------------ #
    def score_crop(self, crop_bgr: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Tek bir araç kesiti için pozitif/negatif grup skorlarını hesaplar.

        Dönüş: {"positive_mean": p, "negative_mean": n, "margin": p - n}
        Hata / geçersiz crop durumunda None döner.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < CLIP_MIN_CROP_SIZE or w < CLIP_MIN_CROP_SIZE:
            return None

        try:
            image_features = self._encode_image(crop_bgr)
            similarity = image_features @ self.text_features.T   # (1, N)

            if self.score_mode == "softmax_group_sum":
                # Softmax ile olasılık dağılımı; grup içi olasılıklar toplanır.
                # Böylece skorlar [0, 1] aralığına oturur ve 0.80 / 0.25
                # eşikleri doğrudan yorumlanabilir hale gelir.
                probs = (CLIP_LOGIT_SCALE * similarity).softmax(dim=-1)[0]
                positive_score = float(probs[: self.positive_count].sum().item())
                negative_score = float(probs[self.positive_count:].sum().item())
            else:
                # Ham cosine benzerliklerinin grup ortalaması
                sims = similarity[0]
                positive_score = float(sims[: self.positive_count].mean().item())
                negative_score = float(sims[self.positive_count:].mean().item())

            return {
                "positive_mean": positive_score,
                "negative_mean": negative_score,
                "margin": positive_score - negative_score,
            }
        except Exception as exc:
            LOGGER.warning("OpenCLIP siniflandirma hatasi: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def aggregate(frame_scores: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """
        Seçilen tüm frame skorlarının ortalamasını alarak nihai skorları üretir.

            final_positive = mean(frame_positive_means)
            final_negative = mean(frame_negative_means)
            final_margin   = final_positive - final_negative
        """
        if not frame_scores:
            return None
        final_positive = float(np.mean([s["positive_mean"] for s in frame_scores]))
        final_negative = float(np.mean([s["negative_mean"] for s in frame_scores]))
        return {
            "final_positive": final_positive,
            "final_negative": final_negative,
            "final_margin": final_positive - final_negative,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def is_target_vehicle(final_positive: float, final_margin: float) -> bool:
        """Nihai karar: iki eşik de aynı anda sağlanmalıdır."""
        return (
            final_positive >= FINAL_POSITIVE_THRESHOLD
            and final_margin >= FINAL_MARGIN_THRESHOLD
        )
