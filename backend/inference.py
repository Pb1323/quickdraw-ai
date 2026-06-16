import json
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch

from ml.model import QuickDrawCNN

from .config import CLASS_NAMES_PATH, DEFAULT_CLASS_PATH, MODEL_PATH
from .image_utils import decode_data_url, preprocess_canvas_image


class PredictorService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_default_classes()
        self.model: QuickDrawCNN | None = None
        self.model_loaded = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    @staticmethod
    def _load_classes(path) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"Class list is invalid: {path}")
        return [str(item) for item in classes]

    def _load_default_classes(self) -> List[str]:
        if DEFAULT_CLASS_PATH.exists():
            return self._load_classes(DEFAULT_CLASS_PATH)
        return []

    def load_model(self) -> bool:
        if CLASS_NAMES_PATH.exists():
            self.class_names = self._load_classes(CLASS_NAMES_PATH)

        if not MODEL_PATH.exists():
            self.model_loaded = False
            return False

        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        saved_classes = checkpoint.get("class_names")
        if saved_classes:
            self.class_names = [str(item) for item in saved_classes]

        if not self.class_names:
            raise RuntimeError("Class names are missing. Train model first.")

        self.model = QuickDrawCNN(num_classes=len(self.class_names)).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_loaded = True
        return True

    def get_classes(self) -> List[str]:
        return list(self.class_names)

    def predict(self, image_b64: str, top_k: int = 5) -> Dict:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Run training first.")

        start = perf_counter()

        image = decode_data_url(image_b64)
        tensor_np, is_blank = preprocess_canvas_image(image)

        if is_blank:
            latency_ms = (perf_counter() - start) * 1000.0
            return {
                "predictions": [],
                "is_blank": True,
                "guessed": False,
                "latency_ms": round(latency_ms, 2),
            }

        tensor = torch.from_numpy(tensor_np).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor)[0].detach().cpu().numpy()
            probs = self._softmax(logits)

        k = min(top_k, len(self.class_names))
        indices = np.argsort(probs)[::-1][:k]
        predictions = [
            {
                "label": self.class_names[idx],
                "prob": round(float(probs[idx]), 6),
            }
            for idx in indices
        ]

        guessed = bool(predictions and predictions[0]["prob"] >= 0.65)
        latency_ms = (perf_counter() - start) * 1000.0

        return {
            "predictions": predictions,
            "is_blank": False,
            "guessed": guessed,
            "latency_ms": round(latency_ms, 2),
        }
