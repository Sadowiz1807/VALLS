"""Model adapter: load VSAD 0.0.4 and produce normalized metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

MODEL_PATH = Path(__file__).parent / "Model" / "VSAD" / "0.0.4"


class ModelAdapter:
    def __init__(self, model_dir: Path = MODEL_PATH):
        self.model_dir = model_dir
        self._model = None

    def load(self):
        if not (self.model_dir / "VASD.safetensors").exists():
            raise FileNotFoundError(f"Model artifacts not found in {self.model_dir}")
        from Runtime.model import VSADModel
        self._model = VSADModel(self.model_dir)
        return self

    def infer(self, text: str, context: Optional[Sequence[Dict[str, Any]]] = None,
              state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._model is None:
            self.load()
        raw = self._model.infer(text, context=context, state=state)
        return {
            "act": raw.get("act"),
            "goal": raw.get("goal"),
            "parameters": raw.get("parameters", {}),
            "response": raw.get("response"),
            "model_version": "VSAD-0.0.4",
        }


if __name__ == "__main__":
    adapter = ModelAdapter().load()
    for text in ["mở spotify", "tắt máy", "xin chào"]:
        result = adapter.infer(text)
        print(f"Input: {text}")
        print(f"  act={result['act']}, goal={result['goal']}, params={result['parameters']}")