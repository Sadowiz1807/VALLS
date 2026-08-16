import json
import sys
from pathlib import Path

# Thêm đường dẫn project
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TRANSFORMER_ROOT = Path(r"C:\Users\ASUS\Transformer")
if TRANSFORMER_ROOT.is_dir():
    sys.path.insert(0, str(TRANSFORMER_ROOT))

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer
import importlib

from Runtime.engine import RuntimeEngine

def main():
    # 1. Setup Runtime Model
    release_dir = ROOT / "Runtime/Model/VSAD/0.0.3"
    config = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(release_dir / "tokenizer.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mods = {name: importlib.import_module(f"Multi-task transformer.{name}") for name in ("dataset", "model", "training")}
    model = mods["model"].build_model(config).to(device)
    load_model(model, str(release_dir / "VSAD.safetensors"), strict=True, device=str(device))
    model.eval()

    # 2. Khởi tạo Engine
    engine = RuntimeEngine(ROOT / "Runtime/Registry")

    def infer(text: str) -> dict:
        record = {
            "sample_id": "cli", "dialogue_id": "cli", "turn_id": 0,
            "context": [], "state": {},
            "metadata": {"language_mode": "MIXED", "locale": "vi-VN", "asr_noise": "CLEAN", "source_dataset": "cli"},
            "current_text": text,
            "target": {"act": "UNSUPPORTED", "goal": None, "parameters": {}},
            "gold_response_text": "Placeholder",
            "response_metadata": {"phase": "direct_response", "provenance": "cli", "model_generated": False, "owner": "dataset_gold"},
        }
        item = mods["dataset"].MultiTaskDataset([record], tokenizer, config)[0]
        batch = mods["dataset"].collate_batch([item])
        for k in ("input_ids", "input_mask", "token_offsets"):
            batch[k] = batch[k].to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = mods["training"].predict(model, batch, config)
        return mods["training"].assemble_frame(prediction, config)

    print("=== LOCAL VOICE ASSISTANT RUNTIME (PHASE 1 PROTOTYPE) ===")
    print("Sẵn sàng nhận câu lệnh. Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:
        try:
            text = input("User> ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break

            # Infer & Dispatch
            frame = infer(text)
            dispatch_res = engine.dispatch_turn(text, frame)
            
            print(f"Assistant> {dispatch_res.get('response')}")
            if dispatch_res.get("status") == "EXECUTED":
                print(f"  [EXECUTION]: {dispatch_res.get('skill_id')} -> {dispatch_res.get('result', {}).get('args')}")
            elif dispatch_res.get("status") == "AWAITING_CONFIRMATION":
                print(f"  [RISK]: {dispatch_res.get('risk')} (Đang chờ xác nhận lượt sau)")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
