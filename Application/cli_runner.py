import sys
from pathlib import Path

# Thêm đường dẫn project
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from Runtime.engine import RuntimeEngine
from Runtime.model import VSADModel

def main():
    # 1. Setup Runtime Model
    release_dir = ROOT / "Runtime/Model/VSAD/0.0.3"
    vsad = VSADModel(release_dir)

    # 2. Khởi tạo Engine
    engine = RuntimeEngine(ROOT / "Runtime/Registry")

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
            frame = vsad.infer(text)
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
