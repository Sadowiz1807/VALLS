import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Runtime.model import VSADModel
from Runtime.engine import AgentHarness

def main():
    release_dir = ROOT / "Runtime/Model/VSAD/0.0.3"
    vsad = VSADModel(release_dir)
    harness = AgentHarness(ROOT / "Runtime/Registry")

    print("=== LOCAL AGENTIC VOICE ASSISTANT (MULTI-TURN & HARNESS) ===")
    print("Sẵn sàng nhận lệnh. Gõ 'exit' để thoát.\n")

    while True:
        try:
            text = input("User> ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break

            # Agentic Step: Memory -> Inference -> Tool Dispatch -> State Update
            dispatch_res = harness.step(text, vsad)
            
            print(f"Assistant> {dispatch_res.get('response')}")
            if dispatch_res.get("status") == "EXECUTED":
                print(f"  [EXECUTION]: {dispatch_res.get('skill_id')} -> {dispatch_res.get('result', {}).get('args')}")
            elif dispatch_res.get("status") == "AWAITING_CONFIRMATION":
                print(f"  [RISK]: {dispatch_res.get('risk')} (Đang chờ xác nhận lượt sau)")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
