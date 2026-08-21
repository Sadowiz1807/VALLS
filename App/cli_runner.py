import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Runtime.engine import AgentHarness
from Runtime.model import VSADModel


def run_once(text: str, *, execute: bool = False, model=None):
    model = model or VSADModel(ROOT / "Runtime/Model/VSAD/0.0.4")
    result = AgentHarness(ROOT / "Runtime/Registry", execute=execute).step(text, model)
    print(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--execute", action="store_true", help="Cho phép mở app/browser thật; mặc định dry-run")
    args = parser.parse_args()
    if args.text:
        run_once(args.text, execute=args.execute)
        return

    model = VSADModel(ROOT / "Runtime/Model/VSAD/0.0.4")
    harness = AgentHarness(ROOT / "Runtime/Registry", execute=args.execute)
    while True:
        try:
            text = input("User> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if text.lower() in ("exit", "quit"):
            break
        if text:
            print(json.dumps(harness.step(text, model), ensure_ascii=False))


if __name__ == "__main__":
    main()
