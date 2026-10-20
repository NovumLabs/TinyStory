from __future__ import annotations

import argparse

from tokenizers import Tokenizer

from config import resolve_device
from generate import generate_text
from model import GPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive story generation.")
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = GPT.from_checkpoint(args.ckpt, device=device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    settings = {"temperature": 0.5, "top_k": 40, "rep_penalty": 1.0,
                "max_new_tokens": 180}
    print("Enter a story opening, or /quit to exit. "
          "Settings: /temp, /topk, /penalty, /tokens.")

    while True:
        try:
            line = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "q", "quit", "exit"):
            break
        if line.startswith("/"):
            name, _, value = line[1:].partition(" ")
            try:
                if name == "temp":
                    settings["temperature"] = float(value)
                elif name == "topk":
                    settings["top_k"] = int(value)
                elif name == "penalty":
                    settings["rep_penalty"] = float(value)
                elif name == "tokens":
                    settings["max_new_tokens"] = int(value)
                else:
                    print(f"Unknown command: /{name}")
                    continue
                print(f"Settings: {settings}")
            except ValueError:
                print(f"Invalid value for /{name}: {value!r}")
            continue
        print()
        print(generate_text(model, tokenizer, line, **settings))


if __name__ == "__main__":
    main()
